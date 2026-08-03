# coding: utf-8
r"""
HM4SR adapted for MMRec.

File name:
    src/models/hm4sr.py

Class name / model name:
    HM4SR

Why this adapter exists:
    The official HM4SR implementation is a RecBole sequential recommender.
    This file adapts the main HM4SR ideas to the MMRec GeneralRecommender API:
        calculate_loss(interaction): users, pos_items, neg_items
        full_sort_predict(interaction): return [batch_size, n_items] scores

Main components retained:
    1. ID / text / image sequence encoders.
    2. Interactive MoE between ID, text and image sequence representations.
    3. Temporal MoE based on timestamps and adjacent time intervals.
    4. Optional ID contrastive learning and placeholder contrastive learning.

Dataset requirement:
    The model reads train-only chronological user sequences from the .inter file.
    Your dataset yaml already provides:
        inter_file_name: 'baby_temporal.inter' / ...
        USER_ID_FIELD: userID
        ITEM_ID_FIELD: itemID
        TIME_FIELD: timestamp
        inter_splitting_label: x_label in overall.yaml
"""

import os
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class HM4SR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HM4SR, self).__init__(config, dataset)

        self.config = config
        self.dataset_obj = dataset
        self.dataset_name = str(config["dataset"])
        self.data_path = os.path.abspath(str(config["data_path"]))
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, self.dataset_name))

        # ------------------------------------------------------------------
        # Important:
        # overall.yaml contains hidden_size: 4 for raw feature modules.
        # HM4SR therefore uses hm4sr_hidden_size, defaulting to embedding_size.
        # ------------------------------------------------------------------
        self.hidden_size = int(config["hm4sr_hidden_size"]) if "hm4sr_hidden_size" in config else int(config["embedding_size"])
        self.max_seq_length = int(config["hm4sr_max_seq_length"]) if "hm4sr_max_seq_length" in config else 50
        self.n_layers = int(config["hm4sr_n_layers"]) if "hm4sr_n_layers" in config else 2
        self.n_heads = int(config["hm4sr_n_heads"]) if "hm4sr_n_heads" in config else 2
        self.inner_size = int(config["hm4sr_inner_size"]) if "hm4sr_inner_size" in config else 4 * self.hidden_size
        self.hidden_dropout_prob = float(config["hm4sr_hidden_dropout_prob"]) if "hm4sr_hidden_dropout_prob" in config else 0.2
        self.attn_dropout_prob = float(config["hm4sr_attn_dropout_prob"]) if "hm4sr_attn_dropout_prob" in config else 0.2
        self.layer_norm_eps = float(config["hm4sr_layer_norm_eps"]) if "hm4sr_layer_norm_eps" in config else 1e-12
        self.initializer_range = float(config["hm4sr_initializer_range"]) if "hm4sr_initializer_range" in config else 0.02

        self.temperature = float(config["hm4sr_temperature"]) if "hm4sr_temperature" in config else 0.2
        self.phcl_temperature = float(config["hm4sr_phcl_temperature"]) if "hm4sr_phcl_temperature" in config else 1.0
        self.idcl_weight = float(config["hm4sr_idcl_weight"]) if "hm4sr_idcl_weight" in config else 0.05
        self.pcl_weight = float(config["hm4sr_pcl_weight"]) if "hm4sr_pcl_weight" in config else 0.0
        self.pcl_mask_ratio = float(config["hm4sr_pcl_mask_ratio"]) if "hm4sr_pcl_mask_ratio" in config else 0.3
        self.reg_weight = float(config["reg_weight"]) if "reg_weight" in config else 0.0

        self.start_expert_num = int(config["hm4sr_start_expert_num"]) if "hm4sr_start_expert_num" in config else 4
        self.temporal_expert_num = int(config["hm4sr_temporal_expert_num"]) if "hm4sr_temporal_expert_num" in config else 4
        self.interval_scale = float(config["hm4sr_interval_scale"]) if "hm4sr_interval_scale" in config else 100.0

        self.use_idcl = bool(config["hm4sr_use_idcl"]) if "hm4sr_use_idcl" in config else True
        self.use_pcl = bool(config["hm4sr_use_pcl"]) if "hm4sr_use_pcl" in config else False

        self.inter_file_name = config["inter_file_name"] if "inter_file_name" in config else None
        self.strict_inter_file = bool(config["hm4sr_strict_inter_file"]) if "hm4sr_strict_inter_file" in config else True

        self.user_field = str(config["USER_ID_FIELD"]).split(":")[0] if "USER_ID_FIELD" in config else "userID"
        self.item_field = str(config["ITEM_ID_FIELD"]).split(":")[0] if "ITEM_ID_FIELD" in config else "itemID"
        self.time_field = str(config["TIME_FIELD"]).split(":")[0] if "TIME_FIELD" in config else "timestamp"
        self.split_field = str(config["inter_splitting_label"]).split(":")[0] if "inter_splitting_label" in config else "x_label"

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_projection = nn.Linear(self.v_feat.shape[1], self.hidden_size)
        else:
            self.image_embedding = None
            self.image_projection = None

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projection = nn.Linear(self.t_feat.shape[1], self.hidden_size)
        else:
            self.text_embedding = None
            self.text_projection = None

        self.item_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.txt_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.img_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.item_seq_encoder = self._build_transformer_encoder()
        self.txt_seq_encoder = self._build_transformer_encoder()
        self.img_seq_encoder = self._build_transformer_encoder()

        self.start_moe = HM4SRInteractiveMoE(
            hidden_size=self.hidden_size,
            expert_num=self.start_expert_num,
            initializer_range=self.initializer_range,
        )
        self.time_moe = HM4SRTemporalMoE(
            hidden_size=self.hidden_size,
            expert_num=self.temporal_expert_num,
            interval_scale=self.interval_scale,
        )

        self.placeholder_txt = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.placeholder_img = nn.Linear(2 * self.hidden_size, self.hidden_size)

        seq_items, seq_times = self._build_user_sequences()
        self.register_buffer("hm4sr_user_seq_items", torch.LongTensor(seq_items))
        self.register_buffer("hm4sr_user_seq_times", torch.FloatTensor(seq_times))

        self.apply(self._init_weights)

        print("[HM4SR] dataset={}, data_path={}, inter_file={}".format(
            self.dataset_name, self.data_path, self._get_inter_path()
        ))
        print("[HM4SR] users={}, items={}, hidden_size={}, max_seq_len={}".format(
            self.n_users, self.n_items, self.hidden_size, self.max_seq_length
        ))
        print("[HM4SR] use_idcl={}, idcl_weight={}, use_pcl={}, pcl_weight={}".format(
            self.use_idcl, self.idcl_weight, self.use_pcl, self.pcl_weight
        ))
        print("[HM4SR] v_feat={}, t_feat={}".format(
            None if self.v_feat is None else tuple(self.v_feat.shape),
            None if self.t_feat is None else tuple(self.t_feat.shape),
        ))

    # ------------------------------------------------------------------
    # Initialization and transformer
    # ------------------------------------------------------------------
    def _build_transformer_encoder(self):
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        return nn.TransformerEncoder(layer, num_layers=self.n_layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "weight", None) is not None:
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # ------------------------------------------------------------------
    # Data parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_header(name):
        return str(name).split(":")[0].strip()

    def _find_col(self, header, target):
        target = self._normalize_header(target)
        for i, name in enumerate(header):
            if self._normalize_header(name) == target:
                return i
        return -1

    def _get_inter_path(self):
        if self.inter_file_name not in {None, "", "none", "None", "null", "NULL"}:
            inter_name = str(self.inter_file_name)
        else:
            inter_name = "{}.inter".format(self.dataset_name)

        candidates = []
        if os.path.isabs(inter_name):
            candidates.append(inter_name)
        else:
            candidates.extend([
                os.path.join(self.dataset_path, inter_name),
                os.path.join(self.data_path, self.dataset_name, inter_name),
                os.path.join(self.data_path, inter_name),
                os.path.abspath(inter_name),
            ])

        for p in candidates:
            if os.path.exists(p):
                return p

        # Return the most likely path for readable error message.
        return candidates[0]

    def _read_train_user_item_times(self):
        inter_path = self._get_inter_path()
        rows = []

        if not os.path.exists(inter_path):
            if self.strict_inter_file:
                raise FileNotFoundError("Cannot find inter file for HM4SR: {}".format(inter_path))
            coo = self.dataset_obj.inter_matrix(form="coo")
            for idx, (u, i) in enumerate(zip(coo.row, coo.col)):
                rows.append((int(u), int(i), float(idx)))
            return rows

        with open(inter_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)

            u_col = self._find_col(header, self.user_field)
            i_col = self._find_col(header, self.item_field)
            t_col = self._find_col(header, self.time_field)
            label_col = self._find_col(header, self.split_field)

            # Fallback names.
            if u_col < 0:
                u_col = self._find_col(header, "userID")
            if i_col < 0:
                i_col = self._find_col(header, "itemID")
            if t_col < 0:
                t_col = self._find_col(header, "timestamp")
            if label_col < 0:
                label_col = self._find_col(header, "x_label")

            if u_col < 0 or i_col < 0:
                raise ValueError("Cannot find user/item columns in {}".format(inter_path))

            for row in reader:
                if len(row) <= max(u_col, i_col, t_col if t_col >= 0 else 0, label_col if label_col >= 0 else 0):
                    continue

                if label_col >= 0:
                    try:
                        label = int(float(row[label_col]))
                    except Exception:
                        continue
                    # MMRec temporal split: x_label == 0 means train.
                    if label != 0:
                        continue

                try:
                    u = int(float(row[u_col]))
                    i = int(float(row[i_col]))
                    ts = float(row[t_col]) if t_col >= 0 else float(len(rows))
                except Exception:
                    continue

                if 0 <= u < self.n_users and 0 <= i < self.n_items:
                    rows.append((u, i, ts))

        return rows

    def _build_user_sequences(self):
        hist = defaultdict(list)
        for u, i, ts in self._read_train_user_item_times():
            hist[u].append((ts, i))

        seq_items = np.zeros((self.n_users, self.max_seq_length), dtype=np.int64)
        seq_times = np.zeros((self.n_users, self.max_seq_length), dtype=np.float32)

        for u in range(self.n_users):
            arr = sorted(hist.get(u, []), key=lambda x: x[0])
            arr = arr[-self.max_seq_length:]
            offset = self.max_seq_length - len(arr)
            for k, (ts, i) in enumerate(arr):
                seq_items[u, offset + k] = int(i)
                seq_times[u, offset + k] = float(ts)

        return seq_items, seq_times

    def _batch_sequences(self, users, target_items=None, remove_target=True):
        seq_items = self.hm4sr_user_seq_items[users].clone()
        seq_times = self.hm4sr_user_seq_times[users].clone()

        if target_items is not None and remove_target:
            # Remove one occurrence of the current positive item from right to reduce leakage.
            for b in range(seq_items.size(0)):
                tgt = int(target_items[b].detach().cpu().item())
                pos = torch.nonzero(seq_items[b] == tgt, as_tuple=False)
                if pos.numel() > 0:
                    idx = int(pos[-1].item())
                    seq_items[b, idx] = 0
                    seq_times[b, idx] = 0.0

        return seq_items, seq_times

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _text_item_table(self):
        if self.text_embedding is None:
            return torch.zeros((self.n_items, self.hidden_size), device=self.device)
        return self.text_projection(self.text_embedding.weight)

    def _image_item_table(self):
        if self.image_embedding is None:
            return torch.zeros((self.n_items, self.hidden_size), device=self.device)
        return self.image_projection(self.image_embedding.weight)

    def _embed_sequence(self, item_seq):
        item_emb = self.item_embedding(item_seq)

        if self.text_embedding is not None:
            txt_emb = self.text_projection(self.text_embedding(item_seq))
        else:
            txt_emb = torch.zeros_like(item_emb)

        if self.image_embedding is not None:
            img_emb = self.image_projection(self.image_embedding(item_seq))
        else:
            img_emb = torch.zeros_like(item_emb)

        pos_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0).expand(item_seq.size(0), -1)
        pos_emb = self.position_embedding(pos_ids)

        item_emb = item_emb + pos_emb
        txt_emb = txt_emb + pos_emb
        img_emb = img_emb + pos_emb

        mask = (item_seq != 0).unsqueeze(-1).float()
        return item_emb * mask, txt_emb * mask, img_emb * mask

    @staticmethod
    def _last_nonzero_index(item_seq):
        valid_len = (item_seq != 0).long().sum(dim=1)
        return torch.clamp(valid_len - 1, min=0)

    @staticmethod
    def _gather_last(seq_out, last_index):
        idx = last_index.view(-1, 1, 1).expand(-1, 1, seq_out.size(-1))
        return seq_out.gather(dim=1, index=idx).squeeze(1)

    def _encode_sequence(self, item_seq, timestamp):
        item_emb, txt_emb, img_emb = self._embed_sequence(item_seq)

        delta_item, delta_txt, delta_img = self.start_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1))
        item_emb = item_emb + delta_item
        txt_emb = txt_emb + delta_txt
        img_emb = img_emb + delta_img

        item_emb, txt_emb, img_emb = self.time_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1), timestamp)

        item_emb = self.dropout(self.item_ln(item_emb))
        txt_emb = self.dropout(self.txt_ln(txt_emb))
        img_emb = self.dropout(self.img_ln(img_emb))

        padding_mask = item_seq == 0
        item_out = self.item_seq_encoder(item_emb, src_key_padding_mask=padding_mask)
        txt_out = self.txt_seq_encoder(txt_emb, src_key_padding_mask=padding_mask)
        img_out = self.img_seq_encoder(img_emb, src_key_padding_mask=padding_mask)

        last_idx = self._last_nonzero_index(item_seq)
        item_vec = self._gather_last(item_out, last_idx)
        txt_vec = self._gather_last(txt_out, last_idx)
        img_vec = self._gather_last(img_out, last_idx)

        return (item_emb, txt_emb, img_emb), (item_vec, txt_vec, img_vec)

    def _score_all_items(self, seq_vecs):
        item_vec, txt_vec, img_vec = seq_vecs
        id_table = self.item_embedding.weight
        txt_table = self._text_item_table()
        img_table = self._image_item_table()
        return (
            torch.matmul(item_vec, id_table.transpose(0, 1))
            + torch.matmul(txt_vec, txt_table.transpose(0, 1))
            + torch.matmul(img_vec, img_table.transpose(0, 1))
        )

    def _score_items(self, seq_vecs, item_ids):
        item_vec, txt_vec, img_vec = seq_vecs
        id_table = self.item_embedding(item_ids)
        txt_table = self._text_item_table()[item_ids]
        img_table = self._image_item_table()[item_ids]
        return (
            torch.sum(item_vec * id_table, dim=-1)
            + torch.sum(txt_vec * txt_table, dim=-1)
            + torch.sum(img_vec * img_table, dim=-1)
        )

    @staticmethod
    def bpr_loss_by_scores(pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------
    def id_contrastive_loss(self, seq_vec, pos_items):
        seq_output = F.normalize(seq_vec, dim=1)
        pos_emb = F.normalize(self.item_embedding(pos_items), dim=1)

        same_pos = pos_items.unsqueeze(1).eq(pos_items.unsqueeze(0))
        same_pos = torch.logical_xor(
            same_pos,
            torch.eye(pos_items.size(0), dtype=torch.bool, device=pos_items.device),
        )

        pos_logits = torch.exp(torch.sum(seq_output * pos_emb, dim=1) / self.temperature)
        all_logits = torch.matmul(seq_output, pos_emb.transpose(0, 1)) / self.temperature
        all_logits = torch.where(same_pos, torch.zeros_like(all_logits), all_logits)
        neg_logits = torch.exp(all_logits).sum(dim=1).clamp_min(1e-12)
        return -torch.log(pos_logits / neg_logits).mean()

    def seq2seq_contrastive_loss(self, seq_a, seq_b, pos_items):
        seq_a = F.normalize(seq_a, dim=1)
        seq_b = F.normalize(seq_b, dim=1)

        same_pos = pos_items.unsqueeze(1).eq(pos_items.unsqueeze(0))
        same_pos = torch.logical_xor(
            same_pos,
            torch.eye(pos_items.size(0), dtype=torch.bool, device=pos_items.device),
        )

        pos_logits = torch.exp(torch.sum(seq_a * seq_b, dim=1) / self.phcl_temperature)
        all_logits = torch.matmul(seq_a, seq_b.transpose(0, 1)) / self.phcl_temperature
        all_logits = torch.where(same_pos, torch.zeros_like(all_logits), all_logits)
        neg_logits = torch.exp(all_logits).sum(dim=1).clamp_min(1e-12)
        return -torch.log(pos_logits / neg_logits).mean()

    def pcl_loss(self, item_seq, timestamp, seq_vecs, pos_items):
        if self.pcl_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        item_emb, txt_emb, img_emb = self._embed_sequence(item_seq)
        valid = item_seq != 0

        rand = torch.rand_like(item_seq.float())
        mask = (rand < self.pcl_mask_ratio) & valid

        time_emb = self.time_moe.get_time_embedding(timestamp)
        txt_aug = txt_emb.masked_fill(mask.unsqueeze(-1), 0.0)
        img_aug = img_emb.masked_fill(mask.unsqueeze(-1), 0.0)

        txt_aug = txt_aug + self.placeholder_txt(time_emb).masked_fill(~mask.unsqueeze(-1), 0.0)
        img_aug = img_aug + self.placeholder_img(time_emb).masked_fill(~mask.unsqueeze(-1), 0.0)

        txt_aug = self.dropout(self.txt_ln(txt_aug))
        img_aug = self.dropout(self.img_ln(img_aug))

        padding_mask = item_seq == 0
        txt_out = self.txt_seq_encoder(txt_aug, src_key_padding_mask=padding_mask)
        img_out = self.img_seq_encoder(img_aug, src_key_padding_mask=padding_mask)

        last_idx = self._last_nonzero_index(item_seq)
        txt_vec_aug = self._gather_last(txt_out, last_idx)
        img_vec_aug = self._gather_last(img_out, last_idx)

        txt_loss = self.seq2seq_contrastive_loss(seq_vecs[1], txt_vec_aug, pos_items)
        img_loss = self.seq2seq_contrastive_loss(seq_vecs[2], img_vec_aug, pos_items)
        return 0.5 * (txt_loss + img_loss)

    # ------------------------------------------------------------------
    # MMRec API
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        users = interaction[0].long()
        pos_items = interaction[1].long()
        neg_items = interaction[2].long()

        item_seq, timestamp = self._batch_sequences(users, pos_items, remove_target=True)
        _, seq_vecs = self._encode_sequence(item_seq, timestamp)

        pos_scores = self._score_items(seq_vecs, pos_items)
        neg_scores = self._score_items(seq_vecs, neg_items)
        loss = self.bpr_loss_by_scores(pos_scores, neg_scores)

        if self.use_idcl and self.idcl_weight > 0.0:
            loss = loss + self.idcl_weight * self.id_contrastive_loss(seq_vecs[0], pos_items)

        if self.use_pcl and self.pcl_weight > 0.0:
            loss = loss + self.pcl_weight * self.pcl_loss(item_seq, timestamp, seq_vecs, pos_items)

        if self.reg_weight > 0.0:
            reg = (
                self.item_embedding(pos_items).pow(2).sum(dim=1)
                + self.item_embedding(neg_items).pow(2).sum(dim=1)
            ).mean()
            loss = loss + self.reg_weight * reg

        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0].long()
        item_seq, timestamp = self._batch_sequences(users, target_items=None, remove_target=False)
        _, seq_vecs = self._encode_sequence(item_seq, timestamp)
        return self._score_all_items(seq_vecs)


class HM4SRInteractiveMoE(nn.Module):
    def __init__(self, hidden_size, expert_num=4, initializer_range=0.02):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_num = int(expert_num)

        self.gate_id = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_txt = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_img = nn.Linear(self.hidden_size, self.expert_num)
        self.expert = nn.ModuleList([
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3)
            for _ in range(self.expert_num)
        ])
        self.weight = nn.Parameter(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=initializer_range)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, vector):
        id_x = vector[:, :, :self.hidden_size]
        txt_x = vector[:, :, self.hidden_size:2 * self.hidden_size]
        img_x = vector[:, :, 2 * self.hidden_size:]

        expert_output = torch.stack([expert(vector) for expert in self.expert], dim=2)

        id_gate = F.softmax(self.gate_id(id_x), dim=-1).unsqueeze(-1)
        txt_gate = F.softmax(self.gate_txt(txt_x), dim=-1).unsqueeze(-1)
        img_gate = F.softmax(self.gate_img(img_x), dim=-1).unsqueeze(-1)

        id_delta = torch.sum(expert_output[:, :, :, :self.hidden_size] * id_gate, dim=2)
        txt_delta = torch.sum(expert_output[:, :, :, self.hidden_size:2 * self.hidden_size] * txt_gate, dim=2)
        img_delta = torch.sum(expert_output[:, :, :, 2 * self.hidden_size:] * img_gate, dim=2)

        return self.weight[0] * id_delta, self.weight[1] * txt_delta, self.weight[2] * img_delta


class HM4SRTemporalMoE(nn.Module):
    def __init__(self, hidden_size, expert_num=4, interval_scale=100.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_num = int(expert_num)
        self.interval_scale = float(interval_scale)

        self.absolute_w = nn.Linear(1, self.hidden_size)
        self.time_embedding = nn.Embedding(4096, self.hidden_size)
        self.gate = nn.Linear(2 * self.hidden_size, self.expert_num)
        # Shape is [1, 1, E, 3H] so it broadcasts with
        # vector.unsqueeze(2): [B, L, 1, 3H].
        self.expert = nn.Parameter(torch.randn(1, 1, self.expert_num, self.hidden_size * 3) * 0.1)

    def freq_enhance_ab(self, timestamp):
        freq = 10000.0
        freq_seq = torch.arange(0, self.hidden_size, 1.0, dtype=torch.float, device=timestamp.device)
        inv_freq = 1.0 / torch.pow(
            torch.tensor(freq, dtype=torch.float, device=timestamp.device),
            (freq_seq / self.hidden_size),
        ).view(1, 1, -1)
        return timestamp * inv_freq

    def get_time_embedding(self, timestamp):
        ts = timestamp.float()
        valid = ts > 0
        if valid.any():
            ts_min = ts[valid].min()
            ts = torch.where(valid, ts - ts_min, torch.zeros_like(ts))
            scale = ts[valid].max().clamp_min(1.0)
            ts_norm = ts / scale
        else:
            ts_norm = ts

        absolute = torch.cos(self.freq_enhance_ab(self.absolute_w(ts_norm.unsqueeze(-1))))

        delta = torch.zeros_like(ts)
        delta[:, 1:] = torch.clamp(ts[:, 1:] - ts[:, :-1], min=0.0)
        interval = torch.log2(delta + 1.0)
        interval_idx = torch.floor(self.interval_scale * interval).long().clamp(min=0, max=4095)
        interval_emb = self.time_embedding(interval_idx)
        return torch.cat([interval_emb, absolute], dim=-1)

    def forward(self, vector, timestamp):
        time_emb = self.get_time_embedding(timestamp)
        route = F.softmax(self.gate(time_emb), dim=-1)
        expert = self.expert.to(vector.device)
        expert_output = vector.unsqueeze(2) * expert  # [B, L, E, 3H]
        out = torch.sum(expert_output * route.unsqueeze(-1), dim=2)
        return (
            out[:, :, :self.hidden_size],
            out[:, :, self.hidden_size:2 * self.hidden_size],
            out[:, :, 2 * self.hidden_size:],
        )
