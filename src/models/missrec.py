# coding: utf-8
r"""
MISSRec adapted for MMRec.

File:
    src/models/missrec.py

Class/model name:
    MISSRec

Official MISSRec is a RecBole-style sequential recommendation framework with
multi-modal interest-aware sequence representation, pre-training/transfer, and
dynamic item fusion. This adapter keeps the parts that can run inside MMRec's
GeneralRecommender trainer:
    1. train-only chronological user sequences from .inter;
    2. ID / image / text sequence representation;
    3. Transformer interest encoder-decoder;
    4. dynamic user-adaptive item fusion;
    5. BPR training with MMRec interactions.

This is designed as a runnable sequential/multimodal baseline for the user's
baby/sports/clothing/microlens datasets.
"""

import os
import csv
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class MISSRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MISSRec, self).__init__(config, dataset)

        self.config = config
        self.dataset_obj = dataset
        self.dataset_name = str(config["dataset"])
        self.data_path = os.path.abspath(str(config["data_path"]))
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, self.dataset_name))

        # Dedicated keys avoid conflict with overall.yaml.
        self.hidden_size = int(config["missrec_hidden_size"]) if "missrec_hidden_size" in config else int(config["embedding_size"])
        self.max_seq_length = int(config["missrec_max_seq_length"]) if "missrec_max_seq_length" in config else 50
        self.n_layers = int(config["missrec_n_layers"]) if "missrec_n_layers" in config else 2
        self.n_heads = int(config["missrec_n_heads"]) if "missrec_n_heads" in config else 4
        self.inner_size = int(config["missrec_inner_size"]) if "missrec_inner_size" in config else 256
        self.hidden_dropout_prob = float(config["missrec_hidden_dropout_prob"]) if "missrec_hidden_dropout_prob" in config else 0.5
        self.attn_dropout_prob = float(config["missrec_attn_dropout_prob"]) if "missrec_attn_dropout_prob" in config else 0.5
        self.hidden_act = str(config["missrec_hidden_act"]) if "missrec_hidden_act" in config else "gelu"
        self.layer_norm_eps = float(config["missrec_layer_norm_eps"]) if "missrec_layer_norm_eps" in config else 1e-12
        self.initializer_range = float(config["missrec_initializer_range"]) if "missrec_initializer_range" in config else 0.02

        self.temperature = float(config["missrec_temperature"]) if "missrec_temperature" in config else 0.07
        self.gamma = float(config["missrec_gamma"]) if "missrec_gamma" in config else 1e-4
        self.reg_weight = float(config["reg_weight"]) if "reg_weight" in config else 0.0

        self.seq_mm_fusion = str(config["missrec_seq_mm_fusion"]) if "missrec_seq_mm_fusion" in config else "add"
        self.item_mm_fusion = str(config["missrec_item_mm_fusion"]) if "missrec_item_mm_fusion" in config else "dynamic_shared"
        self.id_type = str(config["missrec_id_type"]) if "missrec_id_type" in config else "id"
        self.num_interest = int(config["missrec_num_interest"]) if "missrec_num_interest" in config else 8

        assert self.seq_mm_fusion in ["add", "contextual"]
        assert self.item_mm_fusion in ["static", "dynamic_shared", "dynamic_instance"]
        assert self.id_type in ["id", "none"]

        self.strict_inter_file = bool(config["missrec_strict_inter_file"]) if "missrec_strict_inter_file" in config else True
        self.inter_file_name = config["inter_file_name"] if "inter_file_name" in config else None

        self.user_field = str(config["USER_ID_FIELD"]).split(":")[0] if "USER_ID_FIELD" in config else "userID"
        self.item_field = str(config["ITEM_ID_FIELD"]).split(":")[0] if "ITEM_ID_FIELD" in config else "itemID"
        self.time_field = str(config["TIME_FIELD"]).split(":")[0] if "TIME_FIELD" in config else "timestamp"
        self.split_field = str(config["inter_splitting_label"]).split(":")[0] if "inter_splitting_label" in config else "x_label"

        # Embeddings / projections.
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.interest_embedding = nn.Embedding(self.num_interest + 1, self.hidden_size, padding_idx=0)

        if self.v_feat is not None:
            self.img_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.img_adaptor = nn.Linear(self.v_feat.shape[1], self.hidden_size)
        else:
            self.img_embedding = None
            self.img_adaptor = None

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_adaptor = nn.Linear(self.t_feat.shape[1], self.hidden_size)
        else:
            self.text_embedding = None
            self.text_adaptor = None

        if self.item_mm_fusion == "dynamic_shared":
            self.fusion_factor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        elif self.item_mm_fusion == "dynamic_instance":
            self.fusion_factor = nn.Parameter(torch.zeros(self.n_items, dtype=torch.float32))
        else:
            self.register_parameter("fusion_factor", None)

        if self.n_layers % 2 != 0:
            raise ValueError("missrec_n_layers must be even because encoder/decoder split uses n_layers // 2.")

        self.trm_model = nn.Transformer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers // 2,
            num_decoder_layers=self.n_layers // 2,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,
            norm_first=False,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        seq_items, seq_lens = self._build_user_sequences()
        self.register_buffer("missrec_user_seq_items", torch.LongTensor(seq_items))
        self.register_buffer("missrec_user_seq_lens", torch.LongTensor(seq_lens))

        self.apply(self._init_weights)

        print("[MISSRec] dataset={}, inter_file={}".format(self.dataset_name, self._get_inter_path()))
        print("[MISSRec] users={}, items={}, hidden_size={}, max_seq_len={}, num_interest={}".format(
            self.n_users, self.n_items, self.hidden_size, self.max_seq_length, self.num_interest
        ))
        print("[MISSRec] seq_mm_fusion={}, item_mm_fusion={}, id_type={}".format(
            self.seq_mm_fusion, self.item_mm_fusion, self.id_type
        ))
        print("[MISSRec] v_feat={}, t_feat={}".format(
            None if self.v_feat is None else tuple(self.v_feat.shape),
            None if self.t_feat is None else tuple(self.t_feat.shape),
        ))

    # ------------------------------------------------------------------
    # Init / data
    # ------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "weight", None) is not None:
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def _normalize_header(name):
        return str(name).split(":")[0].strip()

    def _find_col(self, header, target):
        target = self._normalize_header(target)
        for idx, name in enumerate(header):
            if self._normalize_header(name) == target:
                return idx
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
        return candidates[0]

    def _read_train_user_item_times(self):
        inter_path = self._get_inter_path()
        rows = []

        if not os.path.exists(inter_path):
            if self.strict_inter_file:
                raise FileNotFoundError("Cannot find inter file for MISSRec: {}".format(inter_path))
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
        seq_lens = np.zeros((self.n_users,), dtype=np.int64)
        for u in range(self.n_users):
            arr = sorted(hist.get(u, []), key=lambda x: x[0])
            items = [int(i) for _, i in arr[-self.max_seq_length:]]
            seq_lens[u] = len(items)
            offset = self.max_seq_length - len(items)
            for k, item in enumerate(items):
                seq_items[u, offset + k] = item
        return seq_items, seq_lens

    def _batch_sequences(self, users, target_items=None, remove_target=True):
        item_seq = self.missrec_user_seq_items[users].clone()
        if target_items is not None and remove_target:
            for b in range(item_seq.size(0)):
                tgt = int(target_items[b].detach().cpu().item())
                pos = torch.nonzero(item_seq[b] == tgt, as_tuple=False)
                if pos.numel() > 0:
                    item_seq[b, int(pos[-1].item())] = 0
        seq_len = (item_seq != 0).long().sum(dim=1).clamp_min(1)
        return item_seq, seq_len

    @staticmethod
    def gather_indexes(output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, 1, output.size(-1))
        return output.gather(dim=1, index=gather_index).squeeze(1)

    # ------------------------------------------------------------------
    # Embeddings and model
    # ------------------------------------------------------------------
    def _text_table(self):
        if self.text_embedding is None:
            return torch.zeros((self.n_items, self.hidden_size), device=self.device)
        return self.text_adaptor(self.text_embedding.weight)

    def _img_table(self):
        if self.img_embedding is None:
            return torch.zeros((self.n_items, self.hidden_size), device=self.device)
        return self.img_adaptor(self.img_embedding.weight)

    def _sequence_modal_embeddings(self, item_seq):
        text_emb = self._text_table()[item_seq] if self.text_embedding is not None else torch.zeros(item_seq.size(0), item_seq.size(1), self.hidden_size, device=item_seq.device)
        img_emb = self._img_table()[item_seq] if self.img_embedding is not None else torch.zeros(item_seq.size(0), item_seq.size(1), self.hidden_size, device=item_seq.device)
        id_emb = self.item_embedding(item_seq)

        if self.seq_mm_fusion == "add":
            seq_emb = text_emb + img_emb
            if self.id_type != "none":
                seq_emb = seq_emb + id_emb
            item_modal_empty_mask = (item_seq == 0).unsqueeze(1)
        else:
            # contextual: [B, M, L, H] -> flatten to [B, M*L, H]
            parts = [text_emb, img_emb]
            if self.id_type != "none":
                parts.append(id_emb)
            seq_emb = torch.stack(parts, dim=1)
            item_modal_empty_mask = (item_seq == 0).unsqueeze(1).expand(-1, seq_emb.size(1), -1)
        return seq_emb, item_modal_empty_mask

    def _interest_tokens(self, item_seq):
        # Lightweight interest discovery: hashed item IDs map to trainable interest tokens.
        # 0 is reserved for padding.
        idx = torch.remainder(item_seq, self.num_interest) + 1
        idx = idx.masked_fill(item_seq == 0, 0)
        emb = self.interest_embedding(idx)
        return idx, emb

    def _encoder_attention_mask(self, interest_seq):
        key_padding_mask = interest_seq == 0
        return None, key_padding_mask

    def _decoder_attention_mask(self, item_seq, item_modal_empty_mask):
        batch_size, num_modality, seq_len = item_modal_empty_mask.shape
        if self.seq_mm_fusion == "add":
            key_padding_mask = item_seq == 0
            attn_mask = None
        else:
            key_padding_mask = torch.logical_or((item_seq == 0).unsqueeze(1), item_modal_empty_mask).flatten(1)
            attn_mask = None
        return attn_mask, None, key_padding_mask

    def forward(self, item_seq, item_seq_len):
        interest_seq, interest_emb = self._interest_tokens(item_seq)
        item_emb, item_modal_empty_mask = self._sequence_modal_embeddings(item_seq)

        src_attn_mask, src_key_padding_mask = self._encoder_attention_mask(interest_seq)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)
        if self.seq_mm_fusion == "add":
            dec_input_emb = item_emb + position_embedding
        else:
            dec_input_emb = item_emb + position_embedding.view(1, 1, item_seq.size(1), -1)
            dec_input_emb = dec_input_emb.view(dec_input_emb.size(0), -1, dec_input_emb.size(-1))

        dec_input_emb = self.LayerNorm(dec_input_emb)
        dec_input_emb = self.dropout(dec_input_emb)

        memory = self.trm_model.encoder(
            src=interest_emb,
            mask=src_attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        tgt_attn_mask, tgt_cross_attn_mask, tgt_key_padding_mask = self._decoder_attention_mask(item_seq, item_modal_empty_mask)

        trm_output = self.trm_model.decoder(
            dec_input_emb,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=tgt_cross_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        output = self.gather_indexes(trm_output, item_seq_len - 1)

        # Diversity regularization proxy for interest tokens.
        valid = (~src_key_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled_interest = (memory * valid).sum(dim=1) / denom
        interest_reg = (pooled_interest * pooled_interest).sum(dim=1).mean() / self.hidden_size

        output = F.normalize(output, dim=1)
        return output, interest_reg

    def _test_item_embeddings(self):
        text_emb = self._text_table()
        img_emb = self._img_table()
        if self.id_type != "none":
            id_emb = self.item_embedding.weight
        else:
            id_emb = torch.zeros_like(text_emb)
        return text_emb, img_emb, id_emb

    def _dynamic_fused_scores(self, seq_output):
        text_emb, img_emb, id_emb = self._test_item_embeddings()
        text_emb = F.normalize(text_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        id_emb = F.normalize(id_emb, dim=1)

        text_scores = torch.matmul(seq_output, text_emb.transpose(0, 1))
        img_scores = torch.matmul(seq_output, img_emb.transpose(0, 1))
        modality_scores = torch.stack([text_scores, img_scores], dim=-1)

        if self.item_mm_fusion in ["dynamic_shared", "dynamic_instance"]:
            factor = self.fusion_factor
            if self.item_mm_fusion == "dynamic_instance":
                # [n_items] -> [1,n_items,1]
                weight = F.softmax(modality_scores * factor.view(1, -1, 1), dim=-1)
            else:
                weight = F.softmax(modality_scores * factor, dim=-1)
            mm_scores = (modality_scores * weight).sum(dim=-1)
        else:
            mm_scores = modality_scores.mean(dim=-1)

        if self.id_type != "none":
            id_scores = torch.matmul(seq_output, id_emb.transpose(0, 1))
            scores = (id_scores + 2.0 * mm_scores) / 3.0
        else:
            scores = mm_scores
        return scores / self.temperature

    def _score_items(self, scores, item_ids):
        return scores.gather(1, item_ids.view(-1, 1)).squeeze(1)

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    # ------------------------------------------------------------------
    # MMRec API
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        users = interaction[0].long()
        pos_items = interaction[1].long()
        neg_items = interaction[2].long()

        item_seq, item_seq_len = self._batch_sequences(users, pos_items, remove_target=True)
        seq_output, interest_reg = self.forward(item_seq, item_seq_len)

        scores = self._dynamic_fused_scores(seq_output)
        pos_scores = self._score_items(scores, pos_items)
        neg_scores = self._score_items(scores, neg_items)

        loss = self.bpr_loss(pos_scores, neg_scores) + self.gamma * interest_reg
        if self.reg_weight > 0.0:
            reg = (self.item_embedding(pos_items).pow(2).sum(dim=1) + self.item_embedding(neg_items).pow(2).sum(dim=1)).mean()
            loss = loss + self.reg_weight * reg
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0].long()
        item_seq, item_seq_len = self._batch_sequences(users, target_items=None, remove_target=False)
        seq_output, _ = self.forward(item_seq, item_seq_len)
        return self._dynamic_fused_scores(seq_output)
