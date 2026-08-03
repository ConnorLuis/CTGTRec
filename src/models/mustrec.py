# coding: utf-8
r"""
MuSTRec adapted for MMRec.

File:
    src/models/mustrec.py

Class/model name:
    MuSTRec

The anonymous MuSTRec repository was not directly accessible in this environment,
so this adapter implements the design described in the MuSTRec paper/abstract:
    1. multimodal + sequential recommendation;
    2. item-item similarity signals from text/image features;
    3. frequency-based self-attention for short/long preference decomposition;
    4. optional user embedding injection for sequential recommendation.

It is implemented against MMRec's GeneralRecommender API and your existing
baby/sports/clothing/microlens dataset yaml files.
"""
import math
import os
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class MuSTRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MuSTRec, self).__init__(config, dataset)

        self.config = config
        self.dataset_obj = dataset
        self.dataset_name = str(config["dataset"])
        self.data_path = os.path.abspath(str(config["data_path"]))
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, self.dataset_name))

        self.hidden_size = int(config["mustrec_hidden_size"]) if "mustrec_hidden_size" in config else int(config["embedding_size"])
        self.max_seq_length = int(config["mustrec_max_seq_length"]) if "mustrec_max_seq_length" in config else 50
        self.n_layers = int(config["mustrec_n_layers"]) if "mustrec_n_layers" in config else 2
        self.n_heads = int(config["mustrec_n_heads"]) if "mustrec_n_heads" in config else 2
        self.inner_size = int(config["mustrec_inner_size"]) if "mustrec_inner_size" in config else 4 * self.hidden_size
        self.dropout_prob = float(config["mustrec_dropout_prob"]) if "mustrec_dropout_prob" in config else 0.2
        self.layer_norm_eps = float(config["mustrec_layer_norm_eps"]) if "mustrec_layer_norm_eps" in config else 1e-12
        self.initializer_range = float(config["mustrec_initializer_range"]) if "mustrec_initializer_range" in config else 0.02

        self.freq_cut_ratio = float(config["mustrec_freq_cut_ratio"]) if "mustrec_freq_cut_ratio" in config else 0.5
        self.short_weight = float(config["mustrec_short_weight"]) if "mustrec_short_weight" in config else 1.0
        self.long_weight = float(config["mustrec_long_weight"]) if "mustrec_long_weight" in config else 1.0
        self.mm_weight = float(config["mustrec_mm_weight"]) if "mustrec_mm_weight" in config else 1.0
        self.id_weight = float(config["mustrec_id_weight"]) if "mustrec_id_weight" in config else 1.0
        self.use_user_embedding = bool(config["mustrec_use_user_embedding"]) if "mustrec_use_user_embedding" in config else True
        self.reg_weight = float(config["reg_weight"]) if "reg_weight" in config else 0.0

        self.strict_inter_file = bool(config["mustrec_strict_inter_file"]) if "mustrec_strict_inter_file" in config else True
        self.inter_file_name = config["inter_file_name"] if "inter_file_name" in config else None

        self.user_field = str(config["USER_ID_FIELD"]).split(":")[0] if "USER_ID_FIELD" in config else "userID"
        self.item_field = str(config["ITEM_ID_FIELD"]).split(":")[0] if "ITEM_ID_FIELD" in config else "itemID"
        self.time_field = str(config["TIME_FIELD"]).split(":")[0] if "TIME_FIELD" in config else "timestamp"
        self.split_field = str(config["inter_splitting_label"]).split(":")[0] if "inter_splitting_label" in config else "x_label"

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size) if self.use_user_embedding else None
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        if self.v_feat is not None:
            self.img_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.img_proj = nn.Linear(self.v_feat.shape[1], self.hidden_size)
        else:
            self.img_embedding = None
            self.img_proj = None

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.hidden_size)
        else:
            self.text_embedding = None
            self.text_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.seq_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Frequency branch fusion.
        self.freq_gate = nn.Linear(2 * self.hidden_size, 2)
        self.output_proj = nn.Linear(3 * self.hidden_size, self.hidden_size)

        # ID-guided modality gate for candidate item representations.
        self.modality_gate = nn.Linear(3 * self.hidden_size, 2)

        seq_items, seq_lens = self._build_user_sequences()
        self.register_buffer("mustrec_user_seq_items", torch.LongTensor(seq_items))
        self.register_buffer("mustrec_user_seq_lens", torch.LongTensor(seq_lens))

        self.apply(self._init_weights)

        print("[MuSTRec] dataset={}, inter_file={}".format(self.dataset_name, self._get_inter_path()))
        print("[MuSTRec] users={}, items={}, hidden_size={}, max_seq_len={}".format(
            self.n_users, self.n_items, self.hidden_size, self.max_seq_length
        ))
        print("[MuSTRec] use_user_embedding={}, freq_cut_ratio={}".format(
            self.use_user_embedding, self.freq_cut_ratio
        ))
        print("[MuSTRec] v_feat={}, t_feat={}".format(
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
                raise FileNotFoundError("Cannot find inter file for MuSTRec: {}".format(inter_path))
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
        item_seq = self.mustrec_user_seq_items[users].clone()
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
    # Embeddings and frequency block
    # ------------------------------------------------------------------
    def _image_table(self):
        if self.img_embedding is None:
            return torch.zeros((self.n_items, self.hidden_size), device=self.device)
        return self.img_proj(self.img_embedding.weight)

    def _text_table(self):
        if self.text_embedding is None:
            return torch.zeros((self.n_items, self.hidden_size), device=self.device)
        return self.text_proj(self.text_embedding.weight)

    def _item_mm_table(self):
        id_table = self.item_embedding.weight
        img_table = self._image_table()
        text_table = self._text_table()
        gate_input = torch.cat([id_table, img_table, text_table], dim=-1)
        gate = F.softmax(self.modality_gate(gate_input), dim=-1)
        mm_table = gate[:, 0:1] * img_table + gate[:, 1:2] * text_table
        return id_table, mm_table, img_table, text_table

    def _sequence_embedding(self, item_seq, users):
        id_table, mm_table, _, _ = self._item_mm_table()
        seq_id = id_table[item_seq]
        seq_mm = mm_table[item_seq]
        seq = self.id_weight * seq_id + self.mm_weight * seq_mm

        pos_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0).expand(item_seq.size(0), -1)
        seq = seq + self.position_embedding(pos_ids)

        if self.use_user_embedding:
            seq = seq + self.user_embedding(users).unsqueeze(1)

        mask = (item_seq != 0).unsqueeze(-1).float()
        return self.dropout(self.seq_ln(seq * mask))

    def _frequency_decompose(self, x):
        # FFT along sequence length.
        x_freq = torch.fft.rfft(x, dim=1)
        n_freq = x_freq.size(1)
        cut = max(1, min(n_freq, int(math.ceil(n_freq * self.freq_cut_ratio))))

        low_freq = torch.zeros_like(x_freq)
        high_freq = torch.zeros_like(x_freq)
        low_freq[:, :cut, :] = x_freq[:, :cut, :]
        high_freq[:, cut:, :] = x_freq[:, cut:, :]

        low = torch.fft.irfft(low_freq, n=x.size(1), dim=1)
        high = torch.fft.irfft(high_freq, n=x.size(1), dim=1)
        return low, high

    def forward(self, users, item_seq, item_seq_len):
        seq = self._sequence_embedding(item_seq, users)
        padding_mask = item_seq == 0

        enc = self.encoder(seq, src_key_padding_mask=padding_mask)
        low, high = self._frequency_decompose(enc)

        long_vec = self.gather_indexes(low, item_seq_len - 1)
        short_vec = self.gather_indexes(high, item_seq_len - 1)
        base_vec = self.gather_indexes(enc, item_seq_len - 1)

        gate = F.softmax(self.freq_gate(torch.cat([long_vec, short_vec], dim=-1)), dim=-1)
        freq_vec = gate[:, 0:1] * self.long_weight * long_vec + gate[:, 1:2] * self.short_weight * short_vec

        out = self.output_proj(torch.cat([base_vec, long_vec, short_vec], dim=-1)) + freq_vec
        return F.normalize(out, dim=1)

    def _candidate_table(self):
        id_table, mm_table, _, _ = self._item_mm_table()
        cand = self.id_weight * id_table + self.mm_weight * mm_table
        return F.normalize(cand, dim=1)

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
        user_vec = self.forward(users, item_seq, item_seq_len)
        cand = self._candidate_table()

        pos_scores = torch.sum(user_vec * cand[pos_items], dim=-1)
        neg_scores = torch.sum(user_vec * cand[neg_items], dim=-1)
        loss = self.bpr_loss(pos_scores, neg_scores)

        if self.reg_weight > 0.0:
            reg = (self.item_embedding(pos_items).pow(2).sum(dim=1) + self.item_embedding(neg_items).pow(2).sum(dim=1)).mean()
            loss = loss + self.reg_weight * reg
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0].long()
        item_seq, item_seq_len = self._batch_sequences(users, target_items=None, remove_target=False)
        user_vec = self.forward(users, item_seq, item_seq_len)
        cand = self._candidate_table()
        return torch.matmul(user_vec, cand.transpose(0, 1))
