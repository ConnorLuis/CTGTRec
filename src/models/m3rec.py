# coding: utf-8
r"""
M3Rec adapted for MMRec.

File name:
    src/models/m3rec.py

Class name / model name:
    M3Rec

Official reference:
    M³Rec: Selective State Space Models with Mixture-of-Modality Experts for
    Multi-Modal Sequential Recommendation.

Why this adapter exists:
    The official M3Rec implementation is a RecBole SequentialRecommender and
    depends on mamba_ssm. This file adapts the core idea to the MMRec
    GeneralRecommender API:
        calculate_loss(interaction): users, pos_items, neg_items
        full_sort_predict(interaction): return [batch_size, n_items] scores

Main retained ideas:
    1. ID / image / text sequence streams.
    2. Shared Mamba-style sequence block across modalities.
    3. Modality-specific FeedForward experts.
    4. Learnable image/text modality weights.
    5. Candidate scoring by concatenating image/text/ID representations.

Compatibility note:
    If mamba_ssm is installed, this file uses the official Mamba block.
    If mamba_ssm is not installed, it automatically falls back to a lightweight
    GRU-based state-space-like block, so the model can still run and produce
    comparable baseline logs in your MMRec environment.
"""

import os
import csv
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender

try:
    from mamba_ssm import Mamba as OfficialMamba
    MAMBA_AVAILABLE = True
except Exception:
    OfficialMamba = None
    MAMBA_AVAILABLE = False


class M3Rec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(M3Rec, self).__init__(config, dataset)

        self.config = config
        self.dataset_obj = dataset
        self.dataset_name = str(config["dataset"])
        self.data_path = os.path.abspath(str(config["data_path"]))
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, self.dataset_name))

        # Use dedicated keys to avoid conflicts with overall.yaml.
        self.hidden_size = int(config["m3rec_hidden_size"]) if "m3rec_hidden_size" in config else int(config["embedding_size"])
        self.max_seq_length = int(config["m3rec_max_seq_length"]) if "m3rec_max_seq_length" in config else 50
        self.num_layers = int(config["m3rec_num_layers"]) if "m3rec_num_layers" in config else 2
        self.dropout_prob = float(config["m3rec_dropout_prob"]) if "m3rec_dropout_prob" in config else 0.5
        self.loss_type = str(config["m3rec_loss_type"]) if "m3rec_loss_type" in config else "BPR"

        # Mamba hyperparameters.
        self.d_state = int(config["m3rec_d_state"]) if "m3rec_d_state" in config else 32
        self.d_conv = int(config["m3rec_d_conv"]) if "m3rec_d_conv" in config else 4
        self.expand = int(config["m3rec_expand"]) if "m3rec_expand" in config else 2

        self.initializer_range = float(config["m3rec_initializer_range"]) if "m3rec_initializer_range" in config else 0.02
        self.reg_weight = float(config["reg_weight"]) if "reg_weight" in config else 0.0
        self.freeze_mm_sequence = bool(config["m3rec_freeze_mm_sequence"]) if "m3rec_freeze_mm_sequence" in config else True
        self.strict_inter_file = bool(config["m3rec_strict_inter_file"]) if "m3rec_strict_inter_file" in config else True
        self.inter_file_name = config["inter_file_name"] if "inter_file_name" in config else None

        self.user_field = str(config["USER_ID_FIELD"]).split(":")[0] if "USER_ID_FIELD" in config else "userID"
        self.item_field = str(config["ITEM_ID_FIELD"]).split(":")[0] if "ITEM_ID_FIELD" in config else "itemID"
        self.time_field = str(config["TIME_FIELD"]).split(":")[0] if "TIME_FIELD" in config else "timestamp"
        self.split_field = str(config["inter_splitting_label"]).split(":")[0] if "inter_splitting_label" in config else "x_label"

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        if self.v_feat is not None:
            self.img_feat = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            img_dim = self.v_feat.shape[1]
        else:
            self.img_feat = None
            img_dim = self.hidden_size

        if self.t_feat is not None:
            self.text_feat = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            txt_dim = self.t_feat.shape[1]
        else:
            self.text_feat = None
            txt_dim = self.hidden_size

        self.img_alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.text_beta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        self.img_trans = nn.Sequential(
            nn.Linear(img_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.text_trans = nn.Sequential(
            nn.Linear(txt_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.img_LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.text_LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)

        # Shared Mamba blocks across ID / image / text streams.
        self.mamba_block = nn.ModuleList([
            MambaBlock(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            )
            for _ in range(self.num_layers)
        ])

        # Modality-specific feed-forward experts.
        self.id_mamba_ffn = nn.ModuleList([
            FeedForward(self.hidden_size, self.hidden_size * 4, self.dropout_prob)
            for _ in range(self.num_layers)
        ])
        self.img_mamba_ffn = nn.ModuleList([
            FeedForward(self.hidden_size, self.hidden_size * 4, self.dropout_prob)
            for _ in range(self.num_layers)
        ])
        self.text_mamba_ffn = nn.ModuleList([
            FeedForward(self.hidden_size, self.hidden_size * 4, self.dropout_prob)
            for _ in range(self.num_layers)
        ])

        seq_items, seq_lens = self._build_user_sequences()
        self.register_buffer("m3rec_user_seq_items", torch.LongTensor(seq_items))
        self.register_buffer("m3rec_user_seq_lens", torch.LongTensor(seq_lens))

        self.apply(self._init_weights)

        print("[M3Rec] dataset={}, data_path={}, inter_file={}".format(
            self.dataset_name, self.data_path, self._get_inter_path()
        ))
        print("[M3Rec] users={}, items={}, hidden_size={}, max_seq_len={}".format(
            self.n_users, self.n_items, self.hidden_size, self.max_seq_length
        ))
        print("[M3Rec] mamba_available={}, freeze_mm_sequence={}".format(
            MAMBA_AVAILABLE, self.freeze_mm_sequence
        ))
        print("[M3Rec] v_feat={}, t_feat={}".format(
            None if self.v_feat is None else tuple(self.v_feat.shape),
            None if self.t_feat is None else tuple(self.t_feat.shape),
        ))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "weight", None) is not None:
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
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
        return candidates[0]

    def _read_train_user_item_times(self):
        inter_path = self._get_inter_path()
        rows = []

        if not os.path.exists(inter_path):
            if self.strict_inter_file:
                raise FileNotFoundError("Cannot find inter file for M3Rec: {}".format(inter_path))
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

            # Robust fallback names.
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
                    # Your temporal split: x_label == 0 means train.
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
            for k, i in enumerate(items):
                seq_items[u, offset + k] = i

        return seq_items, seq_lens

    def _batch_sequences(self, users, target_items=None, remove_target=True):
        seq_items = self.m3rec_user_seq_items[users].clone()

        if target_items is not None and remove_target:
            # Remove one occurrence of current positive item from the right to reduce leakage.
            for b in range(seq_items.size(0)):
                tgt = int(target_items[b].detach().cpu().item())
                pos = torch.nonzero(seq_items[b] == tgt, as_tuple=False)
                if pos.numel() > 0:
                    seq_items[b, int(pos[-1].item())] = 0

        seq_lens = (seq_items != 0).long().sum(dim=1).clamp_min(1)
        return seq_items, seq_lens

    @staticmethod
    def gather_indexes(output, gather_index):
        # output: [B, L, H], gather_index: [B]
        gather_index = gather_index.view(-1, 1, 1).expand(-1, 1, output.size(-1))
        return output.gather(dim=1, index=gather_index).squeeze(1)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _image_sequence_embedding(self, item_seq):
        if self.img_feat is None:
            return torch.zeros(item_seq.size(0), item_seq.size(1), self.hidden_size, device=item_seq.device)
        return self.img_trans(self.img_feat(item_seq))

    def _text_sequence_embedding(self, item_seq):
        if self.text_feat is None:
            return torch.zeros(item_seq.size(0), item_seq.size(1), self.hidden_size, device=item_seq.device)
        return self.text_trans(self.text_feat(item_seq))

    def _image_item_table(self):
        if self.img_feat is None:
            return torch.zeros(self.n_items, self.hidden_size, device=self.device)
        return self.img_trans(self.img_feat.weight)

    def _text_item_table(self):
        if self.text_feat is None:
            return torch.zeros(self.n_items, self.hidden_size, device=self.device)
        return self.text_trans(self.text_feat.weight)

    # ------------------------------------------------------------------
    # Forward and scoring
    # ------------------------------------------------------------------
    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        img_emb = self._image_sequence_embedding(item_seq)
        text_emb = self._text_sequence_embedding(item_seq)

        item_emb = self.LayerNorm(self.dropout(item_emb))
        img_emb = self.img_LayerNorm(self.dropout(img_emb))
        text_emb = self.text_LayerNorm(self.dropout(text_emb))

        # ID stream: trainable.
        for i in range(self.num_layers):
            item_emb = self.mamba_block[i](item_emb)
            item_emb = self.id_mamba_ffn[i](item_emb)

        # Image/Text streams: official M3Rec uses no_grad here.
        if self.freeze_mm_sequence:
            with torch.no_grad():
                for i in range(self.num_layers):
                    img_emb = self.mamba_block[i](img_emb)
                    img_emb = self.img_mamba_ffn[i](img_emb)
                for i in range(self.num_layers):
                    text_emb = self.mamba_block[i](text_emb)
                    text_emb = self.text_mamba_ffn[i](text_emb)
        else:
            for i in range(self.num_layers):
                img_emb = self.mamba_block[i](img_emb)
                img_emb = self.img_mamba_ffn[i](img_emb)
            for i in range(self.num_layers):
                text_emb = self.mamba_block[i](text_emb)
                text_emb = self.text_mamba_ffn[i](text_emb)

        mm_emb = torch.cat((self.img_alpha * img_emb, self.text_beta * text_emb, item_emb), dim=-1)
        seq_output = self.gather_indexes(mm_emb, item_seq_len - 1)
        return seq_output

    def _candidate_table(self):
        return torch.cat(
            (
                self.img_alpha * self._image_item_table(),
                self.text_beta * self._text_item_table(),
                self.item_embedding.weight,
            ),
            dim=-1,
        )

    def _score_items(self, seq_output, item_ids):
        cand = torch.cat(
            (
                self.img_alpha * self._image_item_table()[item_ids],
                self.text_beta * self._text_item_table()[item_ids],
                self.item_embedding(item_ids),
            ),
            dim=-1,
        )
        return torch.mul(seq_output, cand).sum(dim=1)

    @staticmethod
    def bpr_loss(pos_score, neg_score):
        return -torch.mean(F.logsigmoid(pos_score - neg_score))

    # ------------------------------------------------------------------
    # MMRec API
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        users = interaction[0].long()
        pos_items = interaction[1].long()
        neg_items = interaction[2].long()

        item_seq, item_seq_len = self._batch_sequences(users, pos_items, remove_target=True)
        seq_output = self.forward(item_seq, item_seq_len)

        pos_score = self._score_items(seq_output, pos_items)
        neg_score = self._score_items(seq_output, neg_items)
        loss = self.bpr_loss(pos_score, neg_score)

        if self.reg_weight > 0.0:
            reg = (
                self.item_embedding(pos_items).pow(2).sum(dim=1)
                + self.item_embedding(neg_items).pow(2).sum(dim=1)
            ).mean()
            loss = loss + self.reg_weight * reg

        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0].long()
        item_seq, item_seq_len = self._batch_sequences(users, target_items=None, remove_target=False)
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self._candidate_table()
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class MambaFallback(nn.Module):
    """A lightweight fallback when mamba_ssm is unavailable.

    This is not a full Mamba implementation. It preserves the sequential
    state-space-style role so the model can run in environments without
    causal-conv1d / mamba-ssm.
    """
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        if MAMBA_AVAILABLE:
            self.mamba = OfficialMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = MambaFallback(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
