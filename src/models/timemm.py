# coding: utf-8
r"""
TimeMM adapted for MMRec.

File:
    src/models/TimeMM.py

Class/model name:
    TimeMM

Official TimeMM idea:
    Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation.
    This adapter keeps the core runnable components in MMRec:
    train-only timestamp-aware user-item graph, multi-scale temporal kernel bank,
    adaptive scale routing, multimodal item graph smoothing, modality routing,
    and optional scale-diversity regularization.
"""

import os
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class TimeMM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TimeMM, self).__init__(config, dataset)

        self.config = config
        self.dataset_obj = dataset
        self.dataset_name = str(config["dataset"])
        self.data_path = os.path.abspath(str(config["data_path"]))
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, self.dataset_name))

        self.hidden_size = int(config["timemm_hidden_size"]) if "timemm_hidden_size" in config else int(config["embedding_size"])
        self.feat_embed_dim = int(config["timemm_feat_embed_dim"]) if "timemm_feat_embed_dim" in config else self.hidden_size
        self.num_layers = int(config["timemm_num_layers"]) if "timemm_num_layers" in config else 2
        self.n_mm_layers = int(config["timemm_n_mm_layers"]) if "timemm_n_mm_layers" in config else 1
        self.knn_k = int(config["timemm_knn_k"]) if "timemm_knn_k" in config else 10
        self.mm_image_weight = float(config["timemm_mm_image_weight"]) if "timemm_mm_image_weight" in config else 0.1

        self.tau_age_list = list(config["timemm_tau_age_list"]) if "timemm_tau_age_list" in config else [4.0, 6.0, 10.0]
        self.transform = str(config["timemm_transform"]) if "timemm_transform" in config else "exp"
        self.use_age = bool(config["timemm_use_age"]) if "timemm_use_age" in config else True
        self.use_gap = bool(config["timemm_use_gap"]) if "timemm_use_gap" in config else False

        self.routing_hidden = int(config["timemm_routing_hidden"]) if "timemm_routing_hidden" in config else 64
        self.routing_temperature = float(config["timemm_routing_temperature"]) if "timemm_routing_temperature" in config else 0.7
        self.modality_routing_temperature = float(config["timemm_modality_routing_temperature"]) if "timemm_modality_routing_temperature" in config else 1.0
        self.dropout = float(config["timemm_dropout"]) if "timemm_dropout" in config else 0.0

        self.aux_weight = float(config["timemm_aux_weight"]) if "timemm_aux_weight" in config else 1e-2
        self.reg_weight = float(config["reg_weight"]) if "reg_weight" in config else 0.0
        self.reg_weight1 = float(config["timemm_reg_weight1"]) if "timemm_reg_weight1" in config else 1e-3
        self.reg_weight2 = float(config["timemm_reg_weight2"]) if "timemm_reg_weight2" in config else 1e-2

        self.strict_inter_file = bool(config["timemm_strict_inter_file"]) if "timemm_strict_inter_file" in config else True
        self.inter_file_name = config["inter_file_name"] if "inter_file_name" in config else None

        self.user_field = str(config["USER_ID_FIELD"]).split(":")[0] if "USER_ID_FIELD" in config else "userID"
        self.item_field = str(config["ITEM_ID_FIELD"]).split(":")[0] if "ITEM_ID_FIELD" in config else "itemID"
        self.time_field = str(config["TIME_FIELD"]).split(":")[0] if "TIME_FIELD" in config else "timestamp"
        self.split_field = str(config["inter_splitting_label"]).split(":")[0] if "inter_splitting_label" in config else "x_label"

        self.num_scales = len(self.tau_age_list)
        self.num_nodes = self.n_users + self.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.hidden_size)
        else:
            self.image_embedding = None
            self.image_trs = None

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.hidden_size)
        else:
            self.text_embedding = None
            self.text_trs = None

        self.id_ln = nn.LayerNorm(self.hidden_size)
        self.v_ln = nn.LayerNorm(self.hidden_size)
        self.t_ln = nn.LayerNorm(self.hidden_size)
        self.drop = nn.Dropout(self.dropout)

        train_rows = self._read_train_user_item_times()
        self.train_users_np, self.train_items_np, self.train_times_np = self._rows_to_numpy(train_rows)

        self.register_buffer("user_time_feat", self._build_user_time_features(train_rows).to(self.device))
        self.register_buffer("item_time_feat", self._build_item_time_features(train_rows).to(self.device))
        self.temporal_adjs = self._build_multiscale_temporal_adjs(train_rows)
        self.mm_adj = self._build_mm_adj()

        self.scale_gate = SharedScaleGate(
            user_in=self.user_time_feat.size(1),
            item_in=self.item_time_feat.size(1),
            hidden=self.routing_hidden,
            num_scales=self.num_scales,
            temperature=self.routing_temperature,
            dropout=self.dropout,
        )

        self.user_modality_gate = nn.Sequential(
            nn.Linear(self.user_time_feat.size(1), self.routing_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.routing_hidden, 3),
        )
        self.item_modality_gate = nn.Sequential(
            nn.Linear(self.item_time_feat.size(1), self.routing_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.routing_hidden, 3),
        )

        self.reset_parameters()

        print("[TimeMM] dataset={}, inter_file={}".format(self.dataset_name, self._get_inter_path()))
        print("[TimeMM] users={}, items={}, hidden_size={}, scales={}".format(
            self.n_users, self.n_items, self.hidden_size, self.tau_age_list
        ))
        print("[TimeMM] knn_k={}, mm_image_weight={}, num_layers={}, n_mm_layers={}".format(
            self.knn_k, self.mm_image_weight, self.num_layers, self.n_mm_layers
        ))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
                raise FileNotFoundError("Cannot find inter file for TimeMM: {}".format(inter_path))
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

    @staticmethod
    def _rows_to_numpy(rows):
        if len(rows) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        users = np.asarray([r[0] for r in rows], dtype=np.int64)
        items = np.asarray([r[1] for r in rows], dtype=np.int64)
        times = np.asarray([r[2] for r in rows], dtype=np.float32)
        return users, items, times

    def _zscore(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _build_user_time_features(self, rows):
        hist = defaultdict(list)
        for u, _, ts in rows:
            hist[u].append(float(ts))

        max_ts = max([r[2] for r in rows], default=1.0)
        q75 = np.quantile(self.train_times_np, 0.75) if len(self.train_times_np) > 0 else 0.0
        feats = torch.zeros(self.n_users, 7, dtype=torch.float32)
        for u in range(self.n_users):
            ts = sorted(hist.get(u, []))
            if len(ts) == 0:
                continue
            arr = np.asarray(ts, dtype=np.float32)
            gaps = np.diff(arr) if len(arr) > 1 else np.asarray([0.0], dtype=np.float32)
            span = float(arr[-1] - arr[0])
            recency = float(max_ts - arr[-1])
            density = float(len(arr) / (span + 1.0))
            feats[u] = torch.tensor([
                np.log1p(len(arr)),
                np.log1p(span),
                np.log1p(gaps.mean()),
                np.log1p(gaps.std()),
                np.log1p(recency),
                np.log1p(density),
                float(arr[-1] >= q75),
            ])
        return self._zscore(feats)

    def _build_item_time_features(self, rows):
        hist = defaultdict(list)
        for _, i, ts in rows:
            hist[i].append(float(ts))

        max_ts = max([r[2] for r in rows], default=1.0)
        q75 = np.quantile(self.train_times_np, 0.75) if len(self.train_times_np) > 0 else 0.0
        feats = torch.zeros(self.n_items, 7, dtype=torch.float32)
        for i in range(self.n_items):
            ts = sorted(hist.get(i, []))
            if len(ts) == 0:
                continue
            arr = np.asarray(ts, dtype=np.float32)
            gaps = np.diff(arr) if len(arr) > 1 else np.asarray([0.0], dtype=np.float32)
            span = float(arr[-1] - arr[0])
            recency = float(max_ts - arr[-1])
            late_pop = float((arr >= q75).sum())
            trend = late_pop / max(len(arr), 1)
            feats[i] = torch.tensor([
                np.log1p(len(arr)),
                np.log1p(span),
                np.log1p(gaps.mean()),
                np.log1p(gaps.std()),
                np.log1p(recency),
                np.log1p(trend),
                float(late_pop > 0),
            ])
        return self._zscore(feats)

    def _age_gap_vectors(self, rows):
        if len(rows) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        max_ts = max(r[2] for r in rows)
        age = np.asarray([max_ts - r[2] for r in rows], dtype=np.float32)
        age = np.log1p(np.maximum(age, 0.0))

        gap_by_pair = {}
        last_by_user = {}
        for u, i, ts in sorted(rows, key=lambda x: (x[0], x[2])):
            gap_by_pair[(u, i, ts)] = max(ts - last_by_user[u], 0.0) if u in last_by_user else 0.0
            last_by_user[u] = ts
        gap = np.asarray([gap_by_pair[(u, i, ts)] for u, i, ts in rows], dtype=np.float32)
        gap = np.log1p(gap)

        def norm01(v):
            vmax = float(v.max()) if v.size > 0 else 0.0
            return v / vmax if vmax > 0 else np.zeros_like(v)

        return norm01(age), norm01(gap)

    def _kernel_weight(self, x, tau):
        tau = max(float(tau), 1e-6)
        if self.transform == "exp":
            return np.exp(-x / tau)
        if self.transform == "log":
            return 1.0 / np.log1p(1.0 + x / tau)
        if self.transform == "inverse":
            return 1.0 / (1.0 + x / tau)
        return np.exp(-x / tau)

    def _normalize_sparse_adj(self, row, col, val, size):
        row_t = torch.LongTensor(row).to(self.device)
        col_t = torch.LongTensor(col).to(self.device)
        val_t = torch.FloatTensor(val).to(self.device).clamp_min(1e-8)

        deg = torch.zeros(size, device=self.device)
        deg.scatter_add_(0, row_t, val_t)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        norm_val = val_t * deg_inv_sqrt[row_t] * deg_inv_sqrt[col_t]

        idx = torch.stack([row_t, col_t], dim=0)
        return torch.sparse_coo_tensor(idx, norm_val, (size, size), device=self.device).coalesce()

    def _build_multiscale_temporal_adjs(self, rows):
        if len(rows) == 0:
            return []

        users, items, _ = self._rows_to_numpy(rows)
        item_nodes = items + self.n_users
        age, gap = self._age_gap_vectors(rows)

        signal = np.zeros_like(age)
        if self.use_age:
            signal += age
        if self.use_gap:
            signal += gap
        if not self.use_age and not self.use_gap:
            signal = age

        row_one = users
        col_one = item_nodes
        row = np.concatenate([row_one, col_one], axis=0)
        col = np.concatenate([col_one, row_one], axis=0)

        adjs = []
        for tau in self.tau_age_list:
            w = self._kernel_weight(signal, tau).astype(np.float32)
            val = np.concatenate([w, w], axis=0)
            adjs.append(self._normalize_sparse_adj(row, col, val, self.num_nodes))
        return adjs

    def _build_mm_adj(self):
        if self.v_feat is None and self.t_feat is None:
            return None

        # Raw image/text feature dimensions may differ, e.g. image=4096 and text=384.
        # Project each modality to the shared hidden space before fusing.
        feats = []
        if self.v_feat is not None:
            # _build_mm_adj() runs inside __init__ before quick_start calls
            # model.to(cuda). Therefore the projection layer may still be on
            # CPU even when config['device'] is cuda. Project on the layer's
            # actual device first, then move the result to self.device.
            v_proj_device = self.image_trs.weight.device
            v_raw = self.v_feat.float().to(v_proj_device)
            v = F.normalize(self.image_trs(v_raw), dim=1).to(self.device)
            feats.append(self.mm_image_weight * v)

        if self.t_feat is not None:
            t_proj_device = self.text_trs.weight.device
            t_raw = self.t_feat.float().to(t_proj_device)
            t = F.normalize(self.text_trs(t_raw), dim=1).to(self.device)
            if self.v_feat is not None:
                feats.append((1.0 - self.mm_image_weight) * t)
            else:
                feats.append(t)

        feat = F.normalize(torch.stack(feats, dim=0).sum(dim=0), dim=1)
        sim = torch.matmul(feat, feat.t())
        k = min(self.knn_k, self.n_items)
        _, knn = torch.topk(sim, k=k, dim=-1)
        row = torch.arange(self.n_items, device=self.device).view(-1, 1).expand(-1, k).reshape(-1)
        col = knn.reshape(-1)
        val = torch.ones_like(row, dtype=torch.float32)

        # Make the item-item KNN graph symmetric for more stable propagation.
        row_fwd = row
        col_fwd = col
        row = torch.cat([row_fwd, col_fwd], dim=0)
        col = torch.cat([col_fwd, row_fwd], dim=0)
        val = torch.cat([val, val], dim=0)

        deg = torch.zeros(self.n_items, device=self.device)
        deg.scatter_add_(0, row, val)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        norm_val = val * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        idx = torch.stack([row, col], dim=0)
        return torch.sparse_coo_tensor(idx, norm_val, (self.n_items, self.n_items), device=self.device).coalesce()

    def _propagate_ui_multiscale(self, user_init, item_init):
        node_init = torch.cat([user_init, item_init], dim=0)
        per_scale = []

        for adj in self.temporal_adjs:
            out = node_init
            all_layers = [out]
            for _ in range(self.num_layers):
                out = torch.sparse.mm(adj, out)
                all_layers.append(out)
            per_scale.append(sum(all_layers) / len(all_layers))

        if len(per_scale) == 0:
            per_scale = [node_init]
        return torch.stack(per_scale, dim=0)

    def _fuse_scales(self, per_scale):
        user_gate, item_gate = self.scale_gate(self.user_time_feat, self.item_time_feat)
        gate = torch.cat([user_gate, item_gate], dim=0).t().unsqueeze(-1)
        return (per_scale * gate).sum(dim=0), torch.cat([user_gate, item_gate], dim=0)

    def _smooth_item_mm(self):
        if self.image_embedding is not None:
            v_item = self.image_trs(self.image_embedding.weight)
        else:
            v_item = torch.zeros(self.n_items, self.hidden_size, device=self.device)

        if self.text_embedding is not None:
            t_item = self.text_trs(self.text_embedding.weight)
        else:
            t_item = torch.zeros(self.n_items, self.hidden_size, device=self.device)

        if self.mm_adj is not None:
            for _ in range(self.n_mm_layers):
                v_item = torch.sparse.mm(self.mm_adj, v_item)
                t_item = torch.sparse.mm(self.mm_adj, t_item)

        return self.v_ln(v_item), self.t_ln(t_item)

    def forward(self):
        user_id = self.id_ln(self.user_embedding.weight)
        item_id = self.id_ln(self.item_embedding.weight)

        id_per_scale = self._propagate_ui_multiscale(user_id, item_id)
        id_node, scale_gate = self._fuse_scales(id_per_scale)
        id_user = id_node[:self.n_users]
        id_item = id_node[self.n_users:]

        v_item, t_item = self._smooth_item_mm()
        zero_user = torch.zeros(self.n_users, self.hidden_size, device=self.device)

        v_per_scale = self._propagate_ui_multiscale(zero_user, v_item)
        t_per_scale = self._propagate_ui_multiscale(zero_user, t_item)
        v_node, _ = self._fuse_scales(v_per_scale)
        t_node, _ = self._fuse_scales(t_per_scale)

        v_user, v_item = v_node[:self.n_users], v_node[self.n_users:]
        t_user, t_item = t_node[:self.n_users], t_node[self.n_users:]

        user_mod = F.softmax(
            self.user_modality_gate(self.user_time_feat) / max(self.modality_routing_temperature, 1e-6),
            dim=-1,
        )
        item_mod = F.softmax(
            self.item_modality_gate(self.item_time_feat) / max(self.modality_routing_temperature, 1e-6),
            dim=-1,
        )

        user_rep = user_mod[:, 0:1] * id_user + user_mod[:, 1:2] * v_user + user_mod[:, 2:3] * t_user
        item_rep = item_mod[:, 0:1] * id_item + item_mod[:, 1:2] * v_item + item_mod[:, 2:3] * t_item

        user_rep = self.drop(F.normalize(user_rep, dim=1))
        item_rep = self.drop(F.normalize(item_rep, dim=1))

        return user_rep, item_rep, id_per_scale, scale_gate, user_mod, item_mod

    @staticmethod
    def bpr_loss(pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def _scale_diversity_loss(self, id_per_scale, users, pos_items, neg_items):
        if self.aux_weight <= 0.0 or id_per_scale.size(0) <= 1:
            return torch.tensor(0.0, device=self.device)

        scale_user = id_per_scale[:, users, :]
        scale_item = id_per_scale[:, self.n_users + pos_items, :]
        scale_neg = id_per_scale[:, self.n_users + neg_items, :]

        margins = (scale_user * scale_item).sum(dim=-1) - (scale_user * scale_neg).sum(dim=-1)
        margins = margins - margins.mean(dim=1, keepdim=True)
        margins = F.normalize(margins, dim=1)
        corr = torch.matmul(margins, margins.t())
        eye = torch.eye(corr.size(0), device=corr.device)
        return ((corr - eye) ** 2).mean()

    def _regularization(self, scale_gate, user_mod, item_mod):
        reg = torch.tensor(0.0, device=self.device)
        if self.reg_weight1 > 0:
            eps = 1e-12
            ent_scale = -(scale_gate.clamp_min(eps) * scale_gate.clamp_min(eps).log()).sum(dim=1).mean()
            ent_user = -(user_mod.clamp_min(eps) * user_mod.clamp_min(eps).log()).sum(dim=1).mean()
            ent_item = -(item_mod.clamp_min(eps) * item_mod.clamp_min(eps).log()).sum(dim=1).mean()
            reg = reg - self.reg_weight1 * (ent_scale + 0.5 * ent_user + 0.5 * ent_item)
        if self.reg_weight2 > 0:
            reg = reg + self.reg_weight2 * (self.user_embedding.weight.pow(2).mean() + self.item_embedding.weight.pow(2).mean())
        return reg

    def calculate_loss(self, interaction):
        users = interaction[0].long()
        pos_items = interaction[1].long()
        neg_items = interaction[2].long()

        user_rep, item_rep, id_per_scale, scale_gate, user_mod, item_mod = self.forward()

        u = user_rep[users]
        pos = item_rep[pos_items]
        neg = item_rep[neg_items]

        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)

        loss = self.bpr_loss(pos_scores, neg_scores)
        loss = loss + self.aux_weight * self._scale_diversity_loss(id_per_scale, users, pos_items, neg_items)
        loss = loss + self._regularization(scale_gate, user_mod, item_mod)

        if self.reg_weight > 0:
            loss = loss + self.reg_weight * (
                self.user_embedding(users).pow(2).sum(dim=1).mean()
                + self.item_embedding(pos_items).pow(2).sum(dim=1).mean()
                + self.item_embedding(neg_items).pow(2).sum(dim=1).mean()
            )

        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0].long()
        user_rep, item_rep, _, _, _, _ = self.forward()
        return torch.matmul(user_rep[users], item_rep.t())


class SharedScaleGate(nn.Module):
    def __init__(self, user_in, item_in, hidden, num_scales, temperature=1.0, dropout=0.0):
        super().__init__()
        self.temperature = float(max(temperature, 1e-6))
        self.user_stem = nn.Sequential(
            nn.Linear(user_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_scales),
        )
        self.item_stem = nn.Sequential(
            nn.Linear(item_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_scales),
        )

    def forward(self, user_feat, item_feat):
        user_gate = F.softmax(self.user_stem(user_feat) / self.temperature, dim=-1)
        item_gate = F.softmax(self.item_stem(item_feat) / self.temperature, dim=-1)
        return user_gate, item_gate