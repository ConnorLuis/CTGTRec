# coding: utf-8
r"""
CTGTRec: Temporal Trend-aware Score Calibration with normalization ablation.

Second-stage experiment G3-score.

Purpose
-------
Diagnose whether CT-FREEDOM_tau0p03 gains come partly from global temporal trend.

Base prediction:
    score_ct(u,i)

Trend-calibrated prediction:
    boost : score'(u,i) = score_ct(u,i) + lambda * trend_i
    debias: score'(u,i) = score_ct(u,i) - lambda * trend_i

trend_i is computed from train interactions only:
    trend_i = (pop_i_late / |E_late|) / (pop_i_all / |E_all| + eps)

Then log-normalized and z-normalized for score calibration.

Placement:
    src/models/ct_freedom_g3_score_final.py
"""

import os
import csv
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class CTGTRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CTGTRec, self).__init__(config, dataset)

        self.config = config
        self.dataset_name = str(config["dataset"])
        self.dataset_path = os.path.abspath(os.path.join(config["data_path"], config["dataset"]))

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.cf_model = config["cf_model"]
        self.n_layers = config["n_mm_layers"]
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]
        self.mm_image_weight = config["mm_image_weight"]
        self.dropout = config["dropout"]
        self.degree_ratio = config["degree_ratio"]

        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        self.ct_adj_dir = config["ct_adj_dir"] if "ct_adj_dir" in config else "continuous_time_adj"
        self.ct_time_adj_file = config["ct_time_adj_file"] if "ct_time_adj_file" in config else "ct_adj_user_tau0p03.npz"
        self.g3_score_mode = str(config["g3_score_mode"]).lower()
        if self.g3_score_mode not in {"base", "boost", "debias", "trend_only"}:
            raise ValueError("g3_score_mode must be base/boost/debias/trend_only, got {}".format(self.g3_score_mode))
        self.g3_score_lambda = float(config["g3_score_lambda"]) if "g3_score_lambda" in config else 0.0
        self.g3_late_ratio = float(config["g3_late_ratio"]) if "g3_late_ratio" in config else 0.25
        self.g3_trend_norm = str(config["g3_trend_norm"]).lower() if "g3_trend_norm" in config else "zscore_clip"
        if self.g3_trend_norm not in {"zscore_clip", "minmax", "rank", "log_only"}:
            raise ValueError("g3_trend_norm must be zscore_clip/minmax/rank/log_only, got {}".format(self.g3_trend_norm))
        self.g3_inter_file = config["g3_inter_file"] if "g3_inter_file" in config else None
        self.g3_strict_inter_file = bool(config["g3_strict_inter_file"]) if "g3_strict_inter_file" in config else True
        self.g3_disable_dropout = bool(config["g3_disable_dropout"]) if "g3_disable_dropout" in config else True

        self.norm_adj = self.load_ui_adj(self.ct_time_adj_file).to(self.device)
        self.masked_adj = self.norm_adj

        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.mm_adj = None
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        mm_adj_file = os.path.join(
            self.dataset_path,
            "mm_adj_freedomdsp_{}_{}.pt".format(self.knn_k, int(10 * self.mm_image_weight)),
        )
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location="cpu")
        else:
            image_adj, text_adj = None, None
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach().to(self.device))
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach().to(self.device))
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del image_adj
                del text_adj
            if self.mm_adj is not None:
                torch.save(self.mm_adj.cpu(), mm_adj_file)
        if self.mm_adj is not None:
            self.mm_adj = self.mm_adj.coalesce().to(self.device)

        trend = self.build_item_trend()
        self.register_buffer("g3_item_trend", torch.FloatTensor(trend))

        print("[CT_FREEDOM_G3_SCORE_FINAL_FINAL] dataset_path = {}".format(self.dataset_path))
        print("[CT_FREEDOM_G3_SCORE_FINAL_FINAL] mode = {}, lambda = {}, norm = {}".format(
            self.g3_score_mode, self.g3_score_lambda, self.g3_trend_norm
        ))
        print("[CT_FREEDOM_G3_SCORE_FINAL_FINAL] trend stats mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
            float(trend.mean()), float(trend.std()), float(trend.min()), float(trend.max())
        ))

    # ------------------------------------------------------------------
    # Train inter parsing and trend
    # ------------------------------------------------------------------
    @staticmethod
    def _find_col(header, target):
        for i, name in enumerate(header):
            if name == target or name.startswith(target + ":"):
                return i
        return -1

    def _get_inter_path(self):
        if self.g3_inter_file not in {None, "", "none", "None", "null", "NULL"}:
            inter_name = str(self.g3_inter_file)
        elif "inter_file_name" in self.config and self.config["inter_file_name"] not in {None, "", "none", "None", "null", "NULL"}:
            inter_name = str(self.config["inter_file_name"])
        else:
            inter_name = "{}.inter".format(self.dataset_name)

        if os.path.isabs(inter_name):
            return inter_name
        return os.path.join(self.dataset_path, inter_name)

    def _read_train_item_times(self):
        inter_path = self._get_inter_path()
        if not os.path.exists(inter_path):
            if self.g3_strict_inter_file:
                raise FileNotFoundError("Cannot find inter file for G3: {}".format(inter_path))
            coo = self.interaction_matrix.tocoo()
            return [(int(i), float(idx)) for idx, i in enumerate(coo.col)]

        pairs = []
        with open(inter_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            i_col = self._find_col(header, "itemID")
            t_col = self._find_col(header, "timestamp")
            label_col = self._find_col(header, "x_label")
            if i_col < 0 or t_col < 0:
                raise ValueError("Cannot find itemID/timestamp columns in {}".format(inter_path))

            for row in reader:
                if len(row) <= max(i_col, t_col, label_col):
                    continue
                if label_col >= 0:
                    try:
                        label = int(float(row[label_col]))
                    except Exception:
                        continue
                    if label != 0:
                        continue
                try:
                    i = int(float(row[i_col]))
                    ts = float(row[t_col])
                except Exception:
                    continue
                if 0 <= i < self.n_items:
                    pairs.append((i, ts))
        return pairs

    def build_item_trend(self):
        pairs = self._read_train_item_times()
        if len(pairs) == 0:
            return np.zeros(self.n_items, dtype=np.float32)

        items = np.asarray([p[0] for p in pairs], dtype=np.int64)
        times = np.asarray([p[1] for p in pairs], dtype=np.float64)

        threshold = np.quantile(times, 1.0 - self.g3_late_ratio)
        late_mask = times >= threshold

        pop_all = np.bincount(items, minlength=self.n_items).astype(np.float64)
        pop_late = np.bincount(items[late_mask], minlength=self.n_items).astype(np.float64)

        total_all = max(float(len(items)), 1.0)
        total_late = max(float(late_mask.sum()), 1.0)

        rate_all = pop_all / total_all
        rate_late = pop_late / total_late
        trend = rate_late / (rate_all + 1e-12)

        # Normalize trend for score calibration.
        # zscore_clip is the original G3 setting:
        #     log1p -> z-score -> clip[-3,3]
        # minmax:
        #     log1p -> [0,1], preserves monotonic trend as positive prior.
        # rank:
        #     percentile rank in [0,1], robust to extreme trend items.
        # log_only:
        #     log1p only, mainly diagnostic; lambda is not directly comparable.
        trend = np.log1p(trend)

        if self.g3_trend_norm == "zscore_clip":
            mean, std = trend.mean(), trend.std()
            if std < 1e-8:
                std = 1.0
            trend = (trend - mean) / std
            trend = np.clip(trend, -3.0, 3.0)

        elif self.g3_trend_norm == "minmax":
            mn, mx = trend.min(), trend.max()
            if mx - mn < 1e-12:
                trend = np.zeros_like(trend)
            else:
                trend = (trend - mn) / (mx - mn)

        elif self.g3_trend_norm == "rank":
            order = np.argsort(trend, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(trend), dtype=np.float64)
            denom = max(float(len(trend) - 1), 1.0)
            trend = ranks / denom

        elif self.g3_trend_norm == "log_only":
            # Keep raw log-scale ratio. Useful only as a diagnostic because scale
            # differs from normalized variants.
            pass

        return trend.astype(np.float32)

    # ------------------------------------------------------------------
    # Sparse graph loading
    # ------------------------------------------------------------------
    def _adj_path(self, filename):
        if os.path.isabs(str(filename)):
            return str(filename)
        return os.path.join(self.dataset_path, self.ct_adj_dir, str(filename))

    def load_ui_adj(self, filename):
        path = self._adj_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError("Cannot find UI adjacency file: {}".format(path))
        mat = sp.load_npz(path).astype(np.float32).tocoo()
        indices = torch.LongTensor(np.vstack([mat.row, mat.col]))
        values = torch.FloatTensor(mat.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(mat.shape)).coalesce()

    # FREEDOM utils
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size).to(self.device)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size).coalesce()

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        return r_inv_sqrt[indices[0]] * c_inv_sqrt[indices[1]]

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def pre_epoch_processing(self):
        if self.dropout <= 0.0 or self.g3_disable_dropout:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _mm_item_embedding(self):
        if self.mm_adj is None:
            return torch.zeros_like(self.item_id_embedding.weight)
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        final_item_embeddings = i_g_embeddings + self._mm_item_embedding()
        return u_g_embeddings, final_item_embeddings

    @staticmethod
    def bpr_loss_by_scores(pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        return self.bpr_loss_by_scores(pos_scores, neg_scores)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        main_loss = self.bpr_loss(ua_embeddings[users], ia_embeddings[pos_items], ia_embeddings[neg_items])

        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        return main_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        scores = torch.matmul(restore_user_e[users], restore_item_e.transpose(0, 1))

        trend = self.g3_item_trend.to(self.device).view(1, -1)
        if self.g3_score_mode == "boost":
            scores = scores + self.g3_score_lambda * trend
        elif self.g3_score_mode == "debias":
            scores = scores - self.g3_score_lambda * trend
        elif self.g3_score_mode == "trend_only":
            scores = trend.expand_as(scores)
        return scores
