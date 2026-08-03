# coding: utf-8
r"""CTGTRec: continuous-time graph learning and trend-aware calibration.

The user-item graph is constructed from training interactions only. During
training, optional edge dropout operates on the raw continuous-time edge
weights, preserves the retained temporal weights, and symmetrically
renormalizes the sampled weighted graph. Full-ranking evaluation always uses
the complete normalized continuous-time graph.
"""

from __future__ import annotations

import csv
import os
from typing import Tuple

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
        self.dataset_path = os.path.abspath(
            os.path.join(config["data_path"], config["dataset"])
        )

        self.embedding_dim = int(config["embedding_size"])
        self.feat_embed_dim = int(config["feat_embed_dim"])
        self.knn_k = int(config["knn_k"])
        self.lambda_coeff = float(config["lambda_coeff"])
        self.cf_model = config["cf_model"]
        self.n_layers = int(config["n_mm_layers"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.reg_weight = float(config["reg_weight"])
        self.mm_image_weight = float(config["mm_image_weight"])
        self.dropout = float(config["dropout"])
        self.degree_ratio = float(config["degree_ratio"])
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1), received {}".format(self.dropout)
            )

        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(
            np.float32
        )

        self.ct_adj_dir = (
            config["ct_adj_dir"]
            if "ct_adj_dir" in config
            else "continuous_time_adj"
        )
        self.ct_time_adj_file = (
            config["ct_time_adj_file"]
            if "ct_time_adj_file" in config
            else "ct_adj_user_tau0p03.npz"
        )
        self.ct_raw_adj_file = (
            config["ct_raw_adj_file"]
            if "ct_raw_adj_file" in config
            else self._derive_raw_adj_filename(self.ct_time_adj_file)
        )

        self.g3_score_mode = str(config["g3_score_mode"]).lower()
        if self.g3_score_mode not in {"base", "boost", "debias", "trend_only"}:
            raise ValueError(
                "g3_score_mode must be base/boost/debias/trend_only, got {}".format(
                    self.g3_score_mode
                )
            )
        self.g3_score_lambda = float(config["g3_score_lambda"] or 0.0)
        self.g3_late_ratio = float(config["g3_late_ratio"] or 0.25)
        if not 0.0 < self.g3_late_ratio <= 1.0:
            raise ValueError(
                "g3_late_ratio must be in (0, 1], got {}".format(
                    self.g3_late_ratio
                )
            )
        self.g3_trend_norm = str(config["g3_trend_norm"]).lower()
        if self.g3_trend_norm not in {
            "zscore_clip",
            "minmax",
            "rank",
            "log_only",
        }:
            raise ValueError(
                "g3_trend_norm must be zscore_clip/minmax/rank/log_only, got {}".format(
                    self.g3_trend_norm
                )
            )
        self.g3_inter_file = (
            config["g3_inter_file"] if "g3_inter_file" in config else None
        )
        self.g3_strict_inter_file = (
            bool(config["g3_strict_inter_file"])
            if "g3_strict_inter_file" in config
            else True
        )

        (
            self.norm_adj,
            self.ui_edge_indices,
            self.ui_edge_raw_weights,
            self.ui_edge_sampling_scores,
        ) = self.load_continuous_time_graphs()
        self.norm_adj = self.norm_adj.to(self.device)
        self.ui_edge_indices = self.ui_edge_indices.to(self.device)
        self.ui_edge_raw_weights = self.ui_edge_raw_weights.to(self.device)
        self.ui_edge_sampling_scores = self.ui_edge_sampling_scores.to(
            self.device
        )
        self.masked_adj = self.norm_adj

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.mm_adj = None
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat,
                freeze=False,
            )
            self.image_trs = nn.Linear(
                self.v_feat.shape[1],
                self.feat_embed_dim,
            )
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat,
                freeze=False,
            )
            self.text_trs = nn.Linear(
                self.t_feat.shape[1],
                self.feat_embed_dim,
            )

        mm_adj_file = os.path.join(
            self.dataset_path,
            "mm_adj_freedomdsp_{}_{}.pt".format(
                self.knn_k,
                int(10 * self.mm_image_weight),
            ),
        )
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location="cpu")
        else:
            image_adj, text_adj = None, None
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(
                    self.image_embedding.weight.detach().to(self.device)
                )
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(
                    self.text_embedding.weight.detach().to(self.device)
                )
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = (
                    self.mm_image_weight * image_adj
                    + (1.0 - self.mm_image_weight) * text_adj
                )
            if self.mm_adj is not None:
                torch.save(self.mm_adj.cpu(), mm_adj_file)
        if self.mm_adj is not None:
            self.mm_adj = self.mm_adj.coalesce().to(self.device)

        trend = self.build_item_trend()
        self.register_buffer("g3_item_trend", torch.tensor(trend))

        print("[CTGTRec] dataset_path = {}".format(self.dataset_path))
        print(
            "[CTGTRec] trend mode = {}, lambda = {}, norm = {}".format(
                self.g3_score_mode,
                self.g3_score_lambda,
                self.g3_trend_norm,
            )
        )
        print(
            "[CTGTRec] UI dropout = {:.3f}, weighted edges = {}".format(
                self.dropout,
                self.ui_edge_raw_weights.numel(),
            )
        )

    # ------------------------------------------------------------------
    # Train-only item trend
    # ------------------------------------------------------------------
    @staticmethod
    def _find_col(header, target):
        for index, name in enumerate(header):
            if name == target or name.startswith(target + ":"):
                return index
        return -1

    def _get_inter_path(self):
        null_values = {None, "", "none", "None", "null", "NULL"}
        if self.g3_inter_file not in null_values:
            inter_name = str(self.g3_inter_file)
        elif (
            "inter_file_name" in self.config
            and self.config["inter_file_name"] not in null_values
        ):
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
                raise FileNotFoundError(
                    "Cannot find temporal interaction file: {}".format(inter_path)
                )
            coo = self.interaction_matrix.tocoo()
            return [
                (int(item), float(index))
                for index, item in enumerate(coo.col)
            ]

        pairs = []
        with open(inter_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            item_col = self._find_col(header, "itemID")
            time_col = self._find_col(header, "timestamp")
            label_col = self._find_col(header, "x_label")
            if item_col < 0 or time_col < 0 or label_col < 0:
                raise ValueError(
                    "Cannot find itemID/timestamp/x_label columns in {}".format(
                        inter_path
                    )
                )

            for line_number, row in enumerate(reader, start=2):
                if len(row) <= max(item_col, time_col, label_col):
                    raise ValueError(
                        "Malformed row {} in {}".format(line_number, inter_path)
                    )
                try:
                    label = int(float(row[label_col]))
                    item = int(float(row[item_col]))
                    timestamp = float(row[time_col])
                except ValueError as exc:
                    raise ValueError(
                        "Invalid item/timestamp/label at {}:{}".format(
                            inter_path,
                            line_number,
                        )
                    ) from exc
                if label == 0:
                    if not 0 <= item < self.n_items:
                        raise ValueError(
                            "itemID {} out of range at {}:{}".format(
                                item,
                                inter_path,
                                line_number,
                            )
                        )
                    if not np.isfinite(timestamp):
                        raise ValueError(
                            "Non-finite timestamp at {}:{}".format(
                                inter_path,
                                line_number,
                            )
                        )
                    pairs.append((item, timestamp))
        return pairs

    @staticmethod
    def _linear_quantile(values, quantile):
        try:
            return np.quantile(values, quantile, method="linear")
        except TypeError:  # NumPy < 1.22
            return np.quantile(values, quantile, interpolation="linear")

    def build_item_trend(self):
        pairs = self._read_train_item_times()
        if len(pairs) == 0:
            raise ValueError("No training interactions available for item trend.")

        items = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        times = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        threshold = self._linear_quantile(
            times,
            1.0 - self.g3_late_ratio,
        )
        late_mask = times >= threshold

        pop_all = np.bincount(items, minlength=self.n_items).astype(np.float64)
        pop_late = np.bincount(
            items[late_mask],
            minlength=self.n_items,
        ).astype(np.float64)

        rate_all = pop_all / float(len(items))
        rate_late = pop_late / float(late_mask.sum())
        trend = rate_late / (rate_all + 1e-12)
        trend = np.log1p(trend)

        if self.g3_trend_norm == "zscore_clip":
            mean, std = trend.mean(), trend.std()
            if std < 1e-8:
                std = 1.0
            trend = np.clip((trend - mean) / std, -3.0, 3.0)
        elif self.g3_trend_norm == "minmax":
            minimum, maximum = trend.min(), trend.max()
            if maximum - minimum < 1e-12:
                trend = np.zeros_like(trend)
            else:
                trend = (trend - minimum) / (maximum - minimum)
        elif self.g3_trend_norm == "rank":
            order = np.argsort(trend, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(trend), dtype=np.float64)
            trend = ranks / max(float(len(trend) - 1), 1.0)

        return trend.astype(np.float32)

    # ------------------------------------------------------------------
    # Continuous-time weighted user-item graph
    # ------------------------------------------------------------------
    @staticmethod
    def _derive_raw_adj_filename(normalized_filename):
        filename = str(normalized_filename)
        basename = os.path.basename(filename)
        if not basename.startswith("ct_adj_"):
            raise ValueError(
                "Cannot derive raw adjacency filename from {}. Set "
                "ct_raw_adj_file explicitly.".format(filename)
            )
        raw_basename = "ct_raw_adj_" + basename[len("ct_adj_") :]
        directory = os.path.dirname(filename)
        return os.path.join(directory, raw_basename) if directory else raw_basename

    def _adj_path(self, filename):
        if os.path.isabs(str(filename)):
            return str(filename)
        return os.path.join(self.dataset_path, self.ct_adj_dir, str(filename))

    @staticmethod
    def _validate_scipy_graph(matrix, path, expected_shape):
        matrix = matrix.astype(np.float32).tocsr()
        if matrix.shape != expected_shape:
            raise ValueError(
                "{} has shape {}, expected {}".format(
                    path,
                    matrix.shape,
                    expected_shape,
                )
            )
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError("{} contains non-finite values".format(path))
        if matrix.data.size and (matrix.data < 0).any():
            raise ValueError("{} contains negative values".format(path))
        difference = matrix - matrix.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-6:
            raise ValueError("{} is not symmetric".format(path))
        return matrix

    @staticmethod
    def _symmetric_normalize_scipy(raw_adjacency):
        degree = np.asarray(raw_adjacency.sum(axis=1)).reshape(-1)
        inverse_sqrt = np.zeros_like(degree, dtype=np.float64)
        nonzero = degree > 0
        inverse_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
        diagonal = sp.diags(inverse_sqrt.astype(np.float32), format="csr")
        normalized = diagonal @ raw_adjacency @ diagonal
        normalized = normalized.tocsr().astype(np.float32)
        normalized.eliminate_zeros()
        return normalized

    @staticmethod
    def _scipy_to_torch_sparse(matrix):
        matrix = matrix.tocoo()
        indices = torch.from_numpy(
            np.vstack([matrix.row, matrix.col]).astype(np.int64)
        )
        values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.shape,
            dtype=torch.float32,
        ).coalesce()

    def load_continuous_time_graphs(self) -> Tuple[torch.Tensor, ...]:
        normalized_path = self._adj_path(self.ct_time_adj_file)
        raw_path = self._adj_path(self.ct_raw_adj_file)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(
                "Cannot find normalized UI adjacency: {}".format(normalized_path)
            )
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                "Cannot find raw weighted UI adjacency: {}. Regenerate graphs "
                "with preprocessing/build_continuous_time_adj.py.".format(
                    raw_path
                )
            )

        expected_shape = (self.n_nodes, self.n_nodes)
        raw = self._validate_scipy_graph(
            sp.load_npz(raw_path),
            raw_path,
            expected_shape,
        )
        normalized = self._validate_scipy_graph(
            sp.load_npz(normalized_path),
            normalized_path,
            expected_shape,
        )
        expected_normalized = self._symmetric_normalize_scipy(raw)
        difference = expected_normalized - normalized
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-5:
            raise ValueError(
                "Normalized graph does not match the raw temporal graph: {} "
                "vs {}".format(normalized_path, raw_path)
            )

        raw_coo = raw.tocoo()
        invalid_block = (
            ((raw_coo.row < self.n_users) & (raw_coo.col < self.n_users))
            | (
                (raw_coo.row >= self.n_users)
                & (raw_coo.col >= self.n_users)
            )
        )
        if invalid_block.any():
            raise ValueError("Raw UI adjacency contains non-bipartite edges.")

        forward_mask = (
            (raw_coo.row < self.n_users)
            & (raw_coo.col >= self.n_users)
        )
        users = raw_coo.row[forward_mask].astype(np.int64)
        items = (raw_coo.col[forward_mask] - self.n_users).astype(np.int64)
        raw_weights = raw_coo.data[forward_mask].astype(np.float32)
        if len(users) == 0:
            raise ValueError("Raw UI adjacency contains no user-item edges.")
        if raw.nnz != 2 * len(users):
            raise ValueError(
                "Raw UI adjacency must contain exactly two directions per "
                "aggregated user-item edge."
            )

        normalized_csr = normalized.tocsr()
        sampling_scores = np.asarray(
            normalized_csr[users, self.n_users + items]
        ).reshape(-1).astype(np.float32)
        if (
            not np.isfinite(sampling_scores).all()
            or (sampling_scores <= 0).any()
        ):
            raise ValueError("Invalid edge-sampling scores in normalized graph.")

        edge_indices = torch.from_numpy(np.vstack([users, items]))
        raw_weight_tensor = torch.from_numpy(raw_weights)
        score_tensor = torch.from_numpy(sampling_scores)
        return (
            self._scipy_to_torch_sparse(normalized),
            edge_indices,
            raw_weight_tensor,
            score_tensor,
        )

    def _normalize_sampled_weighted_edges(
        self,
        edge_indices,
        raw_weights,
    ):
        users = edge_indices[0]
        shifted_items = self.n_users + edge_indices[1]
        degree = torch.zeros(
            self.n_nodes,
            dtype=raw_weights.dtype,
            device=raw_weights.device,
        )
        degree.index_add_(0, users, raw_weights)
        degree.index_add_(0, shifted_items, raw_weights)

        normalized_weights = raw_weights * torch.rsqrt(degree[users])
        normalized_weights = normalized_weights * torch.rsqrt(
            degree[shifted_items]
        )
        forward = torch.stack([users, shifted_items], dim=0)
        reverse = torch.stack([shifted_items, users], dim=0)
        indices = torch.cat([forward, reverse], dim=1)
        values = torch.cat([normalized_weights, normalized_weights], dim=0)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.n_nodes, self.n_nodes),
            device=raw_weights.device,
            dtype=raw_weights.dtype,
        ).coalesce()

    def pre_epoch_processing(self):
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return

        edge_count = self.ui_edge_raw_weights.numel()
        keep_count = max(1, int(edge_count * (1.0 - self.dropout)))
        selected = torch.multinomial(
            self.ui_edge_sampling_scores,
            keep_count,
            replacement=False,
        )
        kept_indices = self.ui_edge_indices[:, selected]
        kept_raw_weights = self.ui_edge_raw_weights[selected]
        self.masked_adj = self._normalize_sampled_weighted_edges(
            kept_indices,
            kept_raw_weights,
        )

    # ------------------------------------------------------------------
    # Frozen multimodal graph and recommendation objective
    # ------------------------------------------------------------------
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        )
        similarity = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_indices = torch.topk(similarity, self.knn_k, dim=-1)
        adjacency_size = similarity.size()
        del similarity

        rows = torch.arange(knn_indices.shape[0], device=self.device)
        rows = rows.unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack(
            [torch.flatten(rows), torch.flatten(knn_indices)],
            dim=0,
        )
        return indices, self.compute_normalized_laplacian(
            indices,
            adjacency_size,
        )

    def compute_normalized_laplacian(self, indices, adjacency_size):
        values = torch.ones(
            indices.shape[1],
            dtype=torch.float32,
            device=indices.device,
        )
        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            adjacency_size,
            device=indices.device,
        ).coalesce()
        degree = torch.sparse.sum(adjacency, dim=1).to_dense()
        inverse_sqrt = torch.rsqrt(degree.clamp_min(1e-7))
        normalized_values = (
            inverse_sqrt[indices[0]] * inverse_sqrt[indices[1]]
        )
        return torch.sparse_coo_tensor(
            indices,
            normalized_values,
            adjacency_size,
            device=indices.device,
        ).coalesce()

    def _mm_item_embedding(self):
        if self.mm_adj is None:
            return torch.zeros_like(self.item_id_embedding.weight)
        embedding = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            embedding = torch.sparse.mm(self.mm_adj, embedding)
        return embedding

    def forward(self, adjacency):
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_id_embedding.weight],
            dim=0,
        )
        all_embeddings = [embeddings]
        for _ in range(self.n_ui_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            all_embeddings.append(embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(
            all_embeddings,
            [self.n_users, self.n_items],
            dim=0,
        )
        return user_embeddings, item_embeddings + self._mm_item_embedding()

    @staticmethod
    def bpr_loss_by_scores(positive_scores, negative_scores):
        return -torch.mean(
            F.logsigmoid(positive_scores - negative_scores)
        )

    def bpr_loss(self, users, positive_items, negative_items):
        positive_scores = torch.sum(users * positive_items, dim=1)
        negative_scores = torch.sum(users * negative_items, dim=1)
        return self.bpr_loss_by_scores(positive_scores, negative_scores)

    def calculate_loss(self, interaction):
        users = interaction[0]
        positive_items = interaction[1]
        negative_items = interaction[2]

        user_embeddings, item_embeddings = self.forward(self.masked_adj)
        main_loss = self.bpr_loss(
            user_embeddings[users],
            item_embeddings[positive_items],
            item_embeddings[negative_items],
        )

        image_loss, text_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_features = self.text_trs(self.text_embedding.weight)
            text_loss = self.bpr_loss(
                user_embeddings[users],
                text_features[positive_items],
                text_features[negative_items],
            )
        if self.v_feat is not None:
            image_features = self.image_trs(self.image_embedding.weight)
            image_loss = self.bpr_loss(
                user_embeddings[users],
                image_features[positive_items],
                image_features[negative_items],
            )
        return main_loss + self.reg_weight * (text_loss + image_loss)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embeddings, item_embeddings = self.forward(self.norm_adj)
        scores = torch.matmul(
            user_embeddings[users],
            item_embeddings.transpose(0, 1),
        )

        trend = self.g3_item_trend.to(self.device).view(1, -1)
        if self.g3_score_mode == "boost":
            scores = scores + self.g3_score_lambda * trend
        elif self.g3_score_mode == "debias":
            scores = scores - self.g3_score_lambda * trend
        elif self.g3_score_mode == "trend_only":
            scores = trend.expand_as(scores)
        return scores
