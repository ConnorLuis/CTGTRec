# coding: utf-8
r"""CTGTRec: Continuous-Time Graph and Trend-aware Recommendation.

The released model contains the three components described in the paper:

1. a train-only continuous-time weighted user-item graph;
2. a frozen multimodal item-item graph;
3. train-only item-trend calibration added during full-ranking inference.

For Sports and Clothing, user-item edge dropout samples aggregated user-item
edges from the raw temporal graph, retains their continuous-time weights, and
symmetrically renormalizes the sampled weighted graph. Validation and test
prediction always use the complete normalized temporal graph.
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
    """Continuous-time graph recommendation with item-trend calibration."""

    def __init__(self, config, dataset):
        super(CTGTRec, self).__init__(config, dataset)

        self.config = config
        self.dataset_name = str(config["dataset"])
        self.dataset_path = os.path.abspath(
            os.path.join(config["data_path"], self.dataset_name)
        )

        # Shared representation and graph settings.
        self.embedding_dim = int(config["embedding_size"])
        self.feature_embedding_dim = int(config["feat_embed_dim"])
        self.knn_k = int(config["knn_k"])
        self.n_mm_layers = int(config["n_mm_layers"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.aux_loss_weight = float(config["aux_loss_weight"])
        self.visual_fusion_weight = float(config["visual_fusion_weight"])
        self.ui_edge_dropout = float(config["ui_edge_dropout"])

        if self.embedding_dim <= 0 or self.feature_embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        if self.n_mm_layers < 0 or self.n_ui_layers < 0:
            raise ValueError("Graph layer counts must be non-negative.")
        if self.knn_k <= 0 or self.knn_k > self.n_items:
            raise ValueError(
                "knn_k must be in [1, n_items], received {} for {} items.".format(
                    self.knn_k,
                    self.n_items,
                )
            )
        if self.aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative.")
        if not 0.0 <= self.visual_fusion_weight <= 1.0:
            raise ValueError("visual_fusion_weight must be in [0, 1].")
        if not 0.0 <= self.ui_edge_dropout < 1.0:
            raise ValueError("ui_edge_dropout must be in [0, 1).")

        # Continuous-time graph files.
        self.temporal_graph_dir = str(config["temporal_graph_dir"])
        self.ct_raw_graph_file = str(config["ct_raw_graph_file"])
        self.ct_normalized_graph_file = str(
            config["ct_normalized_graph_file"]
        )

        # Fixed item-trend calibration protocol.
        self.trend_weight = float(config["trend_weight"])
        self.trend_recent_ratio = float(config["trend_recent_ratio"])
        self.trend_epsilon = float(config["trend_epsilon"])
        self.trend_clip = float(config["trend_clip"])

        if self.trend_weight < 0.0:
            raise ValueError("trend_weight must be non-negative.")
        if not 0.0 < self.trend_recent_ratio <= 1.0:
            raise ValueError("trend_recent_ratio must be in (0, 1].")
        if self.trend_epsilon <= 0.0 or not np.isfinite(self.trend_epsilon):
            raise ValueError("trend_epsilon must be finite and positive.")
        if self.trend_clip <= 0.0 or not np.isfinite(self.trend_clip):
            raise ValueError("trend_clip must be finite and positive.")

        self.n_nodes = self.n_users + self.n_items
        (
            self.normalized_ui_graph,
            self.ui_edge_indices,
            self.ui_edge_raw_weights,
            self.ui_edge_sampling_scores,
        ) = self._load_continuous_time_graphs()
        self.normalized_ui_graph = self.normalized_ui_graph.to(self.device)
        self.ui_edge_indices = self.ui_edge_indices.to(self.device)
        self.ui_edge_raw_weights = self.ui_edge_raw_weights.to(self.device)
        self.ui_edge_sampling_scores = self.ui_edge_sampling_scores.to(
            self.device
        )
        self.training_ui_graph = self.normalized_ui_graph

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(
            self.n_items,
            self.embedding_dim,
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_embedding = None
        self.text_embedding = None
        self.image_projection = None
        self.text_projection = None

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat,
                freeze=False,
            )
            self.image_projection = nn.Linear(
                self.v_feat.shape[1],
                self.feature_embedding_dim,
            )
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat,
                freeze=False,
            )
            self.text_projection = nn.Linear(
                self.t_feat.shape[1],
                self.feature_embedding_dim,
            )

        self.multimodal_item_graph = self._load_or_build_multimodal_graph()
        self.multimodal_item_graph = self.multimodal_item_graph.to(self.device)

        item_trend, trend_metadata = self._build_item_trend()
        self.register_buffer(
            "item_trend",
            torch.from_numpy(item_trend),
        )
        self.trend_metadata = trend_metadata

        print("[CTGTRec] dataset_path = {}".format(self.dataset_path))
        print(
            "[CTGTRec] temporal_graph = {}, dropout = {:.3f}, edges = {}".format(
                self.ct_normalized_graph_file,
                self.ui_edge_dropout,
                self.ui_edge_raw_weights.numel(),
            )
        )
        print(
            "[CTGTRec] trend_weight = {:.3f}, recent_ratio = {:.3f}, "
            "threshold = {:.6f}, recent_edges = {}".format(
                self.trend_weight,
                self.trend_recent_ratio,
                self.trend_metadata["threshold"],
                self.trend_metadata["recent_interactions"],
            )
        )

    # ------------------------------------------------------------------
    # Train-only item trend
    # ------------------------------------------------------------------
    @staticmethod
    def _find_column(header, target):
        for index, name in enumerate(header):
            if name == target or name.startswith(target + ":"):
                return index
        return -1

    def _temporal_interaction_path(self):
        inter_name = self.config["inter_file_name"]
        if inter_name in {None, "", "none", "None", "null", "NULL"}:
            raise ValueError(
                "inter_file_name must identify the strict temporal interaction "
                "file used by the active dataset configuration."
            )
        inter_name = str(inter_name)
        if os.path.isabs(inter_name):
            return inter_name
        return os.path.join(self.dataset_path, inter_name)

    def _read_train_item_times(self):
        inter_path = self._temporal_interaction_path()
        if not os.path.isfile(inter_path):
            raise FileNotFoundError(
                "Cannot find strict temporal interaction file: {}".format(
                    inter_path
                )
            )

        pairs = []
        observed_labels = set()
        with open(inter_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(
                    "Temporal interaction file is empty: {}".format(inter_path)
                ) from exc

            item_col = self._find_column(header, "itemID")
            time_col = self._find_column(header, "timestamp")
            label_col = self._find_column(header, "x_label")
            if item_col < 0 or time_col < 0 or label_col < 0:
                raise ValueError(
                    "Expected itemID, timestamp, and x_label columns in {}.".format(
                        inter_path
                    )
                )

            required_width = max(item_col, time_col, label_col)
            for line_number, row in enumerate(reader, start=2):
                if len(row) <= required_width:
                    raise ValueError(
                        "Malformed row {} in {}.".format(
                            line_number,
                            inter_path,
                        )
                    )
                try:
                    item = int(float(row[item_col]))
                    timestamp = float(row[time_col])
                    label = int(float(row[label_col]))
                except ValueError as exc:
                    raise ValueError(
                        "Invalid itemID, timestamp, or x_label at {}:{}.".format(
                            inter_path,
                            line_number,
                        )
                    ) from exc

                if label not in {0, 1, 2}:
                    raise ValueError(
                        "Unexpected x_label {} at {}:{}; expected 0, 1, or 2.".format(
                            label,
                            inter_path,
                            line_number,
                        )
                    )
                if not 0 <= item < self.n_items:
                    raise ValueError(
                        "itemID {} is outside [0, {}) at {}:{}.".format(
                            item,
                            self.n_items,
                            inter_path,
                            line_number,
                        )
                    )
                if not np.isfinite(timestamp):
                    raise ValueError(
                        "Non-finite timestamp at {}:{}.".format(
                            inter_path,
                            line_number,
                        )
                    )

                observed_labels.add(label)
                if label == 0:
                    pairs.append((item, timestamp))

        if observed_labels != {0, 1, 2}:
            raise ValueError(
                "Temporal file must contain train/validation/test labels 0, 1, "
                "and 2; found {} in {}.".format(
                    sorted(observed_labels),
                    inter_path,
                )
            )
        if not pairs:
            raise ValueError(
                "No training interactions found in {}.".format(inter_path)
            )
        return pairs

    @staticmethod
    def _linear_quantile(values, quantile):
        try:
            return np.quantile(values, quantile, method="linear")
        except TypeError:  # NumPy < 1.22
            return np.quantile(values, quantile, interpolation="linear")

    @classmethod
    def _compute_item_trend(
        cls,
        items,
        timestamps,
        *,
        n_items,
        recent_ratio,
        epsilon,
        clip_value,
    ):
        """Compute log-ratio, z-score, and clipped train-only item trend."""
        items = np.asarray(items, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if items.ndim != 1 or timestamps.ndim != 1:
            raise ValueError("items and timestamps must be one-dimensional.")
        if len(items) == 0 or len(items) != len(timestamps):
            raise ValueError("items and timestamps must have equal non-zero length.")
        if not np.isfinite(timestamps).all():
            raise ValueError("timestamps contain non-finite values.")
        if (items < 0).any() or (items >= n_items).any():
            raise ValueError("items contain IDs outside the configured range.")

        threshold = cls._linear_quantile(
            timestamps,
            1.0 - recent_ratio,
        )
        # Include every interaction tied at the quantile threshold.
        recent_mask = timestamps >= threshold
        recent_count = int(recent_mask.sum())
        if recent_count <= 0:
            raise RuntimeError("The recent interaction subset is empty.")

        all_frequency = np.bincount(
            items,
            minlength=n_items,
        ).astype(np.float64)
        recent_frequency = np.bincount(
            items[recent_mask],
            minlength=n_items,
        ).astype(np.float64)

        all_rate = all_frequency / float(len(items))
        recent_rate = recent_frequency / float(recent_count)
        relative_trend = recent_rate / (all_rate + epsilon)
        log_trend = np.log1p(relative_trend)

        mean = float(log_trend.mean())
        std = float(log_trend.std(ddof=0))
        if std < epsilon:
            normalized = np.zeros_like(log_trend)
        else:
            normalized = (log_trend - mean) / std
        normalized = np.clip(normalized, -clip_value, clip_value)

        metadata = {
            "threshold": float(threshold),
            "train_interactions": int(len(items)),
            "recent_interactions": recent_count,
            "log_trend_mean": mean,
            "log_trend_std": std,
        }
        return normalized.astype(np.float32), metadata

    def _build_item_trend(self):
        pairs = self._read_train_item_times()
        items = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        timestamps = np.asarray(
            [pair[1] for pair in pairs],
            dtype=np.float64,
        )
        return self._compute_item_trend(
            items,
            timestamps,
            n_items=self.n_items,
            recent_ratio=self.trend_recent_ratio,
            epsilon=self.trend_epsilon,
            clip_value=self.trend_clip,
        )

    # ------------------------------------------------------------------
    # Continuous-time weighted user-item graph
    # ------------------------------------------------------------------
    def _temporal_graph_path(self, filename):
        if os.path.isabs(str(filename)):
            return str(filename)
        return os.path.join(
            self.dataset_path,
            self.temporal_graph_dir,
            str(filename),
        )

    @staticmethod
    def _validate_scipy_graph(matrix, path, expected_shape):
        matrix = matrix.astype(np.float32).tocsr()
        if matrix.shape != expected_shape:
            raise ValueError(
                "{} has shape {}, expected {}.".format(
                    path,
                    matrix.shape,
                    expected_shape,
                )
            )
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError("{} contains non-finite values.".format(path))
        if matrix.data.size and (matrix.data < 0).any():
            raise ValueError("{} contains negative values.".format(path))
        difference = matrix - matrix.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-6:
            raise ValueError("{} is not symmetric.".format(path))
        return matrix

    @staticmethod
    def _symmetric_normalize_scipy(raw_adjacency):
        degree = np.asarray(raw_adjacency.sum(axis=1)).reshape(-1)
        inverse_sqrt = np.zeros_like(degree, dtype=np.float64)
        nonzero = degree > 0
        inverse_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
        diagonal = sp.diags(
            inverse_sqrt.astype(np.float32),
            format="csr",
        )
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
        values = torch.from_numpy(
            matrix.data.astype(np.float32, copy=False)
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.shape,
            dtype=torch.float32,
        ).coalesce()

    def _load_continuous_time_graphs(self) -> Tuple[torch.Tensor, ...]:
        raw_path = self._temporal_graph_path(self.ct_raw_graph_file)
        normalized_path = self._temporal_graph_path(
            self.ct_normalized_graph_file
        )
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(
                "Cannot find raw continuous-time graph: {}.".format(raw_path)
            )
        if not os.path.isfile(normalized_path):
            raise FileNotFoundError(
                "Cannot find normalized continuous-time graph: {}.".format(
                    normalized_path
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
                "Normalized graph does not match raw temporal graph: {} vs {}.".format(
                    normalized_path,
                    raw_path,
                )
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
            raise ValueError(
                "Raw temporal graph contains non-bipartite edges."
            )

        forward_mask = (
            (raw_coo.row < self.n_users)
            & (raw_coo.col >= self.n_users)
        )
        users = raw_coo.row[forward_mask].astype(np.int64)
        items = (
            raw_coo.col[forward_mask] - self.n_users
        ).astype(np.int64)
        raw_weights = raw_coo.data[forward_mask].astype(np.float32)
        if len(users) == 0:
            raise ValueError(
                "Raw temporal graph contains no user-item edges."
            )
        if raw.nnz != 2 * len(users):
            raise ValueError(
                "Raw temporal graph must contain exactly two directions for "
                "each aggregated user-item edge."
            )

        normalized_csr = normalized.tocsr()
        sampling_scores = np.asarray(
            normalized_csr[users, self.n_users + items]
        ).reshape(-1).astype(np.float32)
        if (
            not np.isfinite(sampling_scores).all()
            or (sampling_scores <= 0).any()
        ):
            raise ValueError(
                "Invalid edge-sampling scores in normalized temporal graph."
            )

        return (
            self._scipy_to_torch_sparse(normalized),
            torch.from_numpy(np.vstack([users, items])),
            torch.from_numpy(raw_weights),
            torch.from_numpy(sampling_scores),
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
        values = torch.cat(
            [normalized_weights, normalized_weights],
            dim=0,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.n_nodes, self.n_nodes),
            device=raw_weights.device,
            dtype=raw_weights.dtype,
        ).coalesce()

    def pre_epoch_processing(self):
        if self.ui_edge_dropout <= 0.0:
            self.training_ui_graph = self.normalized_ui_graph
            return

        edge_count = self.ui_edge_raw_weights.numel()
        keep_count = max(
            1,
            int(edge_count * (1.0 - self.ui_edge_dropout)),
        )
        selected = torch.multinomial(
            self.ui_edge_sampling_scores,
            keep_count,
            replacement=False,
        )
        self.training_ui_graph = self._normalize_sampled_weighted_edges(
            self.ui_edge_indices[:, selected],
            self.ui_edge_raw_weights[selected],
        )

    # ------------------------------------------------------------------
    # Frozen multimodal item-item graph
    # ------------------------------------------------------------------
    @staticmethod
    def _float_token(value):
        return ("{:.6f}".format(float(value))).rstrip("0").rstrip(".").replace(
            ".",
            "p",
        )

    def _multimodal_cache_path(self):
        filename = "mm_adj_ctgtrec_k{}_v{}.pt".format(
            self.knn_k,
            self._float_token(self.visual_fusion_weight),
        )
        return os.path.join(self.dataset_path, filename)

    @staticmethod
    def _load_torch_file(path):
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # PyTorch without weights_only
            return torch.load(path, map_location="cpu")

    def _validate_multimodal_graph(self, graph, path):
        if not isinstance(graph, torch.Tensor) or not graph.is_sparse:
            raise ValueError(
                "Multimodal graph cache is not a sparse tensor: {}.".format(
                    path
                )
            )
        graph = graph.coalesce().to(dtype=torch.float32)
        if tuple(graph.shape) != (self.n_items, self.n_items):
            raise ValueError(
                "Multimodal graph {} has shape {}, expected ({}, {}).".format(
                    path,
                    tuple(graph.shape),
                    self.n_items,
                    self.n_items,
                )
            )
        values = graph.values()
        if not torch.isfinite(values).all():
            raise ValueError(
                "Multimodal graph contains non-finite values: {}.".format(path)
            )
        if torch.any(values < 0):
            raise ValueError(
                "Multimodal graph contains negative values: {}.".format(path)
            )
        return graph

    def _build_knn_item_graph(self, features):
        if features.ndim != 2 or features.shape[0] != self.n_items:
            raise ValueError(
                "Modality feature matrix must have shape [n_items, dim]."
            )
        normalized_features = F.normalize(
            features,
            p=2,
            dim=1,
            eps=1e-12,
        )
        similarity = torch.mm(
            normalized_features,
            normalized_features.transpose(0, 1),
        )
        _, neighbors = torch.topk(
            similarity,
            self.knn_k,
            dim=1,
        )
        del similarity

        rows = torch.arange(
            self.n_items,
            device=features.device,
        ).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack(
            [rows.reshape(-1), neighbors.reshape(-1)],
            dim=0,
        )
        values = torch.ones(
            indices.shape[1],
            dtype=torch.float32,
            device=features.device,
        )
        graph = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.n_items, self.n_items),
            device=features.device,
        ).coalesce()

        degree = torch.sparse.sum(graph, dim=1).to_dense()
        inverse_sqrt = torch.rsqrt(degree.clamp_min(1e-7))
        normalized_values = (
            inverse_sqrt[graph.indices()[0]]
            * graph.values()
            * inverse_sqrt[graph.indices()[1]]
        )
        return torch.sparse_coo_tensor(
            graph.indices(),
            normalized_values,
            size=graph.shape,
            device=features.device,
        ).coalesce()

    def _load_or_build_multimodal_graph(self):
        cache_path = self._multimodal_cache_path()
        if os.path.isfile(cache_path):
            return self._validate_multimodal_graph(
                self._load_torch_file(cache_path),
                cache_path,
            )

        image_graph = None
        text_graph = None
        if self.v_feat is not None:
            image_graph = self._build_knn_item_graph(self.v_feat.detach())
        if self.t_feat is not None:
            text_graph = self._build_knn_item_graph(self.t_feat.detach())

        if image_graph is not None and text_graph is not None:
            graph = (
                self.visual_fusion_weight * image_graph
                + (1.0 - self.visual_fusion_weight) * text_graph
            ).coalesce()
        elif image_graph is not None:
            graph = image_graph.coalesce()
        elif text_graph is not None:
            graph = text_graph.coalesce()
        else:
            raise ValueError(
                "At least one visual or textual feature matrix is required."
            )

        graph = self._validate_multimodal_graph(graph, cache_path)
        temporary_path = cache_path + ".tmp"
        torch.save(graph.cpu(), temporary_path)
        os.replace(temporary_path, cache_path)
        return graph

    def _multimodal_item_embedding(self):
        embedding = self.item_id_embedding.weight
        for _ in range(self.n_mm_layers):
            embedding = torch.sparse.mm(
                self.multimodal_item_graph,
                embedding,
            )
        return embedding

    # ------------------------------------------------------------------
    # Recommendation objective and inference
    # ------------------------------------------------------------------
    def forward(self, ui_graph):
        embedding = torch.cat(
            [self.user_embedding.weight, self.item_id_embedding.weight],
            dim=0,
        )
        layer_embeddings = [embedding]
        for _ in range(self.n_ui_layers):
            embedding = torch.sparse.mm(ui_graph, embedding)
            layer_embeddings.append(embedding)
        embedding = torch.stack(layer_embeddings, dim=1).mean(dim=1)

        user_embedding, item_embedding = torch.split(
            embedding,
            [self.n_users, self.n_items],
            dim=0,
        )
        item_embedding = item_embedding + self._multimodal_item_embedding()
        return user_embedding, item_embedding

    @staticmethod
    def _bpr_loss(user, positive_item, negative_item):
        positive_score = torch.sum(user * positive_item, dim=1)
        negative_score = torch.sum(user * negative_item, dim=1)
        return -torch.mean(
            F.logsigmoid(positive_score - negative_score)
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        positive_items = interaction[1]
        negative_items = interaction[2]

        user_embedding, item_embedding = self.forward(
            self.training_ui_graph
        )
        main_loss = self._bpr_loss(
            user_embedding[users],
            item_embedding[positive_items],
            item_embedding[negative_items],
        )

        text_loss = 0.0
        image_loss = 0.0
        if self.text_embedding is not None:
            text_features = self.text_projection(
                self.text_embedding.weight
            )
            text_loss = self._bpr_loss(
                user_embedding[users],
                text_features[positive_items],
                text_features[negative_items],
            )
        if self.image_embedding is not None:
            image_features = self.image_projection(
                self.image_embedding.weight
            )
            image_loss = self._bpr_loss(
                user_embedding[users],
                image_features[positive_items],
                image_features[negative_items],
            )

        return main_loss + self.aux_loss_weight * (
            text_loss + image_loss
        )

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embedding, item_embedding = self.forward(
            self.normalized_ui_graph
        )
        scores = torch.matmul(
            user_embedding[users],
            item_embedding.transpose(0, 1),
        )
        # Trend calibration is inference-only and does not alter the BPR loss.
        return scores + self.trend_weight * self.item_trend.view(1, -1)
