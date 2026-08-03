# coding: utf-8
"""Top-K evaluator.

Standard metric values are returned at full floating-point precision. Log
formatting and paper-table rounding are applied later, after seed aggregation,
so sample standard deviations are not computed from prematurely rounded values.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.metrics import metrics_dict
from utils.utils import get_local_time


topk_metrics = {
    metric.lower(): metric
    for metric in ["Recall", "Recall2", "Precision", "NDCG", "MAP"]
}


class TopKEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.metrics = config["metrics"]
        self.topk = config["topk"]
        self.save_recom_result = config["save_recommended_topk"]
        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(
                scores_tensor,
                user_len_list,
                dim=0,
            )
            scores_matrix = pad_sequence(
                scores_list,
                batch_first=True,
                padding_value=-np.inf,
            )
        _, topk_index = torch.topk(
            scores_matrix,
            max(self.topk),
            dim=-1,
        )
        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(
            batch_matrix_list,
            dim=0,
        ).cpu().numpy()

        if self.save_recom_result and is_test:
            dataset_name = self.config["dataset"]
            model_name = self.config["model"]
            max_k = max(self.topk)
            directory = os.path.abspath(
                self.config["recommend_topk"]
            )
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(
                directory,
                "{}-{}-idx{}-top{}-{}.csv".format(
                    model_name,
                    dataset_name,
                    idx,
                    max_k,
                    get_local_time(),
                ),
            )
            dataframe = pd.DataFrame(topk_index)
            dataframe.insert(
                0,
                "id",
                eval_data.get_eval_users(),
            )
            dataframe.columns = ["id"] + [
                "top_{}".format(index)
                for index in range(max_k)
            ]
            dataframe.astype(int).to_csv(
                file_path,
                sep="\t",
                index=False,
            )

        if len(pos_len_list) != len(topk_index):
            raise ValueError(
                "Evaluation positives and predictions are not aligned."
            )

        bool_rec_matrix = np.asarray(
            [
                [
                    item in positives
                    for item in recommendations
                ]
                for positives, recommendations in zip(
                    pos_items,
                    topk_index,
                )
            ]
        )

        metric_dict = {}
        result_list = self._calculate_metrics(
            pos_len_list,
            bool_rec_matrix,
        )
        for metric, values in zip(self.metrics, result_list):
            for topk in self.topk:
                # Preserve precision for cross-seed aggregation.
                metric_dict[
                    "{}@{}".format(metric, topk)
                ] = float(values[topk - 1])

        if bool(self.config["history_group_eval"]):
            metric_dict.update(
                self._history_group_metrics(
                    bool_rec_matrix,
                    pos_len_list,
                    eval_data,
                )
            )
        return metric_dict

    def _history_group_metrics(
        self,
        bool_rec_matrix,
        pos_len_list,
        eval_data,
    ):
        topk = int(self.config["history_group_k"] or 20)
        if topk not in self.topk:
            raise ValueError(
                "history_group_k={} must be included in topk={}".format(
                    topk,
                    self.topk,
                )
            )

        history_len = np.asarray(
            eval_data.train_pos_len_list,
            dtype=np.int64,
        )
        eval_users = np.asarray(
            eval_data.get_eval_users(),
            dtype=np.int64,
        )
        pos_len = np.maximum(
            np.asarray(pos_len_list, dtype=np.float64),
            1.0,
        )
        if len(history_len) != len(bool_rec_matrix):
            raise ValueError(
                "Training history lengths are not aligned with users."
            )

        per_user_recall = (
            bool_rec_matrix[:, :topk].sum(axis=1) / pos_len
        )
        stable_order = np.lexsort((eval_users, history_len))
        group_names = ("short", "medium", "long")
        output = {}

        for name, group_index in zip(
            group_names,
            np.array_split(stable_order, 3),
        ):
            if len(group_index) == 0:
                raise ValueError(
                    "Not enough users to form three history groups."
                )
            group_lengths = history_len[group_index]
            output[
                "history_{}_recall@{}".format(name, topk)
            ] = float(per_user_recall[group_index].mean())
            output["history_{}_users".format(name)] = int(
                len(group_index)
            )
            output["history_{}_mean_len".format(name)] = float(
                group_lengths.mean()
            )
            output["history_{}_min_len".format(name)] = int(
                group_lengths.min()
            )
            output["history_{}_max_len".format(name)] = int(
                group_lengths.max()
            )
        return output

    def _check_args(self):
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        elif not isinstance(self.metrics, list):
            raise TypeError("metrics must be str or list")

        for metric in self.metrics:
            if metric.lower() not in topk_metrics:
                raise ValueError(
                    "There is no top-k metric named {}.".format(metric)
                )
        self.metrics = [
            metric.lower()
            for metric in self.metrics
        ]

        if isinstance(self.topk, int):
            self.topk = [self.topk]
        elif not isinstance(self.topk, list):
            raise TypeError("topk must be an integer or list")

        for topk in self.topk:
            if topk <= 0:
                raise ValueError(
                    "topk must contain positive integers."
                )

    def _calculate_metrics(self, pos_len_list, topk_index):
        result_list = []
        for metric in self.metrics:
            metric_function = metrics_dict[metric.lower()]
            result_list.append(
                metric_function(topk_index, pos_len_list)
            )
        return np.stack(result_list, axis=0)

    def __str__(self):
        return "The TopK Evaluator Info:\n\tMetrics:[{}], TopK:[{}]".format(
            ", ".join(
                topk_metrics[metric.lower()]
                for metric in self.metrics
            ),
            ", ".join(map(str, self.topk)),
        )
