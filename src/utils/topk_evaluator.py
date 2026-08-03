# coding: utf-8
"""MMRec Top-K evaluator with optional user-history tertile metrics.

This is a drop-in replacement. Ordinary experiments are unchanged unless
``history_group_eval: true`` is present in the active YAML.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.metrics import metrics_dict
from utils.utils import get_local_time


topk_metrics = {metric.lower(): metric for metric in ["Recall", "Recall2", "Precision", "NDCG", "MAP"]}


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
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)
        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        if self.save_recom_result and is_test:
            dataset_name = self.config["dataset"]
            model_name = self.config["model"]
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config["recommend_topk"])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(
                dir_name,
                "{}-{}-idx{}-top{}-{}.csv".format(
                    model_name, dataset_name, idx, max_k, get_local_time()
                ),
            )
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, "id", eval_data.get_eval_users())
            x_df.columns = ["id"] + ["top_" + str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep="\t", index=False)

        assert len(pos_len_list) == len(topk_index)
        bool_rec_matrix = np.asarray(
            [[True if item in positives else False for item in recs]
             for positives, recs in zip(pos_items, topk_index)]
        )

        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                metric_dict["{}@{}".format(metric, k)] = round(value[k - 1], 4)

        if bool(self.config["history_group_eval"]):
            metric_dict.update(self._history_group_metrics(bool_rec_matrix, pos_len_list, eval_data))
        return metric_dict

    def _history_group_metrics(self, bool_rec_matrix, pos_len_list, eval_data):
        """Return per-user Recall@K averaged inside rank-balanced history tertiles.

        Stable sorting by (train-history length, user id) ensures deterministic,
        near-equal group sizes. Equal history lengths may straddle adjacent tertiles;
        that is preferable to empty or severely imbalanced quantile groups.
        """
        k = int(self.config["history_group_k"] or 20)
        if k not in self.topk:
            raise ValueError("history_group_k={} must be included in topk={}".format(k, self.topk))

        history_len = np.asarray(eval_data.train_pos_len_list, dtype=np.int64)
        eval_users = np.asarray(eval_data.get_eval_users(), dtype=np.int64)
        pos_len = np.maximum(np.asarray(pos_len_list, dtype=np.float64), 1.0)
        if len(history_len) != len(bool_rec_matrix):
            raise ValueError("train history lengths are not aligned with evaluated users")

        per_user_recall = bool_rec_matrix[:, :k].sum(axis=1) / pos_len
        stable_order = np.lexsort((eval_users, history_len))
        group_names = ("short", "medium", "long")
        out = {}
        for name, group_index in zip(group_names, np.array_split(stable_order, 3)):
            if len(group_index) == 0:
                raise ValueError("Not enough evaluated users to form three history groups")
            group_lengths = history_len[group_index]
            out["history_{}_recall@{}".format(name, k)] = round(
                float(per_user_recall[group_index].mean()), 4
            )
            out["history_{}_users".format(name)] = int(len(group_index))
            out["history_{}_mean_len".format(name)] = round(float(group_lengths.mean()), 2)
            out["history_{}_min_len".format(name)] = int(group_lengths.min())
            out["history_{}_max_len".format(name)] = int(group_lengths.max())
        return out

    def _check_args(self):
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError("metrics must be str or list")
        for metric in self.metrics:
            if metric.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(metric))
        self.metrics = [metric.lower() for metric in self.metrics]

        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError("topk must be a positive integer or a list of positive integers")
        else:
            raise TypeError("The topk must be a integer, list")

    def _calculate_metrics(self, pos_len_list, topk_index):
        result_list = []
        for metric in self.metrics:
            metric_func = metrics_dict[metric.lower()]
            result_list.append(metric_func(topk_index, pos_len_list))
        return np.stack(result_list, axis=0)

    def __str__(self):
        return "The TopK Evaluator Info:\n\tMetrics:[{}], TopK:[{}]".format(
            ", ".join([topk_metrics[m.lower()] for m in self.metrics]),
            ", ".join(map(str, self.topk)),
        )
