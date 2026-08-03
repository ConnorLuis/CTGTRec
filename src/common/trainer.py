# coding: utf-8
"""Training and evaluation utilities.

Model selection is performed exclusively on the validation set. Whenever the
validation metric improves, the trainer snapshots that model state. At the end
of training, the best validation snapshot is restored. Test evaluation is not
performed inside ``fit``; callers must evaluate the restored model once.
"""

from __future__ import annotations

import itertools
import os
from logging import getLogger
from time import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.topk_evaluator import TopKEvaluator
from utils.utils import dict2str, early_stopping


class AbstractTrainer(object):
    """Abstract interface for recommendation trainers."""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError("Method [fit] should be implemented.")

    def evaluate(self, eval_data):
        raise NotImplementedError("Method [evaluate] should be implemented.")


class Trainer(AbstractTrainer):
    """Basic trainer with validation-only early stopping and checkpointing."""

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = int(config["epochs"])
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        self.eval_step = min(int(config["eval_step"]), self.epochs)
        if self.eval_step <= 0:
            raise ValueError("eval_step must be positive.")
        self.stopping_step = int(config["stopping_step"])
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = bool(config["valid_metric_bigger"])
        self.test_batch_size = config["eval_batch_size"]
        self.device = config["device"]

        self.weight_decay = 0.0
        if config["weight_decay"] is not None:
            weight_decay = config["weight_decay"]
            self.weight_decay = (
                eval(weight_decay)
                if isinstance(weight_decay, str)
                else weight_decay
            )

        self.req_training = config["req_training"]
        self.start_epoch = 0
        self.cur_step = 0

        empty_result = {}
        for metric, topk in itertools.product(
            config["metrics"],
            config["topk"],
        ):
            empty_result["{}@{}".format(metric.lower(), topk)] = 0.0

        self.best_valid_score = (
            float("-inf")
            if self.valid_metric_bigger
            else float("inf")
        )
        self.best_valid_result = empty_result
        self.best_epoch = None
        self.best_model_state = None
        self.train_loss_dict = {}
        self.optimizer = self._build_optimizer()

        lr_scheduler = config["learning_rate_scheduler"]
        factor = lambda epoch: lr_scheduler[0] ** (
            epoch / lr_scheduler[1]
        )
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=factor,
        )

        self.eval_type = config["eval_type"]
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config["alpha1"]
        self.alpha2 = config["alpha2"]
        self.beta = config["beta"]

    def _build_optimizer(self):
        learner = self.learner.lower()
        kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        if learner == "adam":
            return optim.Adam(self.model.parameters(), **kwargs)
        if learner == "sgd":
            return optim.SGD(self.model.parameters(), **kwargs)
        if learner == "adagrad":
            return optim.Adagrad(self.model.parameters(), **kwargs)
        if learner == "rmsprop":
            return optim.RMSprop(self.model.parameters(), **kwargs)

        self.logger.warning(
            "Received unrecognized optimizer %r; using Adam.",
            self.learner,
        )
        return optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        if not self.req_training:
            return 0.0, []

        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []

        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_interaction = interaction.clone()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(
                    per_loss.item() for per_loss in losses
                )
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item()
                    if total_loss is None
                    else total_loss + losses.item()
                )

            if self._check_nan(loss):
                self.logger.info(
                    "Loss is NaN at epoch %s, batch %s.",
                    epoch_idx,
                    batch_idx,
                )
                return loss, torch.tensor(0.0)

            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses = loss_func(second_interaction)
                loss = sum(losses) if isinstance(losses, tuple) else losses
                if self._check_nan(loss):
                    self.logger.info(
                        "Loss is NaN at epoch %s, batch %s.",
                        epoch_idx,
                        batch_idx,
                    )
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(
                    self.model.parameters(),
                    **self.clip_grad_norm,
                )
            self.optimizer.step()
            loss_batches.append(loss.detach())

        if total_loss is None:
            raise RuntimeError("Training data loader produced no batches.")
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data)
        if self.valid_metric not in valid_result:
            raise KeyError(
                "Validation metric {!r} not found in result keys {}.".format(
                    self.valid_metric,
                    sorted(valid_result),
                )
            )
        return valid_result[self.valid_metric], valid_result

    @staticmethod
    def _check_nan(loss):
        return bool(torch.isnan(loss).item())

    @staticmethod
    def _capture_model_state(model):
        """Copy a portable CPU snapshot of every parameter and buffer."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    def _checkpoint_path(self):
        configured = self.config["checkpoint_file"]
        if configured:
            return os.path.abspath(str(configured))

        checkpoint_dir = os.path.abspath(
            str(self.config["checkpoint_dir"] or "saved")
        )
        filename = "{}-{}-seed{}.pth".format(
            self.config["model"],
            self.config["dataset"],
            self.config["seed"],
        )
        return os.path.join(checkpoint_dir, filename)

    def _save_best_checkpoint(self):
        if self.best_model_state is None or self.best_epoch is None:
            raise RuntimeError("No best model state is available to save.")

        checkpoint_path = self._checkpoint_path()
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        temporary_path = checkpoint_path + ".tmp"
        torch.save(
            {
                "model_state_dict": self.best_model_state,
                "best_epoch": int(self.best_epoch),
                "best_valid_score": float(self.best_valid_score),
                "best_valid_result": dict(self.best_valid_result),
                "model": str(self.config["model"]),
                "dataset": str(self.config["dataset"]),
                "seed": int(self.config["seed"]),
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path

    def _generate_train_loss_output(
        self,
        epoch_idx,
        start_time,
        end_time,
        losses,
    ):
        if isinstance(losses, tuple):
            body = ", ".join(
                "train_loss{}: {:.4f}".format(index + 1, loss)
                for index, loss in enumerate(losses)
            )
            return "epoch {} training [time: {:.2f}s, {}]".format(
                epoch_idx,
                end_time - start_time,
                body,
            )

        return "epoch {} training [time: {:.2f}s, train loss: {:.4f}]".format(
            epoch_idx,
            end_time - start_time,
            losses,
        )

    def fit(
        self,
        train_data,
        valid_data=None,
        saved=False,
        verbose=True,
    ):
        """Train, select by validation, restore the best snapshot, and return it.

        Test data is intentionally not accepted here. The caller must invoke
        ``evaluate(test_data, is_test=True)`` exactly once after this method.
        """
        if valid_data is None:
            raise ValueError(
                "valid_data is required for validation-based model selection."
            )

        checkpoint_path = None

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break

            self.lr_scheduler.step()
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss)
                if isinstance(train_loss, tuple)
                else train_loss
            )
            training_end = time()

            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(
                    self._generate_train_loss_output(
                        epoch_idx,
                        training_start,
                        training_end,
                        train_loss,
                    )
                )
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            validation_start = time()
            valid_score, valid_result = self._valid_epoch(valid_data)
            (
                self.best_valid_score,
                self.cur_step,
                stop_flag,
                update_flag,
            ) = early_stopping(
                valid_score,
                self.best_valid_score,
                self.cur_step,
                max_step=self.stopping_step,
                bigger=self.valid_metric_bigger,
            )
            validation_end = time()

            if verbose:
                self.logger.info(
                    "epoch %d validating [time: %.2fs, valid_score: %.8f]",
                    epoch_idx,
                    validation_end - validation_start,
                    valid_score,
                )
                self.logger.info(
                    "valid result:\n%s",
                    dict2str(valid_result),
                )

            if update_flag:
                self.best_valid_result = dict(valid_result)
                self.best_epoch = int(epoch_idx)
                self.best_model_state = self._capture_model_state(self.model)
                if saved:
                    checkpoint_path = self._save_best_checkpoint()
                if verbose:
                    self.logger.info(
                        "Best validation checkpoint updated at epoch %d.",
                        epoch_idx,
                    )

            if stop_flag:
                if verbose:
                    self.logger.info(
                        "Early stopping after epoch %d; best epoch is %s.",
                        epoch_idx,
                        self.best_epoch,
                    )
                break

        if self.best_model_state is None or self.best_epoch is None:
            raise RuntimeError(
                "Training finished without a valid checkpoint. "
                "Check losses, validation data, and evaluation metrics."
            )

        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)

        if saved and checkpoint_path is None:
            checkpoint_path = self._save_best_checkpoint()
        if checkpoint_path:
            self.logger.info(
                "Restored best validation model from epoch %d; checkpoint: %s",
                self.best_epoch,
                checkpoint_path,
            )
        else:
            self.logger.info(
                "Restored best validation model from epoch %d.",
                self.best_epoch,
            )

        return (
            float(self.best_valid_score),
            dict(self.best_valid_result),
            int(self.best_epoch),
        )

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()

        batch_matrix_list = []
        for batched_data in eval_data:
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(
                scores,
                max(self.config["topk"]),
                dim=-1,
            )
            batch_matrix_list.append(topk_index)

        if not batch_matrix_list:
            raise RuntimeError("Evaluation data loader produced no batches.")

        return self.evaluator.evaluate(
            batch_matrix_list,
            eval_data,
            is_test=is_test,
            idx=idx,
        )

    def plot_train_loss(self, show=True, save_path=None):
        epochs = sorted(self.train_loss_dict)
        values = [
            float(self.train_loss_dict[epoch])
            for epoch in epochs
        ]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
