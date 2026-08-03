# coding: utf-8
"""MMRec quick start with fixed-seed repetition and aggregate reporting.

For every hyperparameter combination, all configured random seeds are run
independently. Each seed uses validation-only early stopping, restores its best
validation checkpoint, and evaluates the test split exactly once. Results are
reported as arithmetic mean and sample standard deviation (ddof=1).

A seed is never selected as the "best seed." When a true hyperparameter search
is present, combinations are compared by the mean validation metric across
seeds; test metrics never participate in model selection.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
from itertools import product
from logging import getLogger
from pathlib import Path

import numpy as np
import torch

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import dict2str, get_model, get_trainer, init_seed


def _as_search_space(value):
    if isinstance(value, (list, tuple)):
        return list(value) if value else [None]
    return [value]


def _unique(values):
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _normalize_seeds(value):
    seeds = _as_search_space(value)
    if not seeds:
        raise ValueError("At least one random seed is required.")

    normalized = []
    for seed in seeds:
        if isinstance(seed, bool):
            raise TypeError("Boolean values are not valid random seeds.")
        integer_seed = int(seed)
        if float(seed) != integer_seed:
            raise ValueError("Random seeds must be integers: {}".format(seed))
        if integer_seed not in normalized:
            normalized.append(integer_seed)
    return normalized


def _metric_keys(config):
    return [
        "{}@{}".format(metric.lower(), topk)
        for metric in config["metrics"]
        for topk in config["topk"]
    ]


def _aggregate_results(results, metric_keys):
    if not results:
        raise ValueError("Cannot aggregate an empty result list.")

    aggregate = {}
    for metric in metric_keys:
        try:
            values = np.asarray(
                [float(result[metric]) for result in results],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise KeyError(
                "Metric {!r} is missing from one or more seed results.".format(
                    metric
                )
            ) from exc

        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_std": (
                float(values.std(ddof=1))
                if len(values) >= 2
                else None
            ),
            "num_seeds": int(len(values)),
        }
    return aggregate


def _format_aggregate(aggregate):
    fields = []
    for metric, statistics in aggregate.items():
        std = statistics["sample_std"]
        std_text = "n/a" if std is None else "{:.6f}".format(std)
        fields.append(
            "{}: mean={:.6f}, sample_std={}".format(
                metric,
                statistics["mean"],
                std_text,
            )
        )
    return "    ".join(fields)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    return str(value)


def _atomic_write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(rows, fieldnames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _result_directory(config, combination_index):
    root = Path(str(config["result_dir"] or "results"))
    return (
        root
        / str(config["model"]).lower()
        / str(config["dataset"]).lower()
        / "combo_{:03d}".format(combination_index)
    )


def _checkpoint_file(config, combination_index, seed):
    root = Path(str(config["checkpoint_dir"] or "saved"))
    filename = "{}-{}-combo{:03d}-seed{}.pth".format(
        str(config["model"]).lower(),
        str(config["dataset"]).lower(),
        combination_index,
        seed,
    )
    return root / filename


def _write_combination_results(
    *,
    config,
    combination_index,
    search_parameters,
    seeds,
    seed_runs,
    validation_summary,
    test_summary,
):
    output_dir = _result_directory(config, combination_index)
    metric_keys = _metric_keys(config)

    seed_rows = []
    for run in seed_runs:
        row = {
            "seed": run["seed"],
            "best_epoch": run["best_epoch"],
            "checkpoint_file": run["checkpoint_file"],
        }
        for metric in metric_keys:
            row["valid_{}".format(metric)] = run["validation"][metric]
            row["test_{}".format(metric)] = run["test"][metric]
        seed_rows.append(row)

    seed_fields = [
        "seed",
        "best_epoch",
        "checkpoint_file",
    ] + [
        "{}_{}".format(split, metric)
        for split in ("valid", "test")
        for metric in metric_keys
    ]
    _atomic_write_csv(
        seed_rows,
        seed_fields,
        output_dir / "seed_results.csv",
    )

    summary_rows = []
    for split_name, summary in (
        ("validation", validation_summary),
        ("test", test_summary),
    ):
        for metric, statistics in summary.items():
            summary_rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "mean": statistics["mean"],
                    "sample_std": statistics["sample_std"],
                    "num_seeds": statistics["num_seeds"],
                }
            )
    _atomic_write_csv(
        summary_rows,
        [
            "split",
            "metric",
            "mean",
            "sample_std",
            "num_seeds",
        ],
        output_dir / "summary.csv",
    )

    payload = {
        "model": str(config["model"]),
        "dataset": str(config["dataset"]),
        "combination_index": int(combination_index),
        "search_parameters": _json_safe(search_parameters),
        "seeds": list(seeds),
        "selection_rule": (
            "select hyperparameter combinations by mean validation metric "
            "across seeds; never select a seed and never select by test"
        ),
        "test_protocol": (
            "restore each seed's best validation checkpoint and evaluate "
            "the test split exactly once"
        ),
        "validation": validation_summary,
        "test": test_summary,
        "seed_runs": seed_runs,
    }
    _atomic_write_json(payload, output_dir / "summary.json")
    return output_dir


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    logger.info("██Server:\t%s", platform.node())
    logger.info("██Dir:\t%s\n", os.getcwd())
    logger.info(config)

    dataset_object = RecDataset(config)
    logger.info(str(dataset_object))

    train_dataset, valid_dataset, test_dataset = dataset_object.split()
    logger.info("\n====Training====\n%s", train_dataset)
    logger.info("\n====Validation====\n%s", valid_dataset)
    logger.info("\n====Testing====\n%s", test_dataset)

    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
    )
    valid_data = EvalDataLoader(
        config,
        valid_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )
    test_data = EvalDataLoader(
        config,
        test_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )

    names = _unique(config["hyper_parameters"] or [])
    seeds = _normalize_seeds(config["seed"])
    search_names = [name for name in names if name != "seed"]
    search_spaces = [
        _as_search_space(config[name])
        for name in search_names
    ]
    combinations = list(product(*search_spaces)) if search_spaces else [()]
    metric_keys = _metric_keys(config)

    validation_metric = config["valid_metric"].lower()
    validation_bigger = bool(config["valid_metric_bigger"])
    best_combination_index = None
    best_validation_mean = (
        float("-inf") if validation_bigger else float("inf")
    )
    combination_summaries = []

    logger.info(
        "Fixed seeds: %s. Search parameters: %s. Combinations: %d.",
        seeds,
        search_names,
        len(combinations),
    )

    global_run_index = 0
    for combination_index, combination in enumerate(combinations):
        search_parameters = dict(zip(search_names, combination))
        for name, value in search_parameters.items():
            config[name] = value

        logger.info(
            "\n===== Combination %d/%d: %s =====",
            combination_index + 1,
            len(combinations),
            search_parameters or "fixed final configuration",
        )

        seed_runs = []
        validation_results = []
        test_results = []

        for seed in seeds:
            config["seed"] = seed
            checkpoint_path = _checkpoint_file(
                config,
                combination_index,
                seed,
            )
            config["checkpoint_file"] = str(checkpoint_path)

            init_seed(seed)
            logger.info(
                "\n--- Seed %d: independent training run ---",
                seed,
            )

            train_data.pretrain_setup()
            model_instance = get_model(config["model"])(
                config,
                train_data,
            ).to(config["device"])
            logger.info(model_instance)

            trainer = get_trainer()(config, model_instance, mg)
            (
                _,
                best_valid_result,
                best_epoch,
            ) = trainer.fit(
                train_data,
                valid_data=valid_data,
                saved=save_model,
            )

            # This is the only test evaluation for this seed.
            test_result = trainer.evaluate(
                test_data,
                is_test=True,
                idx=global_run_index,
            )
            global_run_index += 1

            validation_results.append(dict(best_valid_result))
            test_results.append(dict(test_result))
            seed_runs.append(
                {
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "checkpoint_file": (
                        str(checkpoint_path)
                        if save_model
                        else None
                    ),
                    "validation": {
                        metric: float(best_valid_result[metric])
                        for metric in metric_keys
                    },
                    "test": {
                        metric: float(test_result[metric])
                        for metric in metric_keys
                    },
                }
            )

            logger.info(
                "Seed %d best validation result: %s",
                seed,
                dict2str(best_valid_result),
            )
            logger.info(
                "Seed %d single final test result: %s",
                seed,
                dict2str(test_result),
            )

            del trainer
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        validation_summary = _aggregate_results(
            validation_results,
            metric_keys,
        )
        test_summary = _aggregate_results(
            test_results,
            metric_keys,
        )
        output_dir = _write_combination_results(
            config=config,
            combination_index=combination_index,
            search_parameters=search_parameters,
            seeds=seeds,
            seed_runs=seed_runs,
            validation_summary=validation_summary,
            test_summary=test_summary,
        )

        validation_mean = validation_summary[validation_metric]["mean"]
        is_better = (
            validation_mean > best_validation_mean
            if validation_bigger
            else validation_mean < best_validation_mean
        )
        if is_better:
            best_validation_mean = validation_mean
            best_combination_index = combination_index

        combination_summaries.append(
            {
                "combination_index": combination_index,
                "search_parameters": search_parameters,
                "validation": validation_summary,
                "test": test_summary,
                "result_directory": str(output_dir),
            }
        )

        logger.info(
            "Combination %d validation mean ± sample std: %s",
            combination_index,
            _format_aggregate(validation_summary),
        )
        logger.info(
            "Combination %d test mean ± sample std: %s",
            combination_index,
            _format_aggregate(test_summary),
        )
        logger.info("Result files: %s", output_dir)

    if best_combination_index is None:
        raise RuntimeError("No completed configuration combination.")

    selected = combination_summaries[best_combination_index]
    if search_names:
        logger.info(
            "\n===== Selected combination by mean validation %s =====",
            validation_metric,
        )
        logger.info(
            "Parameters: %s",
            selected["search_parameters"],
        )
    else:
        logger.info(
            "\n===== Fixed final configuration: three-seed summary ====="
        )

    logger.info(
        "Validation: %s",
        _format_aggregate(selected["validation"]),
    )
    logger.info(
        "Test: %s",
        _format_aggregate(selected["test"]),
    )
    logger.info(
        "No best seed was selected. Test results were not used for selection."
    )

    return {
        "selected_combination_index": best_combination_index,
        "selected": selected,
        "all_combinations": combination_summaries,
    }
