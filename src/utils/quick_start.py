# coding: utf-8
"""MMRec quick_start with validation-based combination selection.

Changes from the supplied file:
1. de-duplicate hyper-parameter names merged from several YAML files;
2. wrap scalar search values as one-element lists;
3. choose the final BEST combination by validation metric, never test metric.

Training, early stopping, evaluation and per-run outputs are unchanged.
"""

from itertools import product
from logging import getLogger
import os
import platform

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import dict2str, get_model, get_trainer, init_seed


def _as_search_space(value):
    if isinstance(value, (list, tuple)):
        return list(value) if len(value) > 0 else [None]
    return [value]


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    logger.info("██Server: \t" + platform.node())
    logger.info("██Dir: \t" + os.getcwd() + "\n")
    logger.info(config)

    dataset = RecDataset(config)
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info("\n====Training====\n" + str(train_dataset))
    logger.info("\n====Validation====\n" + str(valid_dataset))
    logger.info("\n====Testing====\n" + str(test_dataset))

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

    hyper_ret = []
    val_metric = config["valid_metric"].lower()
    valid_metric_bigger = bool(config["valid_metric_bigger"])
    best_valid_value = float("-inf") if valid_metric_bigger else float("inf")
    best_valid_idx = 0

    logger.info("\n\n=================================\n\n")

    # Configurator concatenates hyper-parameter lists from overall, dataset and
    # model YAML files.  Preserve first occurrence while removing duplicates.
    names = []
    seen = set()
    for name in config["hyper_parameters"]:
        if name not in seen:
            names.append(name)
            seen.add(name)
    if "seed" not in seen:
        names.insert(0, "seed")
    config["hyper_parameters"] = names

    hyper_ls = [_as_search_space(config[name]) for name in names]
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    for idx, hyper_tuple in enumerate(combinators):
        for name, value in zip(names, hyper_tuple):
            config[name] = value
        init_seed(config["seed"])

        logger.info(
            "========={}/{}: Parameters:{}={}=======".format(
                idx + 1, total_loops, names, hyper_tuple
            )
        )

        train_data.pretrain_setup()
        model_instance = get_model(config["model"])(config, train_data).to(
            config["device"]
        )
        logger.info(model_instance)

        trainer = get_trainer()(config, model_instance, mg)
        _, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data,
            valid_data=valid_data,
            test_data=test_data,
            saved=save_model,
        )
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        current_valid_value = best_valid_result[val_metric]
        is_better = (
            current_valid_value > best_valid_value
            if valid_metric_bigger
            else current_valid_value < best_valid_value
        )
        if is_better:
            best_valid_value = current_valid_value
            best_valid_idx = idx

        logger.info("best valid result: {}".format(dict2str(best_valid_result)))
        logger.info("test result: {}".format(dict2str(best_test_upon_valid)))
        logger.info(
            "████Current BEST by validation████:\nParameters: {}={},\n"
            "Valid: {},\nTest: {}\n\n\n".format(
                names,
                hyper_ret[best_valid_idx][0],
                dict2str(hyper_ret[best_valid_idx][1]),
                dict2str(hyper_ret[best_valid_idx][2]),
            )
        )

    logger.info("\n============All Over=====================")
    for parameters, valid_result, test_result in hyper_ret:
        logger.info(
            "Parameters: {}={},\n best valid: {},\n best test: {}".format(
                names,
                parameters,
                dict2str(valid_result),
                dict2str(test_result),
            )
        )

    logger.info("\n\n█████████████ BEST BY VALIDATION ████████████████")
    logger.info(
        "\tParameters: {}={},\nValid: {},\nTest: {}\n\n".format(
            names,
            hyper_ret[best_valid_idx][0],
            dict2str(hyper_ret[best_valid_idx][1]),
            dict2str(hyper_ret[best_valid_idx][2]),
        )
    )
