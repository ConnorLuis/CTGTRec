# coding: utf-8
# @email: enoche.chow@gmail.com

"""Configuration loading for MMRec and CTGTRec.

Configuration files are merged in this order, from lowest to highest priority:

1. ``configs/overall.yaml``;
2. ``configs/dataset/<dataset>.yaml``;
3. ``configs/model/<model>.yaml``;
4. ``configs/mg.yaml`` when multi-GPU mode is enabled;
5. ``configs/final/<model-lowercase>/<dataset-lowercase>.yaml`` when present;
6. values supplied through ``config_dict``.

The optional final model-dataset file keeps fixed paper configurations separate
from general dataset settings and from a model's shared defaults. Other models
are unaffected because no final override is loaded unless the corresponding
file exists.
"""

from __future__ import annotations

import os
import re
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import yaml


class Config(object):
    """Load and merge model, dataset, final-run, and external configuration."""

    def __init__(self, model=None, dataset=None, config_dict=None, mg=False):
        external_config = dict(config_dict or {})
        external_config["model"] = model
        external_config["dataset"] = dataset

        self.final_config_dict = self._load_dataset_model_config(
            external_config,
            mg,
        )
        # Explicit values passed by the entry point have the highest priority.
        self.final_config_dict.update(external_config)
        self._set_default_parameters()
        self._init_device()

    @staticmethod
    def _name(value: Any) -> str:
        """Return a stable file-name component for a class or string name."""
        if isinstance(value, str):
            return value
        return getattr(value, "__name__", str(value))

    def _config_paths(self, config_dict, mg):
        """Return configuration paths in merge order."""
        config_root = Path(os.getcwd()) / "configs"
        model_name = self._name(config_dict["model"])
        dataset_name = self._name(config_dict["dataset"])

        paths = [
            config_root / "overall.yaml",
            config_root / "dataset" / f"{dataset_name}.yaml",
            config_root / "model" / f"{model_name}.yaml",
        ]
        if mg:
            paths.append(config_root / "mg.yaml")

        paths.append(
            config_root
            / "final"
            / model_name.lower()
            / f"{dataset_name.lower()}.yaml"
        )
        return paths

    def _load_dataset_model_config(self, config_dict, mg):
        file_config_dict = {}
        hyper_parameters = []
        loaded_files = []

        for path in self._config_paths(config_dict, mg):
            if not path.is_file():
                continue

            with path.open("r", encoding="utf-8") as handle:
                file_data = yaml.load(
                    handle.read(),
                    Loader=self._build_yaml_loader(),
                )

            if file_data is None:
                file_data = {}
            if not isinstance(file_data, dict):
                raise TypeError(
                    f"Configuration file must contain a YAML mapping: {path}"
                )

            file_hyper_parameters = file_data.get("hyper_parameters")
            if file_hyper_parameters:
                if not isinstance(file_hyper_parameters, (list, tuple)):
                    raise TypeError(
                        f"hyper_parameters must be a list in {path}, received "
                        f"{type(file_hyper_parameters).__name__}."
                    )
                hyper_parameters.extend(file_hyper_parameters)

            file_config_dict.update(file_data)
            loaded_files.append(str(path))

        file_config_dict["hyper_parameters"] = hyper_parameters
        file_config_dict["loaded_config_files"] = loaded_files
        return file_config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u"tag:yaml.org,2002:float",
            re.compile(
                u"""^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list(u"-+0123456789."),
        )
        return loader

    def _set_default_parameters(self):
        smaller_metric = ["rmse", "mae", "logloss"]
        valid_metric = self.final_config_dict["valid_metric"].split("@")[0]
        self.final_config_dict["valid_metric_bigger"] = (
            False if valid_metric in smaller_metric else True
        )
        if "seed" not in self.final_config_dict["hyper_parameters"]:
            self.final_config_dict["hyper_parameters"].append("seed")

    def _init_device(self):
        use_gpu = self.final_config_dict["use_gpu"]
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.final_config_dict["gpu_id"]
            )
        self.final_config_dict["device"] = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        return self.final_config_dict.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = "\n"
        args_info += "\n".join(
            "{}={}".format(arg, value)
            for arg, value in self.final_config_dict.items()
        )
        args_info += "\n\n"
        return args_info

    def __repr__(self):
        return self.__str__()
