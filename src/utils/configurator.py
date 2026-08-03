# coding: utf-8
"""Configuration loading for MMRec and CTGTRec.

Configuration files are merged from lowest to highest priority:

1. ``src/configs/overall.yaml``;
2. ``src/configs/dataset/<dataset>.yaml``;
3. ``src/configs/model/<model>.yaml``;
4. ``src/configs/mg.yaml`` when multi-GPU mode is enabled;
5. ``src/configs/final/<model-lowercase>/<dataset-lowercase>.yaml`` when present;
6. values supplied by the command-line entry point.

Configuration discovery and project paths are derived from this module's file
location, not from the process working directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
CONFIG_ROOT = SRC_ROOT / "configs"


class Config(object):
    """Load, validate, merge, and resolve project configuration."""

    def __init__(self, model=None, dataset=None, config_dict=None, mg=False):
        external_config = dict(config_dict or {})
        external_config["model"] = model
        external_config["dataset"] = dataset

        self.final_config_dict = self._load_dataset_model_config(
            external_config,
            mg,
        )
        # Explicit values supplied by the entry point have highest priority.
        self.final_config_dict.update(external_config)

        self._set_project_metadata()
        self._set_default_parameters()
        self._resolve_project_paths()
        self._init_device()

    @staticmethod
    def _name(value: Any) -> str:
        if isinstance(value, str):
            name = value.strip()
        else:
            name = getattr(value, "__name__", str(value)).strip()
        if not name:
            raise ValueError("Model and dataset names must be non-empty.")
        return name

    def _config_paths(self, config_dict, mg):
        model_name = self._name(config_dict["model"])
        dataset_name = self._name(config_dict["dataset"])

        required = [
            CONFIG_ROOT / "overall.yaml",
            CONFIG_ROOT / "dataset" / "{}.yaml".format(dataset_name),
            CONFIG_ROOT / "model" / "{}.yaml".format(model_name),
        ]
        optional = []

        if mg:
            required.append(CONFIG_ROOT / "mg.yaml")

        optional.append(
            CONFIG_ROOT
            / "final"
            / model_name.lower()
            / "{}.yaml".format(dataset_name.lower())
        )
        return required, optional

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            file_data = yaml.load(
                handle.read(),
                Loader=Config._build_yaml_loader(),
            )

        if file_data is None:
            return {}
        if not isinstance(file_data, dict):
            raise TypeError(
                "Configuration file must contain a YAML mapping: {}".format(
                    path
                )
            )
        return file_data

    def _load_dataset_model_config(self, config_dict, mg):
        required, optional = self._config_paths(config_dict, mg)
        missing = [path for path in required if not path.is_file()]
        if missing:
            formatted = "\n".join("  - {}".format(path) for path in missing)
            raise FileNotFoundError(
                "Required configuration file(s) not found:\n{}".format(
                    formatted
                )
            )

        file_config_dict: dict[str, Any] = {}
        hyper_parameters = []
        loaded_files = []

        for path in [*required, *optional]:
            if not path.is_file():
                continue

            file_data = self._read_yaml(path)
            file_hyper_parameters = file_data.get("hyper_parameters")
            if file_hyper_parameters is not None:
                if not isinstance(file_hyper_parameters, (list, tuple)):
                    raise TypeError(
                        "hyper_parameters must be a list in {}, received {}."
                        .format(
                            path,
                            type(file_hyper_parameters).__name__,
                        )
                    )
                hyper_parameters.extend(file_hyper_parameters)

            file_config_dict.update(file_data)
            loaded_files.append(str(path.resolve()))

        file_config_dict["hyper_parameters"] = hyper_parameters
        file_config_dict["loaded_config_files"] = loaded_files
        return file_config_dict

    @staticmethod
    def _build_yaml_loader():
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

    def _set_project_metadata(self):
        self.final_config_dict["project_root"] = str(PROJECT_ROOT)
        self.final_config_dict["src_root"] = str(SRC_ROOT)
        self.final_config_dict["config_root"] = str(CONFIG_ROOT)

    @staticmethod
    def _stable_unique(values: Iterable[Any]) -> list[Any]:
        result = []
        for value in values:
            if value not in result:
                result.append(value)
        return result

    def _set_default_parameters(self):
        valid_metric_value = self.final_config_dict.get("valid_metric")
        if not isinstance(valid_metric_value, str) or not valid_metric_value:
            raise ValueError("valid_metric must be a non-empty string.")

        smaller_metric = {"rmse", "mae", "logloss"}
        valid_metric_name = valid_metric_value.split("@", 1)[0].lower()
        self.final_config_dict["valid_metric_bigger"] = (
            valid_metric_name not in smaller_metric
        )

        hyper_parameters = self.final_config_dict.get(
            "hyper_parameters",
            [],
        )
        if not isinstance(hyper_parameters, (list, tuple)):
            raise TypeError("hyper_parameters must be a list.")
        hyper_parameters = self._stable_unique(hyper_parameters)
        if "seed" not in hyper_parameters:
            hyper_parameters.append("seed")
        self.final_config_dict["hyper_parameters"] = hyper_parameters

    @staticmethod
    def _resolve_path(value: Any, key: str) -> Path:
        if value in {None, ""}:
            raise ValueError("{} must not be empty.".format(key))
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def _resolve_project_paths(self):
        data_path = self._resolve_path(
            self.final_config_dict.get("data_path"),
            "data_path",
        )
        # Several upstream models concatenate data_path and dataset as strings.
        # Preserve that compatibility with one explicit trailing separator.
        self.final_config_dict["data_path"] = (
            str(data_path).rstrip("/\\") + os.sep
        )

        path_keys = (
            "checkpoint_dir",
            "result_dir",
            "recommend_topk",
            "log_dir",
        )
        for key in path_keys:
            if key not in self.final_config_dict:
                raise KeyError(
                    "Required project path {!r} is missing from configuration."
                    .format(key)
                )
            self.final_config_dict[key] = str(
                self._resolve_path(
                    self.final_config_dict[key],
                    key,
                )
            )

        checkpoint_file = self.final_config_dict.get("checkpoint_file")
        if checkpoint_file:
            self.final_config_dict["checkpoint_file"] = str(
                self._resolve_path(checkpoint_file, "checkpoint_file")
            )

    def _init_device(self):
        use_gpu = bool(self.final_config_dict["use_gpu"])
        gpu_id = int(self.final_config_dict["gpu_id"])
        if gpu_id < 0:
            raise ValueError("gpu_id must be non-negative.")

        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.final_config_dict["device"] = torch.device(
            "cuda"
            if torch.cuda.is_available() and use_gpu
            else "cpu"
        )

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Configuration keys must be strings.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        return self.final_config_dict.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("Configuration keys must be strings.")
        return key in self.final_config_dict

    def __str__(self):
        lines = [
            "{}={}".format(key, value)
            for key, value in self.final_config_dict.items()
        ]
        return "\n" + "\n".join(lines) + "\n"

    def __repr__(self):
        return self.__str__()
