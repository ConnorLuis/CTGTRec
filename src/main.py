# coding: utf-8
"""CTGTRec command-line entry point.

The canonical invocation is run from the repository root:

    python src/main.py --model CTGTRec --dataset baby

Path resolution is based on this file's location rather than the caller's
current working directory, so configuration, data, logs, checkpoints, and
result files remain anchored to the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _parse_override(text: str) -> tuple[str, Any]:
    """Parse one ``KEY=VALUE`` override using YAML scalar/list syntax."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "--set values must use KEY=VALUE syntax."
        )
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            "--set requires a non-empty configuration key."
        )

    reserved = {
        "model",
        "dataset",
        "project_root",
        "src_root",
        "config_root",
        "loaded_config_files",
    }
    if key in reserved:
        raise argparse.ArgumentTypeError(
            "Configuration key {!r} is managed by the entry point and cannot "
            "be overridden with --set.".format(key)
        )

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid YAML value in override {!r}: {}".format(text, exc)
        ) from exc
    return key, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate CTGTRec or an included baseline from the "
            "repository root."
        )
    )
    parser.add_argument(
        "--model",
        "-m",
        default="CTGTRec",
        help="Model class/config name (default: CTGTRec).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="baby",
        help="Dataset/config name (default: baby).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="CUDA device index. Overrides gpu_id from overall.yaml.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable CUDA even when a GPU is available.",
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each seed's best validation checkpoint (default: enabled).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        type=_parse_override,
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a configuration value for debugging. May be repeated; "
            "values use YAML syntax, for example --set epochs=2."
        ),
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Resolve and print the merged configuration without loading data.",
    )
    parser.add_argument(
        "--numexpr-max-threads",
        type=int,
        default=int(os.environ.get("NUMEXPR_MAX_THREADS", "48")),
        help="Set NUMEXPR_MAX_THREADS before importing the training stack.",
    )
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=None,
        help="Optionally set OMP_NUM_THREADS before importing PyTorch modules.",
    )
    return parser


def _prepare_runtime(args: argparse.Namespace) -> None:
    if args.numexpr_max_threads <= 0:
        raise ValueError("--numexpr-max-threads must be positive.")
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.numexpr_max_threads)

    if args.omp_num_threads is not None:
        if args.omp_num_threads <= 0:
            raise ValueError("--omp-num-threads must be positive.")
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    # Support both ``python src/main.py`` and ``python -m src.main``.
    src_text = str(SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

    # Some legacy baseline code still writes relative outputs. Standardize its
    # working directory without making config discovery depend on cwd.
    os.chdir(PROJECT_ROOT)


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config_dict: dict[str, Any] = {}
    for key, value in args.overrides:
        config_dict[key] = value

    if args.gpu_id is not None:
        if args.gpu_id < 0:
            raise ValueError("--gpu-id must be non-negative.")
        config_dict["gpu_id"] = args.gpu_id
    if args.cpu:
        config_dict["use_gpu"] = False
    return config_dict


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _prepare_runtime(args)
        config_dict = _config_overrides(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.show_config:
        from utils.configurator import Config

        config = Config(
            model=args.model,
            dataset=args.dataset,
            config_dict=config_dict,
        )
        print(config)
        return 0

    from utils.quick_start import quick_start

    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config_dict,
        save_model=args.save_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
