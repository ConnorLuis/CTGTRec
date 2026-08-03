from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_python_sources_compile() -> None:
    sources = []
    for directory in (PROJECT_ROOT / "preprocessing", PROJECT_ROOT / "src"):
        if directory.is_dir():
            sources.extend(directory.rglob("*.py"))
    assert sources
    for source in sources:
        py_compile.compile(str(source), doraise=True)


def test_all_yaml_and_citation_files_parse() -> None:
    yaml_files = []
    config_root = PROJECT_ROOT / "src" / "configs"
    if config_root.is_dir():
        yaml_files.extend(config_root.rglob("*.yaml"))
        yaml_files.extend(config_root.rglob("*.yml"))
    workflow_root = PROJECT_ROOT / ".github" / "workflows"
    if workflow_root.is_dir():
        yaml_files.extend(workflow_root.rglob("*.yaml"))
        yaml_files.extend(workflow_root.rglob("*.yml"))
    citation = PROJECT_ROOT / "CITATION.cff"
    if citation.is_file():
        yaml_files.append(citation)

    for path in yaml_files:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), path


def test_cli_help_is_lightweight_and_strict() -> None:
    entry = PROJECT_ROOT / "src" / "main.py"
    if not entry.is_file():
        pytest.skip("src/main.py is not included in this overlay test bundle")

    completed = subprocess.run(
        [sys.executable, str(entry), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--show-config" in completed.stdout
    assert "--no-save-model" in completed.stdout

    unknown = subprocess.run(
        [sys.executable, str(entry), "--definitely-unknown"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert unknown.returncode != 0
    assert "unrecognized arguments" in unknown.stderr
