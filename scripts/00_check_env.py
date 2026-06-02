#!/usr/bin/env python
"""Check server environment without downloading data or model weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


OPTIONAL_MODULES = [
    "torch",
    "transformers",
    "physical_ai_av",
    "flash_attn",
    "accelerate",
    "pandas",
    "av",
]


def module_available(name: str) -> bool:
    """Return whether a module can be imported, without importing it."""
    return importlib.util.find_spec(name) is not None


def build_report() -> dict[str, Any]:
    """Collect environment facts that are safe to query locally."""
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "requires_python": "3.12.*",
            "version_ok": sys.version_info[:2] == (3, 12),
        },
        "commands": {
            "uv": shutil.which("uv"),
            "conda": shutil.which("conda"),
            "hf": shutil.which("hf"),
            "git": shutil.which("git"),
            "nvcc": shutil.which("nvcc"),
        },
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "modules": {name: module_available(name) for name in OPTIONAL_MODULES},
        "huggingface": {
            "hf_token_env_present": bool(os.environ.get("HF_TOKEN")),
            "hf_home": os.environ.get("HF_HOME"),
        },
        "repository": {
            "has_pyproject": Path("pyproject.toml").exists(),
            "has_official_test_inference": Path("src/alpamayo1_5/test_inference.py").exists(),
            "has_configs": Path("configs").exists(),
        },
    }

    if report["modules"]["torch"]:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }

    return report


def has_environment_manager(report: dict[str, Any]) -> bool:
    """Return whether uv, conda, or an active virtual environment is available."""
    commands = report["commands"]
    environment = report["environment"]
    return bool(
        commands.get("uv")
        or commands.get("conda")
        or environment.get("conda_prefix")
        or environment.get("virtual_env")
    )


def find_failures(report: dict[str, Any], strict: bool) -> list[str]:
    """Return environment failures."""
    failures: list[str] = []
    if not report["python"]["version_ok"]:
        failures.append("Python version is not 3.12.*")
    if strict:
        if not has_environment_manager(report):
            failures.append("Missing environment manager: uv, conda, or active virtual environment")
        for command in ("hf", "git"):
            if not report["commands"].get(command):
                failures.append(f"Missing command: {command}")
        for module in ("torch", "transformers", "physical_ai_av"):
            if not report["modules"][module]:
                failures.append(f"Missing Python module: {module}")
        torch_report = report.get("torch")
        if isinstance(torch_report, dict) and not torch_report.get("cuda_available"):
            failures.append("Torch is installed but CUDA is not available")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on missing server deps.")
    parser.add_argument("--json", action="store_true", help="Print only JSON.")
    args = parser.parse_args()

    report = build_report()
    failures = find_failures(report, strict=args.strict)
    report["failures"] = failures

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Faithful-Alpamayo-1.5 environment check")
        print(json.dumps(report, indent=2, sort_keys=True))
        if failures:
            print("Failures:")
            for failure in failures:
                print(f"- {failure}")

    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
