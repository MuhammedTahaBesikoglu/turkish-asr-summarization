"""Smoke test: the package and its subpackages import cleanly."""

import importlib

import pytest

MODULES = [
    "src",
    "src.config",
    "src.data",
    "src.data.normalization",
    "src.data.common_voice",
    "src.data.stats",
    "src.training",
    "src.evaluation",
    "src.evaluation.metrics",
    "src.evaluation.evaluate_asr",
    "src.summarization",
    "src.pipeline",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
