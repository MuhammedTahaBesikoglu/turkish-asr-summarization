"""Smoke test: the package and its subpackages import cleanly."""

import importlib

import pytest

MODULES = [
    "src",
    "src.data",
    "src.training",
    "src.evaluation",
    "src.summarization",
    "src.pipeline",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
