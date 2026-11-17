"""Tests for config loading helpers."""

import pytest

from src.config import apply_overrides


def test_apply_overrides_parses_yaml_values() -> None:
    config = {"training": {"max_steps": 4000, "learning_rate": 1e-5}, "tracking": {"db": "x"}}
    apply_overrides(config, ["training.max_steps=10", "training.learning_rate=1.0e-3"])
    assert config["training"] == {"max_steps": 10, "learning_rate": 1e-3}
    apply_overrides(config, ["tracking=null", "training.fp16=false"])
    assert config["tracking"] is None
    assert config["training"]["fp16"] is False


def test_apply_overrides_keeps_value_with_equals_sign() -> None:
    config = {"a": {"b": 1}}
    apply_overrides(config, ["a.b=x=y"])
    assert config["a"]["b"] == "x=y"


def test_apply_overrides_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="key.subkey=value"):
        apply_overrides({}, ["no_equals_sign"])
    with pytest.raises(KeyError, match="not a section"):
        apply_overrides({"training": {}}, ["trainig.max_steps=1"])
