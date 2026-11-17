"""Load YAML configs and resolve the paths they contain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a dict."""
    import yaml  # imported lazily so the package imports without pyyaml

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping, got {type(config).__name__}")
    return config


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Return ``path`` as absolute; relative paths are taken relative to ``base``."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.subkey=value`` overrides (values parsed as YAML) to ``config`` in place.

    Example: ``["training.max_steps=10", "tracking=null"]``. Intermediate keys must exist,
    so a typo fails loudly instead of silently adding a new key.
    """
    import yaml

    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Override {item!r}: {part!r} is not a section of the config")
            node = node[part]
        node[leaf] = yaml.safe_load(raw_value)
    return config
