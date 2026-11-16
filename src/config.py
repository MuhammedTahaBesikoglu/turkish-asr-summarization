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
