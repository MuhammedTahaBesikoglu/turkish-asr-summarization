"""Provenance recorded with every result file: git commit and library versions."""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata

from src.config import REPO_ROOT


def git_commit() -> str | None:
    """Return the current commit hash, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def library_versions(names: list[str]) -> dict[str, str]:
    """Return the Python version and the installed version of each package in ``names``."""
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result
