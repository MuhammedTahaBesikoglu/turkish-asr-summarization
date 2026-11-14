"""Print the versions of the key libraries and whether a CUDA GPU is usable.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

PACKAGES = [
    "torch",
    "torchaudio",
    "transformers",
    "datasets",
    "evaluate",
    "accelerate",
    "peft",
    "jiwer",
    "soundfile",
    "librosa",
    "pandas",
    "pyyaml",
]


def package_version(name: str) -> str:
    """Return the installed version of ``name`` or ``"not installed"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def report_gpu() -> None:
    """Print CUDA availability and basic GPU info via PyTorch."""
    try:
        import torch
    except ImportError:
        print("torch: not importable, skipping GPU check")
        return

    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"CUDA (torch)   : {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU            : {props.name}")
        print(f"VRAM           : {props.total_memory / 1024**3:.1f} GiB")
        print(f"bf16 supported : {torch.cuda.is_bf16_supported()}")


def main() -> None:
    print(f"Python         : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print()
    for name in PACKAGES:
        print(f"{name:<15}: {package_version(name)}")
    print()
    report_gpu()


if __name__ == "__main__":
    main()
