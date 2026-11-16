# Environment setup

## Target environments

| Use | Hardware | Notes |
|-----|----------|-------|
| Development, tests, small evaluations | Windows 11, RTX 3050 Ti Laptop (4 GB VRAM), Python 3.12 (`.venv`) | No full fine-tuning locally |
| Training | Google Colab, T4 (16 GB VRAM) | Keep Colab's preinstalled PyTorch |

Code must run in both places; paths come from YAML files under `configs/`, never hard-coded.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch first (needs a CUDA-specific index URL, so it is not in requirements.txt)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132

pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

- `requirements.txt` — runtime ML stack, direct dependencies pinned to locally verified versions.
- `requirements-dev.txt` — ruff, pytest, ipykernel (also used by CI).
- `pyproject.toml` — package metadata and ruff/pytest config. `dependencies` is intentionally
  empty so CI can install the package without the ML stack.
- Audio I/O and resampling use `soundfile` + `librosa`; `torchaudio` is not used.

## `scripts/check_env.py`

Prints the Python version, the installed version of each key library, and GPU information
(CUDA availability, device name, VRAM, bf16 support).

```bash
python scripts/check_env.py
```

A library that is missing shows as `not installed`; if `torch` cannot be imported the GPU check
is skipped instead of failing.
