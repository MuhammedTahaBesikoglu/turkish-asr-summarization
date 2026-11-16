# Turkish ASR + Summarization

End-to-end system that transcribes Turkish speech (meetings, podcasts, lectures) with a
fine-tuned [Whisper](https://huggingface.co/openai/whisper-small) model and summarizes the
transcript with a language model.

> **Status:** work in progress — stage 1 (setup). All results below are **TBD** until the
> evaluation scripts have actually been run.

**Demo:** TBD (Hugging Face Spaces) · **Model:** TBD (Hugging Face Hub)

## Architecture

```
audio ──► Whisper (fine-tuned, Turkish) ──► transcript ──► summarizer ──► summary
```

## Results

Test split: Common Voice Turkish (official `test` split, version TBD).

| Model                     | WER (normalized) | CER (normalized) | WER (raw) |
|---------------------------|------------------|------------------|-----------|
| `whisper-small` zero-shot | TBD              | TBD              | TBD       |
| `whisper-small` full FT   | TBD              | TBD              | TBD       |
| `whisper-small` LoRA      | TBD              | TBD              | TBD       |

### Full fine-tune vs. LoRA

| Method    | WER | Training time | Peak GPU memory | Trainable params |
|-----------|-----|---------------|-----------------|------------------|
| Full FT   | TBD | TBD           | TBD             | TBD              |
| LoRA      | TBD | TBD           | TBD             | TBD              |

## Setup

Requirements: Python 3.10+ (developed on 3.12.10), an NVIDIA GPU is recommended.
Training was run on Google Colab (T4); local development on a 4 GB laptop GPU.

```bash
git clone <repo-url>
cd turkish-asr-summarization
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# 1. PyTorch with CUDA (version used for this project: 2.15.0 + CUDA 13.2).
#    Needs an NVIDIA driver supporting CUDA 13.2+. For other setups, pick the
#    matching command at https://pytorch.org/get-started/locally/.
#    On Google Colab, skip this step and keep the preinstalled PyTorch.
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132

# 2. Project dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# 3. Verify
python scripts/check_env.py
pytest
ruff check .
```

## Data

[Mozilla Common Voice](https://commonvoice.mozilla.org/) Turkish (`tr`), version **TBD**.

Since October 2025 Common Voice is distributed only through the
[Mozilla Data Collective](https://datacollective.mozillafoundation.org); it is no longer
available via `datasets.load_dataset` from the Hugging Face Hub. Download the Turkish
release manually and extract it under `data/raw/` (the `data/` folder is git-ignored).
The official train / dev / test splits are used; the test split is never used for training.

## Project structure

```
configs/          YAML configs for training and evaluation
data/             raw and processed data (git-ignored)
notebooks/        exploratory and error analysis
src/
  data/           data loading, preprocessing, Turkish text normalization
  training/       fine-tuning (full + LoRA)
  evaluation/     WER/CER, baseline evaluation
  summarization/  summarization stage
  pipeline.py     audio -> transcript -> summary
app/              Gradio demo
tests/            pytest tests
results/          metric tables and plots
scripts/          utility scripts (e.g. environment check)
docs/             one Markdown doc per component (see docs/README.md)
```

## Known limitations

TBD

## License

Code: [MIT](LICENSE). Dataset and model licenses: TBD.
