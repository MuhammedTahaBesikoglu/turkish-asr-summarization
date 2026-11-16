# Turkish ASR + Summarization

End-to-end system that transcribes Turkish speech (meetings, podcasts, lectures) with a
fine-tuned [Whisper](https://huggingface.co/openai/whisper-small) model and summarizes the
transcript with a language model.

> **Status:** work in progress. Stages 1–4 (setup, data, normalization, zero-shot baseline)
> are done; fine-tuning is next. Results not yet measured are marked **TBD**.

**Demo:** TBD (Hugging Face Spaces) · **Model:** TBD (Hugging Face Hub)

## Architecture

```
audio ──► Whisper (fine-tuned, Turkish) ──► transcript ──► summarizer ──► summary
```

## Results

Test split: Common Voice Turkish 27.0, official `test` split (11,874 clips, 1,500 speakers).
Corpus-level WER/CER; "normalized" uses the Turkish normalizer in
[docs/normalization.md](docs/normalization.md) (casing, punctuation, numbers spelled out).
How to reproduce: [docs/baseline.md](docs/baseline.md).

| Model                     | WER (normalized) | CER (normalized) | WER (raw) |
|---------------------------|------------------|------------------|-----------|
| `whisper-small` zero-shot | 21.82%           | 6.98%            | 32.17%    |
| `whisper-small` full FT   | TBD              | TBD              | TBD       |
| `whisper-small` LoRA      | TBD              | TBD              | TBD       |

### Full fine-tune vs. LoRA

| Method    | WER | Training time | Peak GPU memory | Trainable params |
|-----------|-----|---------------|-----------------|------------------|
| Full FT   | TBD | TBD           | TBD             | TBD              |
| LoRA      | TBD | TBD           | TBD             | TBD              |

## Setup

Requirements: Python 3.12+ (developed on 3.12.10), an NVIDIA GPU is recommended.
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

[Mozilla Common Voice](https://commonvoice.mozilla.org/) Turkish (`tr`), version **27.0**
(`cv-corpus-27.0-2026-09-11`), released under CC0.

Since October 2025 Common Voice is distributed only through the
[Mozilla Data Collective](https://datacollective.mozillafoundation.org); it is no longer
available via `datasets.load_dataset` from the Hugging Face Hub. Download the Turkish
release manually and extract it under `data/raw/` (the `data/` folder is git-ignored), then
set `dataset.root` in `configs/data.yaml`.
The official train / dev / test splits are used; the test split is never used for training.

| | train | dev | test |
|---|---|---|---|
| Clips | 41,465 | 11,864 | 11,874 |
| Speakers | 28 | 163 | 1,500 |
| Hours | 45.27 | 12.35 | 14.54 |

The training split has very few speakers (top 5 = 62.4% of clips). Details, figures and other
findings: [docs/data.md](docs/data.md), `notebooks/01_data_exploration.ipynb`.

```bash
python -m src.data.stats   # writes results/data_summary.md
```

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
