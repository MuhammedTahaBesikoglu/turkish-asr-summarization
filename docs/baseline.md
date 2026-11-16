# Zero-shot baseline

Modules: `src/evaluation/evaluate_asr.py`, `src/evaluation/metrics.py` ·
Config: `configs/eval_baseline.yaml` · Colab: `notebooks/colab_baseline.ipynb` ·
Tests: `tests/test_metrics.py`

The pretrained `openai/whisper-small`, without any fine-tuning, transcribes the full
Common Voice 27.0 `tr` **test** split (11,874 clips). The same script and metrics are reused
for the fine-tuned models, so the numbers are directly comparable.

## Results

| Model | Clips | WER (normalized) | CER (normalized) | WER (raw) | CER (raw) |
|---|---|---|---|---|---|
| `whisper-small` zero-shot | 11,874 | 21.82% | 6.98% | 32.17% | 10.36% |

Source: [`results/baseline/whisper-small-zeroshot/metrics.json`](../results/baseline/whisper-small-zeroshot/metrics.json)
(full split, `is_full_split: true`).

| | Normalized | Raw |
|---|---|---|
| Substitutions | 11,164 | 17,119 |
| Deletions | 1,278 | 2,312 |
| Insertions | 2,200 | 2,135 |
| Reference words | 67,099 | 67,032 |

The two rows differ in reference word count because normalization changes the
tokens, for example by spelling out numbers. 1,106 hypotheses contain digits while no reference
does, so every digit is an error on raw text; this, together with casing and punctuation, is
why raw WER is higher. No hypothesis was empty.

Run: Colab, Tesla T4, fp16, 672.9 s (17.65 clips/s), peak GPU memory 2.5 GiB. Code at commit
`08e7b72`, Python 3.13.16 (Colab runtime), torch 2.14.1+cu130, transformers 5.23.0,
jiwer 4.0.0, librosa 1.0.1.

## Running

Colab (T4), as used for the reported numbers:

1. Locally: `python scripts/pack_data.py --splits test`, which writes
   `data/colab/cv-corpus-27.0-2026-09-11-test.tar` (355 MiB, test TSV and clips only).
2. Upload the `.tar` to Google Drive (`MyDrive/turkish-asr/` by default).
3. Open `notebooks/colab_baseline.ipynb` in Colab and run all cells.
4. Copy `MyDrive/turkish-asr/results/baseline/whisper-small-zeroshot/` into `results/baseline/`
   and commit it.

Locally (any GPU, or CPU), for example a quick check on 100 clips:

```bash
python -m src.evaluation.evaluate_asr --config configs/eval_baseline.yaml \
    --max-samples 100 --output-dir /tmp/smoke
```

CLI overrides: `--data-config` (e.g. `configs/data_colab.yaml`), `--max-samples`,
`--output-dir`.

## Configuration

| Key | Value | Notes |
|---|---|---|
| `model.name` | `openai/whisper-small` | Hub id or local directory |
| `model.language` / `task` | `tr` / `transcribe` | Forces Turkish; no language detection |
| `eval.split` | `test` | |
| `eval.max_samples` | `null` | Whole split |
| `eval.batch_size` | 32 | Peak GPU memory 2.36 GiB in fp16 in a local smoke run |
| `eval.num_beams` | 1 | Greedy decoding |
| `eval.max_length` | 225 | Decoder tokens incl. 4 prompt tokens |
| `eval.fp16` | `true` | Ignored on CPU |
| `eval.num_workers` | 2 | DataLoader workers decoding MP3 |

## Outputs

`results/baseline/whisper-small-zeroshot/`:

- `predictions.csv`: `clip, client_id, duration_s, reference, hypothesis,
  reference_normalized, hypothesis_normalized, wer` (one row per clip, for error analysis).
- `metrics.json`: metrics, number of hypotheses containing digits, empty hypotheses, run time,
  clips/s, GPU and peak memory, library versions, git commit, and the full config.
  `is_full_split` is `false` for runs limited with `max_samples`; only full-split runs are
  reported.

## Metrics

- **Corpus-level** WER/CER: total edits divided by total reference words (characters) over
  the whole split, not a mean of per-clip rates (`jiwer.process_words` /
  `process_characters`).
- **Normalized:** both sides go through `normalize_text` (see
  [normalization.md](normalization.md)): Turkish lowercasing, numbers spelled out,
  punctuation and apostrophes removed, circumflex folded.
- **Raw:** only whitespace is collapsed; casing, punctuation and digits count as errors.

## Notes

- Audio is decoded with `librosa` at 16 kHz in DataLoader workers and padded to 30 s by the
  Whisper feature extractor (no clip is longer than 13.9 s).
- transformers prints once: *"The attention mask is not set with a batched input..."*.
  It is harmless here: every input is padded to the same 30 s and every sequence starts with
  the same forced decoder prompt, so there is no padding to mask.
- A local smoke run on the first 100 test clips (RTX 3050 Ti, fp16) took about 33 s including
  model loading. These numbers are only a functional check and are not reported.
