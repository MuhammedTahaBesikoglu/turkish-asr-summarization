# Data: Common Voice Turkish

Modules: `src/data/common_voice.py`, `src/data/stats.py`, `src/config.py` ·
Config: `configs/data.yaml` · Tests: `tests/test_common_voice.py` ·
Analysis: `notebooks/01_data_exploration.ipynb`

## Dataset

| | |
|---|---|
| Release | Common Voice **27.0** (`cv-corpus-27.0-2026-09-11`), locale `tr` |
| Source | [Mozilla Data Collective](https://datacollective.mozillafoundation.org), downloaded manually |
| License | CC0 (per the release's `README.md`); users agree not to try to identify speakers |
| Audio | MP3 clips in `clips/`, decoded and resampled to 16 kHz mono |

Since October 2025 Common Voice is no longer served through `datasets.load_dataset`, so the
code reads the extracted release from disk. Expected layout:

```
data/raw/cv-corpus-27.0-2026-09-11/tr/
├── clips/                 # *.mp3
├── train.tsv  dev.tsv  test.tsv
├── clip_durations.tsv     # clip, duration[ms]
└── validated.tsv  other.tsv  invalidated.tsv  ...
```

## Configuration

```yaml
# configs/data.yaml
dataset:
  version: "27.0"
  root: data/raw/cv-corpus-27.0-2026-09-11/tr  # relative to repo root, or absolute
audio:
  sampling_rate: 16000
```

On Colab, point `dataset.root` to an absolute path (e.g. on Google Drive).

## API

```python
from src.config import load_config, resolve_path
from src.data.common_voice import load_audio, load_split, missing_audio

config = load_config("configs/data.yaml")
root = resolve_path(config["dataset"]["root"])

train = load_split(root, "train")  # list[Clip]
clip = train[0]
clip.audio_path, clip.sentence, clip.client_id, clip.duration_s
audio = load_audio(clip.audio_path, sampling_rate=16000)  # np.ndarray, float32, mono
assert not missing_audio(train)
```

- `load_split(root, split)` accepts only `train`, `dev`, `test` (the official splits). The test
  split is for evaluation only and is never used for training.
- TSV files are read with quoting disabled: sentences contain literal `"` characters.
- `duration_s` comes from `clip_durations.tsv`, so statistics need no audio decoding.
- The reader uses only the standard library; `librosa` is imported lazily by `load_audio`
  and `pyyaml` by `load_config`, so the tests run in CI without the ML stack.

## Statistics

```bash
python -m src.data.stats --config configs/data.yaml --out results/data_summary
```

Writes `results/data_summary.md` and `.json` and fails if any audio file is missing.
Figures are produced by the notebook in `results/figures/`.

Summary (from `results/data_summary.md`):

| | train | dev | test |
|---|---|---|---|
| Clips | 41,465 | 11,864 | 11,874 |
| Speakers | 28 | 163 | 1,500 |
| Hours | 45.27 | 12.35 | 14.54 |
| Median duration (s) | 3.28 | 3.28 | 4.10 |
| Max duration (s) | 13.32 | 11.02 | 13.86 |
| Median words / sentence | 4 | 4 | 5 |

## Findings

- **Low speaker diversity in train:** 28 speakers; the largest has 20.2% of train clips, the
  top 5 have 62.4%. The splits share no speakers and no sentences, so test WER measures
  generalization to unseen voices.
- **Demographic shift:** train is dominated by older speakers (sixties 33.7%, eighties 8.3%);
  most test clips have no declared age (60.1%) or gender (66.4%).
- **No clip exceeds 30 s** (Whisper's input window), so no segmentation is needed here.
- **Open vocabulary:** 41.3% of normalized test word types (14.9% of tokens) never appear in
  the train transcripts.
- No reference contains digits (see [normalization.md](normalization.md)).

## Open decision for training (stage 5): extra validated clips

`validated.tsv` has 56,409 clips that are in none of the official splits. Of these, 23,879
(23.36 h) share neither a speaker nor a sentence with dev/test and could be added to train
without leaking evaluation data. They come from only 23 speakers, and the top 5 contribute
93.8%, so they add hours but almost no speaker diversity. Whether to use them is decided
before fine-tuning; the test split stays untouched either way.

## Known limitations

- Three references have lost their digits upstream (e.g. "fonların yaklaşık �'i"; 1 in test,
  2 in dev).
- Self-declared age/gender is missing for many clips, so demographic breakdowns of WER are
  only possible on a subset.
