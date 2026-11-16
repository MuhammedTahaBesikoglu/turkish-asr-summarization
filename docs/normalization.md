# Turkish text normalization

Module: `src/data/normalization.py` · Tests: `tests/test_normalization.py`

WER/CER compares words and characters exactly, so surface differences such as casing,
punctuation or apostrophes would be counted as recognition errors. Every evaluation in this
project (baseline and fine-tuned) applies **the same** `normalize_text` to both the reference
and the hypothesis. Raw (unnormalized) WER is reported alongside it.

## API

```python
from src.data.normalization import normalize_text, turkish_lower

turkish_lower("IŞIK İzmir")  # "ışık izmir"
normalize_text("Rüzgâr, Ankara'ya hâlâ esiyor!")  # "rüzgar ankaraya hala esiyor"
normalize_text("hâlâ", fold_circumflex=False)  # "hâlâ"
```

- `turkish_lower(text)`: NFC, then `I → ı` and `İ → i`, then `str.lower()` for the rest.
- `normalize_text(text, *, fold_circumflex=True)`: the full pipeline below.

Pure standard library, so it runs in CI without the ML stack.

## Rules

Applied in this order:

| # | Rule | Example | Why |
|---|------|---------|-----|
| 1 | Unicode NFC | `I` + U+0307 → `İ` | Decomposed input would otherwise skip rule 2 |
| 2 | Turkish lowercasing | `IŞIK` → `ışık`, `İzmir` → `izmir` | `str.lower()` gives `işik` and `i̇zmir` (stray combining dot) |
| 3 | Circumflex folding (optional) | `hâlâ` → `hala` | Used inconsistently in the data |
| 4 | Remove apostrophes `' ’ ‘ ` ´ ʼ` | `Ankara'ya` → `ankaraya` | Keeps the word as one token with or without the apostrophe |
| 5 | Other non-alphanumeric → space | `Renault-Nissan` → `renault nissan`, `…?!"` → removed | Punctuation is not part of what is spoken |
| 6 | Collapse whitespace, strip | `a   b ` → `a b` | |

Digits and non-Turkish Latin letters (`é`, `ë`) are kept as they are.

## Data-driven decisions

The rules were chosen from a scan of the `sentence` column of Common Voice 27.0 `tr`
(train / dev / test):

- **No digits in any reference.** Numbers are written out ("yüzde kırk dört virgül seksen
  dört"). Digits are therefore *not* converted to words for now. If the baseline shows that
  Whisper often outputs digits, converting hypothesis digits to Turkish words will be
  added here (after checking that the library used really supports Turkish), and the doc
  and tests will be updated.
- **Circumflex is inconsistent:** 523 train, 142 dev and 142 test sentences contain
  â/î/û, and the same word appears with and without it ("hükûmet" / "hükümet"). Folding
  removes this noise. This does not fix every variant: `hükûmet` becomes `hükumet`, which
  still differs from `hükümet`.
- **Three apostrophe styles** (`'`, `’`, `‘`) appear. All are removed.
- **Hyphens** join compounds ("Müdafaa-i Hukuk", "kasım-nisan"). They become spaces.
- Every sentence is already NFC.

## Known limitations

- A few references are corrupted: digits were lost upstream and replaced by `�` or left as
  `%` (e.g. "fonların yaklaşık �'i"; 1 such sentence in test, 2 in dev). After normalization
  a meaningless suffix such as `i` remains, so these clips always produce errors.
- Spelling variants beyond the circumflex (e.g. `hükumet` vs `hükümet`) are not unified.
- Abbreviations and Roman numerals are not expanded.
