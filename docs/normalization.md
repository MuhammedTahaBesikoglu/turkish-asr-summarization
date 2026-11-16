# Turkish text normalization

Modules: `src/data/normalization.py`, `src/data/numbers.py` ·
Tests: `tests/test_normalization.py`, `tests/test_numbers.py`

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
- `normalize_text(text, *, fold_circumflex=True, spell_out_numbers=True)`: the full pipeline
  below.
- `src.data.numbers.expand_numbers(text)`: digits → Turkish words, `%` → `yüzde`.

Pure standard library, so it runs in CI without the ML stack.

## Rules

Applied in this order:

| # | Rule | Example | Why |
|---|------|---------|-----|
| 1 | Unicode NFC | `I` + U+0307 → `İ` | Decomposed input would otherwise skip rule 3 |
| 2 | Spell out numbers (optional) | `1996'da` → `bin dokuz yüz doksan altı'da`, `%44,84` → `yüzde kırk dört virgül seksen dört` | References never contain digits; Whisper often outputs them |
| 3 | Turkish lowercasing | `IŞIK` → `ışık`, `İzmir` → `izmir` | `str.lower()` gives `işik` and `i̇zmir` (stray combining dot) |
| 4 | Circumflex folding (optional) | `hâlâ` → `hala` | Used inconsistently in the data |
| 5 | Remove apostrophes `' ’ ‘ ` ´ ʼ` | `Ankara'ya` → `ankaraya` | Keeps the word as one token with or without the apostrophe |
| 6 | Other non-alphanumeric → space | `Renault-Nissan` → `renault nissan`, `…?!"` → removed | Punctuation is not part of what is spoken |
| 7 | Collapse whitespace, strip | `a   b ` → `a b` | |

Non-Turkish Latin letters (`é`, `ë`) are kept as they are.

### Number rules (`src/data/numbers.py`)

- Integers are read in Turkish with spaces between words: `11` → `on bir`, `1000` → `bin`
  (not `bir bin`), `1000000` → `bir milyon`, up to 999 trilyon; larger numbers are read digit
  by digit.
- `.` followed by groups of exactly three digits is a thousands separator (`12.500`); any
  other `.` or `,` between digits is a decimal mark read as `virgül`, and the fractional part
  is read as a whole number with leading zeros kept (`3,05` → `üç virgül sıfır beş`).
- `%` before or after a number becomes `yüzde` (`%47`, `47%` → `yüzde kırk yedi`).
- Apostrophe suffixes stay attached, so `1996'da` ends up as `... doksan altıda`, which is
  how such words are written out.
- Not handled: ordinals (`3.` → `üçüncü` is ambiguous with a sentence-final period), times
  are read as two numbers (`14:30` → `on dört otuz`), negative numbers and currency symbols.
- `num2words` (0.5.14) was evaluated and rejected: its Turkish output has no spaces
  (`bindokuzyüzdoksanaltı`, `birmilyon`), so it never matches the references.

## Data-driven decisions

The rules were chosen from a scan of the `sentence` column of Common Voice 27.0 `tr`
(train / dev / test):

- **No digits in any reference.** Numbers are written out ("yüzde kırk dört virgül seksen
  dört"). A 100-clip smoke run of zero-shot `whisper-small` on the test split produced digits
  in 17 of 100 hypotheses ("1996", "%44,84"), so number spelling was added (rule 2). Without
  it the zero-shot model would be penalized for formatting, while the fine-tuned model learns
  to write numbers out from the training transcripts, making the comparison unfair.
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
