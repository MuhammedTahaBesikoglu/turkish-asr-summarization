"""Turkish text normalization shared by every WER/CER evaluation.

The same :func:`normalize_text` must be applied to references and hypotheses in both the
baseline and the fine-tuned evaluations, otherwise the numbers are not comparable.
See ``docs/normalization.md`` for the rationale behind each rule.
"""

from __future__ import annotations

import re
import unicodedata

from src.data.numbers import expand_numbers

# Python's str.lower() maps "I" -> "i" and "İ" -> "i̇" (i + U+0307 combining dot), both wrong
# for Turkish. Map the dotted/dotless pair explicitly before lowercasing the rest.
_TURKISH_LOWER = str.maketrans({"I": "ı", "İ": "i"})

# Circumflex is used inconsistently in Common Voice (e.g. both "hükûmet" and "hükümet").
_CIRCUMFLEX = str.maketrans({"â": "a", "î": "i", "û": "u"})

# Apostrophes separate suffixes from proper nouns ("Ankara'ya"); they are removed so the word
# stays a single token whether or not the model writes the apostrophe.
_APOSTROPHES = "'’‘`´ʼ"
_APOSTROPHE_RE = re.compile(f"[{re.escape(_APOSTROPHES)}]")

_WHITESPACE_RE = re.compile(r"\s+")


def turkish_lower(text: str) -> str:
    """Lowercase ``text`` using Turkish casing rules (I -> ı, İ -> i)."""
    # NFC first so a decomposed "I" + U+0307 becomes "İ" and is handled by the table.
    return unicodedata.normalize("NFC", text).translate(_TURKISH_LOWER).lower()


def normalize_text(
    text: str, *, fold_circumflex: bool = True, spell_out_numbers: bool = True
) -> str:
    """Normalize Turkish text for WER/CER computation.

    Steps: Unicode NFC, numbers spelled out in Turkish ("%44,84" -> "yüzde kırk dört virgül
    seksen dört"), Turkish lowercasing, optional circumflex folding (â/î/û -> a/i/u),
    apostrophe removal, every other non-alphanumeric character replaced by a space,
    whitespace collapsed.

    Args:
        text: Raw reference or model output.
        fold_circumflex: Map â, î, û to a, i, u.
        spell_out_numbers: Replace digits with Turkish number words.

    Returns:
        The normalized string (possibly empty).
    """
    text = unicodedata.normalize("NFC", text)
    if spell_out_numbers:
        text = expand_numbers(text)
    text = turkish_lower(text)
    if fold_circumflex:
        text = text.translate(_CIRCUMFLEX)
    text = _APOSTROPHE_RE.sub("", text)
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return _WHITESPACE_RE.sub(" ", text).strip()
