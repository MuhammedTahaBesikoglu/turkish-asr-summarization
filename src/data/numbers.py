"""Spell out digits as Turkish words ("1996" -> "bin dokuz yüz doksan altı").

Common Voice references write every number in words, while Whisper often outputs digits.
:func:`expand_numbers` is part of :func:`src.data.normalization.normalize_text` so that
"1996" and "bin dokuz yüz doksan altı" are scored as the same words.

Off-the-shelf converters were not used: ``num2words`` (0.5.14) joins Turkish number words
without spaces ("bindokuzyüzdoksanaltı"), which never matches the references.
"""

from __future__ import annotations

import re

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
# Index i is the word for 1000**i.
_SCALES = ["", "bin", "milyon", "milyar", "trilyon"]
_MAX = 1000 ** len(_SCALES) - 1

# Thousands separated by dots (Turkish style "12.500"), or plain digits, with an optional
# decimal part after "," or "." ("44,84", "2.5"). A dot followed by exactly three digits is a
# thousands separator, otherwise it is read as a decimal point.
_NUMBER = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
_PERCENT_RE = re.compile(rf"%\s?({_NUMBER})|({_NUMBER})\s?%")
_NUMBER_RE = re.compile(_NUMBER)


def _below_thousand(n: int) -> list[str]:
    hundreds, rest = divmod(n, 100)
    words = []
    if hundreds:
        # "yüz", not "bir yüz"
        words += ([_ONES[hundreds]] if hundreds > 1 else []) + ["yüz"]
    words += [w for w in (_TENS[rest // 10], _ONES[rest % 10]) if w]
    return words


def integer_to_words(n: int) -> str:
    """Turkish words for a non-negative integer (up to 999 trilyon).

    Larger numbers are read digit by digit.
    """
    if n < 0:
        raise ValueError("Only non-negative integers are supported")
    if n == 0:
        return "sıfır"
    if n > _MAX:
        return " ".join(integer_to_words(int(d)) for d in str(n))
    words: list[str] = []
    for scale in range(len(_SCALES) - 1, -1, -1):
        group = (n // 1000**scale) % 1000
        if not group:
            continue
        if scale == 1 and group == 1:
            words.append("bin")  # "bin", not "bir bin"; but "bir milyon"
            continue
        words += _below_thousand(group)
        if scale:
            words.append(_SCALES[scale])
    return " ".join(words)


def number_to_words(token: str) -> str:
    """Turkish words for a number written with digits ("12.500", "44,84", "3,05")."""
    match = re.fullmatch(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:[.,](\d+))?", token)
    if not match:
        raise ValueError(f"Not a number: {token!r}")
    integer, fraction = match.groups()
    words = integer_to_words(int(integer.replace(".", "")))
    if fraction:
        # Read as a whole number after "virgül", keeping leading zeros: 3,05 -> üç virgül sıfır beş
        zeros = len(fraction) - len(fraction.lstrip("0"))
        tail = fraction.lstrip("0")
        parts = ["sıfır"] * zeros + ([integer_to_words(int(tail))] if tail else [])
        words += " virgül " + " ".join(parts)
    return words


def expand_numbers(text: str) -> str:
    """Replace every number in ``text`` with Turkish words; ``%`` becomes ``yüzde``.

    Suffixes attached with an apostrophe are kept ("1996'da" -> "bin dokuz yüz doksan
    altı'da"); the apostrophe is removed later by the normalizer.
    """
    text = _PERCENT_RE.sub(lambda m: f"yüzde {number_to_words(m.group(1) or m.group(2))}", text)

    def replace(match: re.Match[str]) -> str:
        words = number_to_words(match.group(0))
        # Separate numbers glued directly to letters ("3G", "x2") so they don't merge into
        # one word; apostrophe suffixes ("1996'da") are left attached.
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        return (" " if before.isalpha() else "") + words + (" " if after.isalpha() else "")

    return _NUMBER_RE.sub(replace, text)
