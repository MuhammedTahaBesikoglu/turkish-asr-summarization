"""Unit tests for Turkish text normalization."""

import pytest

from src.data.normalization import normalize_text, turkish_lower


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IŞIK", "ışık"),
        ("Iğdır", "ığdır"),
        ("İSTANBUL", "istanbul"),
        ("İzmir", "izmir"),
        ("ÇĞÖŞÜ", "çğöşü"),
        ("İstanbul", "istanbul"),  # decomposed İ (I + combining dot above)
        ("already lower", "already lower"),
    ],
)
def test_turkish_lower(text: str, expected: str) -> None:
    assert turkish_lower(text) == expected


def test_turkish_lower_differs_from_builtin() -> None:
    # Guard against regressions to str.lower(), which gives "i" and "i̇" here.
    assert "IŞIK".lower() != turkish_lower("IŞIK")
    assert "İ".lower() != turkish_lower("İ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Punctuation and casing (real Common Voice sentences)
        (
            "Macide biraz hızlandı, fakat kalabalıkta onları kaybeder gibi oldu.",
            "macide biraz hızlandı fakat kalabalıkta onları kaybeder gibi oldu",
        ),
        ("Neden izin verildi?", "neden izin verildi"),
        ("Bekle… Gel!", "bekle gel"),
        ('"Nefs-i Sadak" köyü', "nefs i sadak köyü"),
        ("İlk: bir; iki", "ilk bir iki"),
        ("(parantez) — tire", "parantez tire"),
        # Apostrophes of every style are removed, keeping the word whole
        ("Ankara'ya", "ankaraya"),
        ("Beyazıt’a", "beyazıta"),
        ("Kartaca‘ya", "kartacaya"),
        # Hyphens split compounds
        ("Renault-Nissan-Mitsubishi", "renault nissan mitsubishi"),
        ("Basel- Bern", "basel bern"),
        # Circumflex folding
        ("Rüzgâr hâlâ esiyor", "rüzgar hala esiyor"),
        ("hükûmet", "hükumet"),
        ("İslâmî", "islami"),
        # Replacement character / symbols from broken references
        ("fonların yaklaşık �'i", "fonların yaklaşık i"),
        ("şirketin %'ini", "şirketin ini"),
        # Digits are spelled out (details in test_numbers.py)
        ("2023 yılında 5 kişi", "iki bin yirmi üç yılında beş kişi"),
        ("1996'da", "bin dokuz yüz doksan altıda"),
        # Non-Turkish Latin letters are kept
        ("Charité hastanesi", "charité hastanesi"),
        # Whitespace
        ("  çok   boşluk\t\nvar  ", "çok boşluk var"),
        ("", ""),
        ("...!?", ""),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


def test_circumflex_folding_can_be_disabled() -> None:
    assert normalize_text("Rüzgâr hâlâ", fold_circumflex=False) == "rüzgâr hâlâ"


@pytest.mark.parametrize(
    "text",
    ["İSTANBUL'a gittim.", "Rüzgâr, hâlâ esiyor!", "IĞDIR-Kars", ""],
)
def test_normalize_text_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_reference_and_hypothesis_variants_match() -> None:
    # Typical surface differences between a reference and a Whisper output.
    reference = "Öte yandan Kosova'da da hâlâ pek çok akrabam var."
    hypothesis = "öte yandan kosovada da hala pek çok akrabam var"
    assert normalize_text(reference) == normalize_text(hypothesis)
