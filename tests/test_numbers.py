"""Tests for spelling out numbers in Turkish."""

import pytest

from src.data.normalization import normalize_text
from src.data.numbers import expand_numbers, integer_to_words, number_to_words


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "sıfır"),
        (1, "bir"),
        (6, "altı"),
        (10, "on"),
        (11, "on bir"),
        (21, "yirmi bir"),
        (44, "kırk dört"),
        (90, "doksan"),
        (100, "yüz"),
        (101, "yüz bir"),
        (200, "iki yüz"),
        (999, "dokuz yüz doksan dokuz"),
        (1000, "bin"),
        (1001, "bin bir"),
        (1996, "bin dokuz yüz doksan altı"),
        (2000, "iki bin"),
        (2023, "iki bin yirmi üç"),
        (12500, "on iki bin beş yüz"),
        (100000, "yüz bin"),
        (101000, "yüz bir bin"),
        (1000000, "bir milyon"),
        (1001000, "bir milyon bin"),
        (2000000000, "iki milyar"),
        (3000000000000, "üç trilyon"),
    ],
)
def test_integer_to_words(n: int, expected: str) -> None:
    assert integer_to_words(n) == expected


def test_integer_to_words_huge_numbers_read_digit_by_digit() -> None:
    assert integer_to_words(10**15) == "bir" + " sıfır" * 15


def test_integer_to_words_rejects_negative() -> None:
    with pytest.raises(ValueError):
        integer_to_words(-1)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("44,84", "kırk dört virgül seksen dört"),
        ("3,05", "üç virgül sıfır beş"),
        ("2.5", "iki virgül beş"),
        ("0,0", "sıfır virgül sıfır"),
        ("12.500", "on iki bin beş yüz"),
        ("1.250.000", "bir milyon iki yüz elli bin"),
        ("1.250,5", "bin iki yüz elli virgül beş"),
        ("007", "yedi"),
    ],
)
def test_number_to_words(token: str, expected: str) -> None:
    assert number_to_words(token) == expected


def test_number_to_words_rejects_non_numbers() -> None:
    with pytest.raises(ValueError):
        number_to_words("12a")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Proje 2 milyar dolar.", "Proje iki milyar dolar."),
        ("1996'da başladı", "bin dokuz yüz doksan altı'da başladı"),
        ("%44,84", "yüzde kırk dört virgül seksen dört"),
        ("% 5 arttı", "yüzde beş arttı"),
        ("50% indirim", "yüzde elli indirim"),
        ("saat 14:30", "saat on dört:otuz"),
        ("1990-1995 arası", "bin dokuz yüz doksan-bin dokuz yüz doksan beş arası"),
        ("3G ağı", "üç G ağı"),
        ("rakam yok", "rakam yok"),
    ],
)
def test_expand_numbers(text: str, expected: str) -> None:
    assert expand_numbers(text) == expected


@pytest.mark.parametrize(
    ("hypothesis", "reference"),
    [
        # Real whisper-small outputs vs. Common Voice references
        (
            "Karadziç 1996 yılında kanundan kaçmaya başladı.",
            "Karadziç bin dokuz yüz doksan altı yılında kanundan kaçmaya başladı.",
        ),
        (
            "Asgari fiyat 6 milyon avro olarak belirlendi.",
            "Asgari fiyat altı milyon avro olarak belirlendi.",
        ),
        ("Katılım %47 oldu.", "Katılım yüzde kırk yedi oldu."),
        ("1996'da", "Bin dokuz yüz doksan altıda"),
    ],
)
def test_normalized_digits_match_spelled_out_reference(hypothesis: str, reference: str) -> None:
    assert normalize_text(hypothesis) == normalize_text(reference)


def test_spell_out_numbers_can_be_disabled() -> None:
    assert normalize_text("2023 yılında", spell_out_numbers=False) == "2023 yılında"
