"""Tests for WER/CER computation."""

import math

import pytest

from src.evaluation.metrics import compute_metrics, error_rates, raw_text, utterance_wer


def test_raw_text_only_touches_whitespace() -> None:
    assert raw_text("  Ankara'ya,   gitti. ") == "Ankara'ya, gitti."


def test_error_rates_counts_edits() -> None:
    result = error_rates(["a b c", "d e"], ["a x c", ""])
    assert result["substitutions"] == 1
    assert result["deletions"] == 2
    assert result["insertions"] == 0
    assert result["reference_words"] == 5
    assert result["wer"] == pytest.approx(3 / 5)


def test_error_rates_is_corpus_level_not_mean_of_utterances() -> None:
    # 1 error in 1 word + 0 errors in 4 words -> 1/5, not mean(1.0, 0.0) = 0.5
    result = error_rates(["a", "b c d e"], ["x", "b c d e"])
    assert result["wer"] == pytest.approx(1 / 5)


def test_error_rates_validates_input() -> None:
    with pytest.raises(ValueError, match="references vs"):
        error_rates(["a"], [])
    with pytest.raises(ValueError, match="No references"):
        error_rates([], [])


def test_normalization_removes_surface_errors() -> None:
    references = ["Öte yandan Kosova'da da hâlâ pek çok akrabam var."]
    hypotheses = [" öte yandan Kosovada da hala pek çok akrabam var"]
    result = compute_metrics(references, hypotheses)
    assert result["normalized"]["wer"] == 0.0
    assert result["normalized"]["cer"] == 0.0
    # Raw text keeps casing, apostrophe, circumflex and punctuation differences.
    assert result["raw"]["wer"] > 0.0


def test_turkish_casing_is_not_an_error() -> None:
    result = compute_metrics(["IŞIK İSTANBUL"], ["ışık istanbul"])
    assert result["normalized"]["wer"] == 0.0


def test_utterance_wer() -> None:
    assert utterance_wer("Bir iki üç dört.", "bir iki üç beş") == pytest.approx(0.25)
    assert utterance_wer("...", "") == 0.0
    assert math.isinf(utterance_wer("...", "bir"))
