"""WER/CER on normalized and raw text.

Every evaluation in the project (zero-shot baseline, full fine-tune, LoRA) goes through
:func:`compute_metrics`, so all reported numbers use the same normalization
(:func:`src.data.normalization.normalize_text`) and the same corpus-level aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.data.normalization import normalize_text


def raw_text(text: str) -> str:
    """Minimal cleanup for "raw" scores: only strip and collapse whitespace."""
    return " ".join(text.split())


def error_rates(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, Any]:
    """Corpus-level WER and CER (total edits / total reference units) with edit counts."""
    import jiwer  # imported lazily so the package imports without the ML stack

    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references vs {len(hypotheses)} hypotheses")
    if not references:
        raise ValueError("No references to score")
    words = jiwer.process_words(list(references), list(hypotheses))
    chars = jiwer.process_characters(list(references), list(hypotheses))
    return {
        "wer": words.wer,
        "cer": chars.cer,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference_words": words.hits + words.substitutions + words.deletions,
    }


def compute_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, Any]:
    """WER/CER on Turkish-normalized text (``normalized``) and on raw text (``raw``)."""
    return {
        "normalized": error_rates(
            [normalize_text(r) for r in references], [normalize_text(h) for h in hypotheses]
        ),
        "raw": error_rates([raw_text(r) for r in references], [raw_text(h) for h in hypotheses]),
    }


def utterance_wer(reference: str, hypothesis: str) -> float:
    """WER of a single normalized pair, for per-clip error analysis."""
    import jiwer

    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return jiwer.wer(reference, hypothesis)
