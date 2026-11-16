"""Read a locally downloaded Common Voice release (``.tsv`` metadata + ``clips/``).

Only the official ``train`` / ``dev`` / ``test`` splits are exposed. The test split is for
evaluation only and must never be used for training.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SPLITS = ("train", "dev", "test")
DURATIONS_FILE = "clip_durations.tsv"


@dataclass(frozen=True)
class Clip:
    """One recorded sentence."""

    audio_path: Path
    sentence: str
    client_id: str
    duration_s: float | None
    age: str
    gender: str
    up_votes: int
    down_votes: int


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    """Read a Common Voice TSV file.

    Quoting is disabled: sentences contain literal ``"`` characters that must not be parsed
    as CSV quotes.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def read_durations(root: str | Path) -> dict[str, float]:
    """Map clip file name to duration in seconds, from ``clip_durations.tsv``."""
    rows = read_tsv(Path(root) / DURATIONS_FILE)
    return {row["clip"]: int(row["duration[ms]"]) / 1000 for row in rows}


def load_split(root: str | Path, split: str) -> list[Clip]:
    """Load the metadata of one official split.

    Args:
        root: Locale directory of the release (contains ``train.tsv``, ``clips/`` ...).
        split: One of ``"train"``, ``"dev"``, ``"test"``.

    Returns:
        One :class:`Clip` per row, in file order. ``duration_s`` is ``None`` if the clip is
        missing from ``clip_durations.tsv`` (or that file does not exist).
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    root = Path(root)
    durations = read_durations(root) if (root / DURATIONS_FILE).exists() else {}
    clips_dir = root / "clips"
    return [
        Clip(
            audio_path=clips_dir / row["path"],
            sentence=row["sentence"],
            client_id=row["client_id"],
            duration_s=durations.get(row["path"]),
            age=row.get("age", ""),
            gender=row.get("gender", ""),
            up_votes=int(row.get("up_votes") or 0),
            down_votes=int(row.get("down_votes") or 0),
        )
        for row in read_tsv(root / f"{split}.tsv")
    ]


def missing_audio(clips: list[Clip]) -> list[Path]:
    """Return the audio paths that do not exist on disk."""
    return [clip.audio_path for clip in clips if not clip.audio_path.is_file()]


def load_audio(path: str | Path, sampling_rate: int = 16000) -> np.ndarray:
    """Decode an audio file to mono float32, resampled to ``sampling_rate``."""
    import librosa  # heavy import, only needed when audio is actually read

    audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return audio
