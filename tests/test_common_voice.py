"""Tests for the local Common Voice reader and split statistics (no real audio needed)."""

from pathlib import Path

import pytest

from src.data.common_voice import load_split, missing_audio, read_durations
from src.data.stats import summarize_split, to_markdown

HEADER = (
    "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\tdown_votes"
    "\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
)


def _row(client: str, clip: str, sentence: str, age: str = "", gender: str = "") -> str:
    return f"{client}\t{clip}\tsid\t{sentence}\t\t2\t0\t{age}\t{gender}\t\t\ttr\t\n"


@pytest.fixture
def cv_root(tmp_path: Path) -> Path:
    """A tiny fake Common Voice locale directory."""
    (tmp_path / "clips").mkdir()
    (tmp_path / "train.tsv").write_text(
        HEADER
        + _row("spk1", "a.mp3", 'O "gaipten" gelen emri getirdi.', "twenties", "male_masculine")
        + _row("spk1", "b.mp3", "Bir şeyler yapmalıyız.")
        + _row("spk2", "c.mp3", "Rüzgâr Kartaca'ya sürükledi."),
        encoding="utf-8",
    )
    (tmp_path / "test.tsv").write_text(HEADER, encoding="utf-8")
    (tmp_path / "clip_durations.tsv").write_text(
        "clip\tduration[ms]\na.mp3\t3624\nb.mp3\t31000\n", encoding="utf-8"
    )
    for name in ("a.mp3", "c.mp3"):
        (tmp_path / "clips" / name).write_bytes(b"")
    return tmp_path


def test_load_split_reads_rows(cv_root: Path) -> None:
    clips = load_split(cv_root, "train")
    assert len(clips) == 3
    first = clips[0]
    # Literal double quotes must survive (QUOTE_NONE).
    assert first.sentence == 'O "gaipten" gelen emri getirdi.'
    assert first.audio_path == cv_root / "clips" / "a.mp3"
    assert first.duration_s == pytest.approx(3.624)
    assert (first.age, first.gender, first.up_votes, first.down_votes) == (
        "twenties",
        "male_masculine",
        2,
        0,
    )
    assert clips[2].duration_s is None  # not in clip_durations.tsv


def test_load_split_rejects_unknown_split(cv_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        load_split(cv_root, "validated")


def test_empty_split(cv_root: Path) -> None:
    assert load_split(cv_root, "test") == []
    assert summarize_split([])["clips"] == 0


def test_read_durations(cv_root: Path) -> None:
    assert read_durations(cv_root) == {"a.mp3": 3.624, "b.mp3": 31.0}


def test_missing_audio(cv_root: Path) -> None:
    missing = missing_audio(load_split(cv_root, "train"))
    assert missing == [cv_root / "clips" / "b.mp3"]


def test_summarize_split(cv_root: Path) -> None:
    summary = summarize_split(load_split(cv_root, "train"))
    assert summary["clips"] == 3
    assert summary["speakers"] == 2
    assert summary["top_speaker_share"] == pytest.approx(2 / 3)
    assert summary["clips_without_duration"] == 1
    assert summary["clips_over_30s"] == 1
    assert summary["hours"] == pytest.approx((3.624 + 31.0) / 3600)
    assert summary["words_per_sentence"]["max"] == 5
    assert summary["gender"] == {"unspecified": 2, "male_masculine": 1}


def test_to_markdown(cv_root: Path) -> None:
    summary = {"train": summarize_split(load_split(cv_root, "train"))}
    markdown = to_markdown(summary, "Title")
    assert markdown.startswith("# Title")
    assert "| Speakers | 2 |" in markdown
    assert "| Clips > 30 s | 1 |" in markdown
