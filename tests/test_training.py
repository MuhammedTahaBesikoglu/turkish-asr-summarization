"""Tests for the training helpers that run without the ML stack (CI-safe)."""

import copy
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from src.data.common_voice import Clip
from src.training.data import IGNORE_INDEX, pad_labels, sample_clips
from src.training.tracking import backup_db, restore_db, tracking_uri
from src.training.train_whisper import validate_config

SOT = 50258  # <|startoftranscript|>


def _clips(n: int) -> list[Clip]:
    return [Clip(Path(f"{i}.mp3"), f"cümle {i}", "spk", 1.0, "", "", 2, 0) for i in range(n)]


def test_pad_labels_strips_start_token_and_masks_padding() -> None:
    labels = pad_labels([[SOT, 1, 2, 3], [SOT, 4]], SOT)
    assert labels == [[1, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]


def test_pad_labels_keeps_start_token_unless_every_sequence_has_it() -> None:
    assert pad_labels([[SOT, 1], [2, 3]], SOT) == [[SOT, 1], [2, 3]]


def test_sample_clips_is_deterministic_and_order_preserving() -> None:
    clips = _clips(100)
    subset = sample_clips(clips, 10, seed=42)
    assert subset == sample_clips(clips, 10, seed=42)
    assert subset != sample_clips(clips, 10, seed=7)
    assert len(subset) == 10
    positions = [clips.index(c) for c in subset]
    assert positions == sorted(positions)


def test_sample_clips_returns_everything_when_n_is_none_or_large() -> None:
    clips = _clips(5)
    assert sample_clips(clips, None, seed=0) == clips
    assert sample_clips(clips, 50, seed=0) == clips
    with pytest.raises(ValueError):
        sample_clips(clips, 0, seed=0)


BASE_CONFIG = {
    "method": "full",
    "lora": None,
    "data": {"train_split": "train", "eval_split": "dev"},
}


def test_validate_config_accepts_train_and_dev() -> None:
    validate_config(copy.deepcopy(BASE_CONFIG))


@pytest.mark.parametrize("key", ["train_split", "eval_split"])
def test_validate_config_rejects_test_split(key: str) -> None:
    config = copy.deepcopy(BASE_CONFIG)
    config["data"][key] = "test"
    with pytest.raises(ValueError, match="test split"):
        validate_config(config)


def test_validate_config_checks_method_and_lora_section() -> None:
    config = copy.deepcopy(BASE_CONFIG)
    config["method"] = "qlora"
    with pytest.raises(ValueError, match="method"):
        validate_config(config)
    config["method"] = "lora"
    with pytest.raises(ValueError, match="lora"):
        validate_config(config)


def test_tracking_uri_is_absolute_sqlite(tmp_path: Path) -> None:
    uri = tracking_uri(tmp_path / "mlflow.db")
    assert uri.startswith("sqlite:///")
    assert "\\" not in uri


def test_backup_and_restore_db(tmp_path: Path) -> None:
    db = tmp_path / "local" / "mlflow.db"
    db.parent.mkdir()
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE runs (id TEXT)")
        conn.execute("INSERT INTO runs VALUES ('abc')")
        conn.commit()
    backup_dir = tmp_path / "drive"
    backup_db(db, backup_dir)
    assert not db.with_suffix(".backup.tmp").exists()

    db.unlink()
    assert restore_db(db, backup_dir)
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT id FROM runs").fetchall() == [("abc",)]
    # An existing local database is never overwritten by the backup.
    assert not restore_db(db, backup_dir)
    assert not restore_db(tmp_path / "other.db", None)
