"""MLflow experiment tracking with a SQLite store that survives Colab disconnects.

The database lives on local disk (SQLite over the Google Drive mount is prone to locking
problems) and is copied to an optional backup directory, e.g. on Drive, at every checkpoint
and at the end of training. A resumed training run restores the database from the backup and
continues the same MLflow run, whose id is stored in the training output directory.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from src.config import resolve_path

RUN_ID_FILE = "mlflow_run_id.txt"

# MLflow >= 3.x prints an "agent hint" on import; it is noise in training logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def tracking_uri(db_path: Path) -> str:
    """SQLAlchemy URI of a SQLite file (absolute, forward slashes, also on Windows)."""
    return f"sqlite:///{db_path.resolve().as_posix()}"


def backup_db(db_path: Path, backup_dir: Path) -> None:
    """Copy the SQLite database to ``backup_dir`` with SQLite's online backup API.

    The backup is written to a local temporary file first and then copied, so the Drive
    mount only sees one plain file write.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".backup.tmp")
    # closing(): sqlite3's own context manager only commits; an open handle would keep the
    # file locked on Windows.
    with closing(sqlite3.connect(db_path)) as src, closing(sqlite3.connect(tmp)) as dst:
        src.backup(dst)
    shutil.copyfile(tmp, backup_dir / db_path.name)
    tmp.unlink()


def restore_db(db_path: Path, backup_dir: Path | None) -> bool:
    """Copy the backup into place if there is no local database yet; return whether it did."""
    if backup_dir is None or db_path.exists():
        return False
    backup = backup_dir / db_path.name
    if not backup.is_file():
        return False
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backup, db_path)
    return True


def start_run(
    tracking_cfg: dict[str, Any], output_dir: Path, resume: bool, backup_dir: Path | None
) -> str:
    """Start (or, when resuming, reopen) the MLflow run of a training job; return its id.

    Also sets ``MLFLOW_TRACKING_URI`` so the ``transformers`` MLflow callback logs into the
    same store and attaches to this active run.
    """
    import mlflow

    db_path = resolve_path(tracking_cfg["db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if restore_db(db_path, backup_dir):
        print(f"Restored MLflow database from {backup_dir}")
    uri = tracking_uri(db_path)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(tracking_cfg["experiment"])

    run_id_file = output_dir / RUN_ID_FILE
    if resume and run_id_file.is_file():
        run = mlflow.start_run(run_id=run_id_file.read_text(encoding="utf-8").strip())
    else:
        run = mlflow.start_run(run_name=tracking_cfg["run_name"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(run.info.run_id, encoding="utf-8")
    return run.info.run_id


def flush() -> None:
    """Wait for asynchronously logged metrics to reach the database."""
    import mlflow

    mlflow.flush_async_logging()
