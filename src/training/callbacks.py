"""Trainer callbacks: resource accounting across resumed sessions, tracking backups."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from src.training import tracking

USAGE_FILE = "resource_usage.json"


class ResourceTracker(TrainerCallback):
    """Measure training time and peak GPU memory, summed over Colab sessions.

    The totals are written into every checkpoint directory, so a resumed run continues from
    the values of the checkpoint it resumes from; work lost after the last checkpoint is not
    counted, because it is redone. ``train_seconds`` is wall time minus evaluation time
    (``eval_runtime``), so it is comparable between methods whose evaluation speed differs.
    """

    def __init__(self, resume_from: Path | None = None) -> None:
        self.totals: dict[str, Any] = {
            "wall_seconds": 0.0,
            "eval_seconds": 0.0,
            "peak_gpu_memory_gib": None,
            "sessions": 0,
        }
        if resume_from is not None and (resume_from / USAGE_FILE).is_file():
            self.totals.update(json.loads((resume_from / USAGE_FILE).read_text(encoding="utf-8")))
        self._start = 0.0
        self._eval_seconds = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.totals["sessions"] += 1
        self._start = time.perf_counter()
        self._eval_seconds = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_runtime" in metrics:
            self._eval_seconds += metrics["eval_runtime"]

    def current(self) -> dict[str, Any]:
        """Totals including the current session so far."""
        import torch

        peak = self.totals["peak_gpu_memory_gib"]
        if torch.cuda.is_available():
            session_peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            peak = session_peak if peak is None else max(peak, session_peak)
        wall = self.totals["wall_seconds"] + time.perf_counter() - self._start
        evals = self.totals["eval_seconds"] + self._eval_seconds
        return {
            "wall_seconds": round(wall, 1),
            "eval_seconds": round(evals, 1),
            "train_seconds": round(wall - evals, 1),
            "peak_gpu_memory_gib": peak,
            "sessions": self.totals["sessions"],
        }

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            usage = self.current()
            usage.pop("train_seconds")
            (checkpoint / USAGE_FILE).write_text(json.dumps(usage, indent=2), encoding="utf-8")


class MlflowBackup(TrainerCallback):
    """Copy the MLflow SQLite database to a backup directory after every checkpoint."""

    def __init__(self, db_path: Path, backup_dir: Path) -> None:
        self.db_path = db_path
        self.backup_dir = backup_dir

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            tracking.flush()
            tracking.backup_db(self.db_path, self.backup_dir)
