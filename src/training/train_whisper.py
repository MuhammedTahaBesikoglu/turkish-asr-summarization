"""Fine-tune Whisper on Common Voice Turkish: full fine-tune or LoRA, chosen in the config.

Usage:
    python -m src.training.train_whisper --config configs/train_full.yaml
    python -m src.training.train_whisper --config configs/train_lora.yaml \\
        --data-config configs/data_colab.yaml --tracking-backup /content/drive/MyDrive/x/mlflow
    python -m src.training.train_whisper --config configs/train_lora.yaml --resume
    python -m src.training.train_whisper --config configs/train_lora.yaml \\
        --set training.max_steps=10 --set data.eval_samples=8      # smoke run

Only the ``train`` split is trained on; the best checkpoint is selected by WER on a fixed
subset of ``dev``. The ``test`` split is never loaded here.

Outputs:
    <training.output_dir>/checkpoint-*   Trainer checkpoints (for --resume)
    <training.output_dir>/final          best model (LoRA: merged into the base weights) and
                                         processor, loadable by src.evaluation.evaluate_asr
    <training.output_dir>/adapter        LoRA only: the adapter weights alone
    <training.output_dir>/train_summary.json   copy of the summary; marks the run as finished
    <output.results_dir>/train_summary.json
        time, peak GPU memory, parameter counts, dev metrics, log history, versions, config
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import apply_overrides, load_config, resolve_path
from src.data.common_voice import Clip, load_split
from src.evaluation.metrics import compute_metrics
from src.run_info import git_commit, library_versions
from src.training import tracking

METHODS = ("full", "lora")
SUMMARY_FILE = "train_summary.json"
_VERSIONED = ["torch", "transformers", "accelerate", "peft", "mlflow", "jiwer", "librosa"]


def validate_config(config: dict[str, Any]) -> None:
    """Fail early on configs that would train or select models on the test split."""
    if config["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {config['method']!r}")
    if config["method"] == "lora" and not config.get("lora"):
        raise ValueError("method 'lora' needs a 'lora' section")
    data = config["data"]
    for key in ("train_split", "eval_split"):
        if data[key] == "test":
            raise ValueError(f"data.{key} is 'test'; the test split is for final evaluation only")
    if data["train_split"] == data["eval_split"]:
        raise ValueError("data.train_split and data.eval_split must differ")


def count_parameters(model: Any) -> dict[str, int]:
    """Trainable and total parameter counts (shared tensors counted once)."""
    params = list(model.parameters())
    return {
        "trainable": sum(p.numel() for p in params if p.requires_grad),
        "total": sum(p.numel() for p in params),
    }


def build_model(config: dict[str, Any]) -> Any:
    """Load the pretrained model, set Turkish transcription and wrap it with LoRA if asked."""
    from transformers import WhisperForConditionalGeneration

    model_cfg = config["model"]
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["name"])
    # Generation during evaluation (and in the saved model) always transcribes Turkish.
    model.generation_config.language = model_cfg["language"]
    model.generation_config.task = model_cfg["task"]
    model.generation_config.forced_decoder_ids = None
    if config["method"] == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(**config["lora"]))
    return model


def make_compute_metrics(tokenizer: Any, eval_clips: list[Clip]):
    """Score generated token ids against the original dev sentences (same metrics as test)."""
    import numpy as np

    references = [clip.sentence for clip in eval_clips]

    def compute(pred: Any) -> dict[str, float]:
        ids = np.where(pred.predictions == -100, tokenizer.pad_token_id, pred.predictions)
        hypotheses = tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        metrics = compute_metrics(references, hypotheses)
        return {
            "wer": metrics["normalized"]["wer"],
            "cer": metrics["normalized"]["cer"],
            "wer_raw": metrics["raw"]["wer"],
            "cer_raw": metrics["raw"]["cer"],
        }

    return compute


def find_checkpoint(output_dir: Path, resume: bool) -> Path | None:
    """Return the checkpoint to resume from; refuse to silently overwrite an existing run."""
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if last and not resume:
        raise FileExistsError(
            f"{output_dir} already has checkpoints; pass --resume to continue that run or "
            "choose another training.output_dir"
        )
    return Path(last) if last else None


def save_final(trainer: Any, processor: Any, config: dict[str, Any], output_dir: Path) -> Path:
    """Save the best model as a plain Whisper model (LoRA merged) plus the processor."""
    final_dir = output_dir / "final"
    model = trainer.model
    if config["method"] == "lora":
        model.save_pretrained(output_dir / "adapter")
        model = model.merge_and_unload()
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper (full or LoRA).")
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--data-config", help="Override the config's data_config")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint")
    parser.add_argument("--tracking-backup", help="Directory to back up the MLflow database to")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value"
    )
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    if args.data_config:
        config["data_config"] = args.data_config
    validate_config(config)

    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor

    from src.training.callbacks import MlflowBackup, ResourceTracker
    from src.training.data import SpeechDataset, WhisperCollator, sample_clips

    data_config = load_config(resolve_path(config["data_config"]))
    model_cfg, data_cfg, tracking_cfg = config["model"], config["data"], config.get("tracking")
    train_cfg = dict(config["training"])
    output_dir = resolve_path(train_cfg.pop("output_dir"))
    if (output_dir / SUMMARY_FILE).is_file():
        # After training the Trainer deletes the last checkpoint when it is not the best one,
        # so "resuming" a finished run would silently train again from the best checkpoint.
        print(f"{output_dir} already holds a finished run ({SUMMARY_FILE}); nothing to do.")
        return
    checkpoint = find_checkpoint(output_dir, args.resume)
    backup_dir = resolve_path(args.tracking_backup) if args.tracking_backup else None

    root = resolve_path(data_config["dataset"]["root"])
    sampling_rate = data_config["audio"]["sampling_rate"]
    train_clips = sample_clips(
        load_split(root, data_cfg["train_split"]), data_cfg["max_train_samples"], data_cfg["seed"]
    )
    eval_clips = sample_clips(
        load_split(root, data_cfg["eval_split"]), data_cfg["eval_samples"], data_cfg["seed"]
    )
    print(
        f"{config['method']} fine-tune of {model_cfg['name']}: {len(train_clips)} "
        f"{data_cfg['train_split']} clips, {len(eval_clips)} {data_cfg['eval_split']} clips "
        f"for evaluation" + (f", resuming from {checkpoint.name}" if checkpoint else "")
    )

    processor = WhisperProcessor.from_pretrained(
        model_cfg["name"], language=model_cfg["language"], task=model_cfg["task"]
    )
    model = build_model(config)
    params = count_parameters(model)
    print(f"Trainable parameters: {params['trainable']:,} / {params['total']:,}")

    callbacks: list[Any] = []
    if tracking_cfg:
        run_id = tracking.start_run(tracking_cfg, output_dir, checkpoint is not None, backup_dir)
        print(f"MLflow run {run_id} in {tracking.tracking_uri(resolve_path(tracking_cfg['db']))}")
        if backup_dir:
            callbacks.append(MlflowBackup(resolve_path(tracking_cfg["db"]), backup_dir))
    usage = ResourceTracker(resume_from=checkpoint)
    callbacks.append(usage)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        run_name=tracking_cfg["run_name"] if tracking_cfg else None,
        report_to=["mlflow"] if tracking_cfg else "none",
        predict_with_generate=True,
        remove_unused_columns=False,  # the datasets yield exactly the model inputs
        label_names=["labels"],
        dataloader_num_workers=data_cfg["num_workers"],
        fp16=train_cfg.pop("fp16") and torch.cuda.is_available(),
        **train_cfg,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=SpeechDataset(train_clips, processor, sampling_rate),
        eval_dataset=SpeechDataset(eval_clips, processor, sampling_rate),
        data_collator=WhisperCollator(model.config.decoder_start_token_id),
        processing_class=processor,
        compute_metrics=make_compute_metrics(processor.tokenizer, eval_clips),
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    resources = usage.current()
    # With load_best_model_at_end the trainer now holds the best checkpoint.
    dev_metrics = trainer.evaluate(metric_key_prefix="best_eval")
    final_dir = save_final(trainer, processor, config, output_dir)

    summary = {
        "model": model_cfg["name"],
        "method": config["method"],
        "parameters": params,
        "trainable_fraction": round(params["trainable"] / params["total"], 6),
        "train_clips": len(train_clips),
        "eval_clips": len(eval_clips),
        "global_step": trainer.state.global_step,
        "epochs": round(trainer.state.epoch or 0.0, 3),
        "train_loss": train_result.training_loss,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_metrics": dev_metrics,
        "resources": resources,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_model_dir": str(final_dir),
        "log_history": trainer.state.log_history,
        "versions": library_versions(_VERSIONED),
        "git_commit": git_commit(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
    }
    results_dir = resolve_path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    # Also next to the model (on Drive in Colab), where it marks the run as finished.
    for directory in (results_dir, output_dir):
        (directory / SUMMARY_FILE).write_text(summary_json, encoding="utf-8")

    if tracking_cfg:
        import mlflow

        # Dev metrics of the best model are already logged by trainer.evaluate (best_eval_*).
        summary_metrics = {
            "train_seconds": resources["train_seconds"],
            "wall_seconds": resources["wall_seconds"],
            "peak_gpu_memory_gib": resources["peak_gpu_memory_gib"],
            "trainable_params": params["trainable"],
            "total_params": params["total"],
        }
        mlflow.log_metrics({f"summary/{k}": v for k, v in summary_metrics.items() if v is not None})
        mlflow.end_run()
        if backup_dir:
            tracking.backup_db(resolve_path(tracking_cfg["db"]), backup_dir)

    print(
        f"Best dev WER {dev_metrics['best_eval_wer']:.2%} (CER {dev_metrics['best_eval_cer']:.2%}) "
        f"on {len(eval_clips)} clips; train time {resources['train_seconds'] / 3600:.2f} h, "
        f"peak GPU memory {resources['peak_gpu_memory_gib']} GiB"
    )
    print(f"Saved {final_dir} and {results_dir / SUMMARY_FILE}")


if __name__ == "__main__":
    main()
