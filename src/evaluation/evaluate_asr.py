"""Transcribe a Common Voice split with a Whisper model and score it.

Used for the zero-shot baseline and, later, for the fine-tuned models.

Usage:
    python -m src.evaluation.evaluate_asr --config configs/eval_baseline.yaml
    python -m src.evaluation.evaluate_asr --config configs/eval_baseline.yaml \\
        --data-config configs/data_colab.yaml --max-samples 100

Outputs (in ``output.dir``):
    predictions.csv  one row per clip: reference, hypothesis, normalized texts, utterance WER
    metrics.json     WER/CER (normalized and raw), run time, hardware, versions, config
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_path
from src.data.common_voice import Clip, load_audio, load_split
from src.data.normalization import normalize_text
from src.evaluation.metrics import compute_metrics, utterance_wer
from src.run_info import git_commit, library_versions

_DIGIT_RE = re.compile(r"\d")
_VERSIONED = ["torch", "transformers", "jiwer", "librosa", "soundfile"]


def model_source(name: str) -> str:
    """Resolve a local model directory against the repository root; keep Hub ids as given."""
    path = resolve_path(name)
    return str(path) if path.is_dir() else name


class AudioDataset:
    """Map-style dataset that decodes clips on the fly (in DataLoader workers).

    Defined at module level so it can be pickled by spawned workers (Windows).
    """

    def __init__(self, clips: list[Clip], sampling_rate: int) -> None:
        self.clips = clips
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        return load_audio(self.clips[index].audio_path, sampling_rate=self.sampling_rate)


class FeatureCollator:
    """Turn a batch of waveforms into Whisper log-mel input features (padded to 30 s)."""

    def __init__(self, feature_extractor: Any, sampling_rate: int) -> None:
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, batch: list[Any]):
        return self.feature_extractor(
            batch, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features


def transcribe(
    clips: list[Clip], model_cfg: dict[str, Any], eval_cfg: dict[str, Any], sampling_rate: int
) -> tuple[list[str], dict[str, Any]]:
    """Run Whisper on ``clips`` in batches; return hypotheses and runtime info."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if eval_cfg["fp16"] and device == "cuda" else torch.float32

    name = model_source(model_cfg["name"])
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=dtype)
    model.to(device).eval()

    loader = DataLoader(
        AudioDataset(clips, sampling_rate),
        batch_size=eval_cfg["batch_size"],
        num_workers=eval_cfg["num_workers"],
        collate_fn=FeatureCollator(processor.feature_extractor, sampling_rate),
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hypotheses: list[str] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for features in tqdm(loader, desc="transcribe", unit="batch"):
            generated = model.generate(
                features.to(device, dtype=dtype),
                language=model_cfg["language"],
                task=model_cfg["task"],
                num_beams=eval_cfg["num_beams"],
                # max_length (not max_new_tokens): the model's generation config already sets
                # max_length, and passing both triggers a warning on every batch.
                max_length=eval_cfg["max_length"],
            )
            hypotheses += processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    seconds = time.perf_counter() - start

    runtime = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(dtype).removeprefix("torch."),
        "seconds": round(seconds, 1),
        "clips_per_second": round(len(clips) / seconds, 2),
        "peak_gpu_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None
        ),
    }
    return hypotheses, runtime


def write_predictions(path: Path, clips: list[Clip], hypotheses: list[str]) -> None:
    """Write one CSV row per clip for later error analysis."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip",
                "client_id",
                "duration_s",
                "reference",
                "hypothesis",
                "reference_normalized",
                "hypothesis_normalized",
                "wer",
            ]
        )
        for clip, hyp in zip(clips, hypotheses, strict=True):
            writer.writerow(
                [
                    clip.audio_path.name,
                    clip.client_id,
                    clip.duration_s,
                    clip.sentence,
                    hyp.strip(),
                    normalize_text(clip.sentence),
                    normalize_text(hyp),
                    round(utterance_wer(clip.sentence, hyp), 4),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper model on Common Voice.")
    parser.add_argument("--config", required=True, help="Evaluation YAML config")
    parser.add_argument("--data-config", help="Override the config's data_config")
    parser.add_argument("--max-samples", type=int, help="Override eval.max_samples")
    parser.add_argument("--output-dir", help="Override output.dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_config:
        config["data_config"] = args.data_config
    if args.max_samples is not None:
        config["eval"]["max_samples"] = args.max_samples
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    data_config = load_config(resolve_path(config["data_config"]))
    eval_cfg, model_cfg = config["eval"], config["model"]

    clips = load_split(resolve_path(data_config["dataset"]["root"]), eval_cfg["split"])
    if eval_cfg["max_samples"]:
        clips = clips[: eval_cfg["max_samples"]]
    print(f"{model_cfg['name']} on {eval_cfg['split']}: {len(clips)} clips")

    hypotheses, runtime = transcribe(
        clips, model_cfg, eval_cfg, data_config["audio"]["sampling_rate"]
    )
    references = [clip.sentence for clip in clips]
    metrics = compute_metrics(references, hypotheses)

    out_dir = resolve_path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "predictions.csv", clips, hypotheses)
    dataset = data_config["dataset"]
    report = {
        "model": model_cfg["name"],
        "dataset": f"{dataset['name']} {dataset['version']} {dataset['locale']}",
        "split": eval_cfg["split"],
        "num_clips": len(clips),
        "is_full_split": not eval_cfg["max_samples"],
        "metrics": metrics,
        "hypotheses_with_digits": sum(bool(_DIGIT_RE.search(h)) for h in hypotheses),
        "empty_hypotheses": sum(not h.strip() for h in hypotheses),
        "runtime": runtime,
        "versions": library_versions(_VERSIONED),
        "git_commit": git_commit(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    norm, raw = metrics["normalized"], metrics["raw"]
    print(
        f"WER {norm['wer']:.2%}  CER {norm['cer']:.2%}  (raw WER {raw['wer']:.2%}, "
        f"raw CER {raw['cer']:.2%})"
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
