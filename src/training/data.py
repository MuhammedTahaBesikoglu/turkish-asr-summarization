"""Training data for Whisper: on-the-fly features and labels, batch collation.

Audio is decoded in DataLoader workers (as in the evaluation script), so no preprocessed copy
of the dataset is needed on disk. Labels are the raw Common Voice sentences (casing and
punctuation kept); normalization is applied only when scoring.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from src.data.common_voice import Clip, load_audio

if TYPE_CHECKING:
    import torch

IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss


def sample_clips(clips: Sequence[Clip], n: int | None, seed: int) -> list[Clip]:
    """Return a fixed random subset of ``n`` clips in their original order (all if ``n`` is
    ``None`` or not smaller than ``len(clips)``)."""
    if n is None or n >= len(clips):
        return list(clips)
    if n <= 0:
        raise ValueError(f"Subset size must be positive, got {n}")
    keep = set(random.Random(seed).sample(range(len(clips)), n))
    return [clip for i, clip in enumerate(clips) if i in keep]


def pad_labels(label_ids: Sequence[Sequence[int]], decoder_start_token_id: int) -> list[list[int]]:
    """Prepare tokenized sentences as decoder labels for a batch.

    The tokenizer prepends ``<|startoftranscript|>``, but the model adds the decoder start
    token itself when it shifts the labels right, so a leading start token is removed (when
    every sequence has one, as in the Hugging Face fine-tuning recipe). Sequences are then
    right-padded with :data:`IGNORE_INDEX` so padding does not count in the loss.
    """
    if label_ids and all(seq and seq[0] == decoder_start_token_id for seq in label_ids):
        label_ids = [seq[1:] for seq in label_ids]
    width = max((len(seq) for seq in label_ids), default=0)
    return [list(seq) + [IGNORE_INDEX] * (width - len(seq)) for seq in label_ids]


class SpeechDataset:
    """Map-style dataset yielding Whisper log-mel features and label token ids per clip.

    Defined at module level so spawned DataLoader workers (Windows) can pickle it.
    """

    def __init__(self, clips: Sequence[Clip], processor: Any, sampling_rate: int) -> None:
        self.clips = list(clips)
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[index]
        audio = load_audio(clip.audio_path, sampling_rate=self.sampling_rate)
        features = self.processor.feature_extractor(
            audio, sampling_rate=self.sampling_rate
        ).input_features[0]
        labels = self.processor.tokenizer(clip.sentence).input_ids
        return {"input_features": features, "labels": labels}


class WhisperCollator:
    """Stack log-mel features (all padded to 30 s) and pad labels into a training batch."""

    def __init__(self, decoder_start_token_id: int) -> None:
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        import numpy as np
        import torch

        features = np.stack([ex["input_features"] for ex in examples])
        labels = pad_labels([ex["labels"] for ex in examples], self.decoder_start_token_id)
        return {
            "input_features": torch.from_numpy(features),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
