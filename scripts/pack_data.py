"""Pack selected Common Voice splits into one .tar for upload to Google Drive / Colab.

Only the TSVs and the clips referenced by the chosen splits are included, so the archive is
much smaller than the full release. Archive paths keep the release layout
(``cv-corpus-27.0-2026-09-11/tr/...``), so extracting it into ``/content/data`` matches
``configs/data_colab.yaml``.

Usage:
    python scripts/pack_data.py --splits test               # baseline
    python scripts/pack_data.py --splits train dev test     # fine-tuning
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from src.config import load_config, resolve_path
from src.data.common_voice import DURATIONS_FILE, SPLITS, load_split, missing_audio


def pack(root: Path, splits: list[str], out: Path) -> int:
    """Write ``out`` with the metadata and clips of ``splits``; return the clip count."""
    base = root.parent.parent  # .../data/raw; arcnames start at the release folder
    files = [root / "README.md", root / DURATIONS_FILE] + [root / f"{s}.tsv" for s in splits]
    clips = [clip for split in splits for clip in load_split(root, split)]
    missing = missing_audio(clips)
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips missing, e.g. {missing[0]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # MP3s are already compressed, so an uncompressed tar is nearly as small and much faster.
    with tarfile.open(out, "w") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=path.relative_to(base).as_posix())
        for clip in clips:
            tar.add(clip.audio_path, arcname=clip.audio_path.relative_to(base).as_posix())
    return len(clips)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, required=True)
    parser.add_argument("--out", help="Output .tar (default: data/colab/<release>-<splits>.tar)")
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    root = resolve_path(config["dataset"]["root"])
    release = root.parent.name
    out = resolve_path(args.out or f"data/colab/{release}-{'-'.join(args.splits)}.tar")
    count = pack(root, args.splits, out)
    print(f"Wrote {out} ({count} clips, {out.stat().st_size / 1024**2:.0f} MiB)")


if __name__ == "__main__":
    main()
