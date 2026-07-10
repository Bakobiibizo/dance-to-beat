#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.video.rotation import create_rotating_video

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".flac", ".m4a", ".mkv", ".mp3", ".mp4", ".wav"}
EFFECTS = ("beat_marker", "color_shift", "edge", "multi_band", "pulse", "subtitle")


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="rotate-to-beat",
        description="Create a beat-synchronized visualizer video from an image and audio.",
    )
    result.add_argument("--image", required=True, type=existing_file)
    result.add_argument("--audio", required=True, type=existing_file)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--effects", nargs="+", choices=EFFECTS, default=[])
    result.add_argument("--subtitles", type=existing_file, help="Subtitle JSON file")
    result.add_argument("--cpu", action="store_true", help="Disable optional CUDA processing")
    result.add_argument("--debug", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    setup_logging(args.debug)
    if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Unsupported image format: {args.image.suffix}")
    if args.audio.suffix.lower() not in AUDIO_EXTENSIONS:
        raise SystemExit(f"Unsupported audio format: {args.audio.suffix}")
    if "subtitle" in args.effects and args.subtitles is None:
        raise SystemExit("--subtitles is required when the subtitle effect is enabled")
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        create_rotating_video(
            image_path=str(args.image),
            audio_path=str(args.audio),
            output_path=str(args.output),
            effects=args.effects,
            subtitle_path=str(args.subtitles) if args.subtitles else None,
            debug=args.debug,
            use_gpu=not args.cpu,
        )
    except Exception as error:
        logging.error("Video generation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
