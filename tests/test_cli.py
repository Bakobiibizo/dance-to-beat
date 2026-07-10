from pathlib import Path
from unittest.mock import patch

import pytest

from rotate_to_beat_cli import main, parser


def test_parser_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parser().parse_args([
            "--image", str(tmp_path / "missing.png"),
            "--audio", str(tmp_path / "missing.wav"),
            "--output", str(tmp_path / "out.mp4"),
        ])


def test_subtitle_effect_requires_subtitle_file(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    audio = tmp_path / "audio.wav"
    image.touch()
    audio.touch()
    with pytest.raises(SystemExit, match="--subtitles is required"):
        main([
            "--image", str(image), "--audio", str(audio),
            "--output", str(tmp_path / "out.mp4"), "--effects", "subtitle",
        ])


def test_cli_forwards_validated_paths(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    audio = tmp_path / "audio.wav"
    image.touch()
    audio.touch()
    output = tmp_path / "nested" / "out.mp4"
    with patch("rotate_to_beat_cli.create_rotating_video") as create:
        assert main([
            "--image", str(image), "--audio", str(audio), "--output", str(output),
            "--effects", "pulse", "--cpu",
        ]) == 0
    assert output.parent.is_dir()
    assert create.call_args.kwargs["use_gpu"] is False
