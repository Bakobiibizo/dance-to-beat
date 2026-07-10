# Dance to Beat

`dance-to-beat` creates an MP4 music visualizer by synchronizing a source image
with an audio track. It detects tempo and beat positions, rotates and pulses the
image, and can add color, edge, frequency-band, and subtitle effects.

## Requirements

- Python 3.11 or newer
- ffmpeg available on `PATH`
- Optional CUDA-enabled OpenCV and CuPy for GPU processing

CPU rendering is fully supported and is the default fallback.

## Install

```bash
uv tool install .
```

For development:

```bash
uv sync --extra dev
uv run pytest
```

## Render

```bash
rotate-to-beat \
  --image cover.png \
  --audio track.mp3 \
  --output visualizer.mp4 \
  --effects pulse color_shift beat_marker
```

Available effects are `pulse`, `color_shift`, `beat_marker`, `edge`,
`multi_band`, and `subtitle`. Use `--cpu` to explicitly disable CUDA.

Subtitles use the JSON format documented in `docs/features.md`:

```bash
rotate-to-beat \
  --image cover.png \
  --audio track.mp3 \
  --output visualizer.mp4 \
  --effects pulse subtitle \
  --subtitles lyrics.json
```

The command creates missing output directories, rejects unsupported or missing
inputs before processing, returns a nonzero status on render failure, and uses a
unique temporary video per invocation so concurrent renders do not collide.

## Circular Images

The optional mask helper prepares rectangular artwork as a centered circle:

```bash
python -m circle_image --image cover.jpg --padding 24
```

## Validation

The test suite covers beat envelopes, multiband extraction, frame preparation,
CLI validation, and a real ffmpeg video render with an attached audio stream.
CI runs on Python 3.11, 3.12, and 3.13 and builds both wheel and source archives.

## License

MIT
