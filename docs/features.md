# Advanced Video Effects Features

This document describes the advanced video effects features available in the video processing pipeline.

## Canny Edge Detection Effect

The Canny edge detection effect applies an artistic edge detection filter to your video frames. This effect is similar to the edge detection used in ControlNet and Stable Diffusion pipelines, creating a stylized outline effect.

### Usage

To use the edge detection effect, add the `edge` effect to your command:

```bash
python rotate_to_beat.py --image input.jpg --audio music.mp3 --output output.mp4 --effects color_shift pulse edge
```

### Configuration

The edge detection effect can be configured in `src/config/audio_config.py`:

- `EDGE_INTENSITY`: Controls the blend intensity between the edge detection and the original frame (0.0-1.0)
- `EDGE_LOW_THRESHOLD`: Lower threshold for Canny edge detection (higher values = fewer edges)
- `EDGE_HIGH_THRESHOLD`: Higher threshold for Canny edge detection (higher values = stronger edges)

## Karaoke-Style Subtitles

The karaoke-style subtitles feature allows you to add synchronized lyrics to your videos. Words are highlighted as they are spoken/sung, creating a karaoke-like effect.

### Usage

To add subtitles to your video:

```bash
python rotate_to_beat.py --image input.jpg --audio music.mp3 --output output.mp4 --effects color_shift pulse subtitle --subtitle-path lyrics.json
```

### Subtitle JSON Format

Subtitles must be provided in a JSON file with the following format:

```json
[
    {"text": "First line of lyrics", "start_time": 1.0, "end_time": 3.0},
    {"text": "Second line of lyrics", "start_time": 3.5, "end_time": 5.5},
    ...
]
```

Each entry contains:
- `text`: The text to display
- `start_time`: When to start displaying this line (in seconds)
- `end_time`: When to stop displaying this line (in seconds)

### Speech-to-Text Integration

The `subtitle_generator.py` utility helps convert speech-to-text API outputs to the required subtitle format:

```bash
python -m src.utils.subtitle_generator --input transcription.json --output lyrics.json --format google
```

Supported formats:
- `google`: Google Cloud Speech-to-Text API
- `azure`: Azure Speech Service API
- `aws`: AWS Transcribe API
- `generic`: Generic word-level timestamps

### Configuration

Subtitle appearance can be configured in `src/config/audio_config.py`:

- `SUBTITLE_FONT_SCALE`: Font size scale factor
- `SUBTITLE_FONT_THICKNESS`: Font thickness
- `SUBTITLE_FONT_COLOR`: Text color (RGB tuple)
- `SUBTITLE_HIGHLIGHT_COLOR`: Color for highlighted/active words (RGB tuple)
- `SUBTITLE_BG_COLOR`: Background color (RGB tuple)
- `SUBTITLE_BG_ALPHA`: Background opacity (0.0-1.0)
- `SUBTITLE_POSITION`: Vertical position ("top", "middle", "bottom")
- `SUBTITLE_MAX_WIDTH_RATIO`: Maximum width as ratio of frame width

### Demo

Try the subtitle demo script:

```bash
python examples/subtitle_demo.py --image input.jpg --audio music.mp3 --output output.mp4
```

This uses the sample lyrics provided in `examples/sample_lyrics.json`.
