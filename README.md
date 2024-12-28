# Rotate Image to Beat

A Python toolkit for creating mesmerizing music visualizations by rotating images in sync with audio beats. The project includes tools for creating circular masks and generating beat-synchronized video effects.

## Features

### Image Processing (`circle_image.py`)
- Create perfect circular masks for images
- Automatic square image conversion
- Adjustable padding for circle size
- Smooth edge handling for the circular mask

### Video Generation (`rotate_to_beat.py`)
- Automatic BPM detection from audio files
- Frequency-filtered beat detection (100-350Hz range)
- Multiple visual effects:
  - Beat-synchronized rotation
  - Color shifting
  - Pulsing effect based on audio intensity
  - Edge glow effect on beats
- Supports multiple image formats (PNG, JPEG, WEBP)
- Supports multiple audio formats (MP3, WAV, MP4, MKV)
- Creates MP4 video output

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Create a Circular Mask

First, create a circular mask for your image:
```bash
python circle_image.py --image path/to/image.jpg --padding 16
```

Parameters:
- `--image`: Path to the input image (PNG, JPEG, or WEBP)
- `--padding`: Padding from edges in pixels (default: 16)

### 2. Generate the Video

Then create the video using the masked image:
```bash
python rotate_to_beat.py --image path/to/masked_image.png --audio path/to/audio.mp3 --output output.mp4 --effects color_shift pulse beat_marker
```

Parameters:
- `--image`: Path to the masked image (PNG, JPEG, or WEBP)
- `--audio`: Path to the audio file (MP3, WAV, MP4, or MKV)
- `--output`: Path for the output video file (MP4)
- `--effects`: List of effects to apply (optional):
  - `color_shift`: Shifts colors based on rotation
  - `pulse`: Pulses the image size based on audio intensity
  - `beat_marker`: Adds a glowing effect on beats

Example workflow:
```bash
# 1. Create circular mask with 32px padding
python circle_image.py --image "./media/spiral.jpeg" --padding 32

# 2. Create video with all effects
python rotate_to_beat.py --image "./media/masked_image.png" --audio "./media/music.mp3" --output "./output/output.mp4" --effects color_shift pulse beat_marker
```

## Technical Details

### Beat Detection
- Uses librosa for audio analysis
- Frequency filtering in 100-350Hz range for better beat detection
- Onset strength detection for accurate beat timing

### Image Processing
- OpenCV for image manipulation
- Numpy for efficient array operations
- Smooth edge handling in circular masks
- Automatic square conversion with centered content

### Video Generation
- MoviePy for video creation
- OpenCV for real-time frame manipulation
- Efficient frame generation with numpy operations

## Requirements
- Python 3.8+
- OpenCV (cv2)
- Numpy
- Librosa
- MoviePy

## License
MIT License - Feel free to use and modify as needed!
