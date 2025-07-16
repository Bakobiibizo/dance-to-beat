import logging
from src.video.color_utils import get_band_positions, interpolate_color_wheel
from src.config.audio_config import COLOR_WHEEL_COLORS, LOOP_FRAMES, BAND_SPEEDS

# Set up logging
frame = 240  # loop point
positions = get_band_positions(frame)
colors = [interpolate_color_wheel(COLOR_WHEEL_COLORS, p) for p in positions]

print(f"Frame {frame}:")
for i, color in enumerate(colors):
    print(f"  Band {i+1} color: {color}")