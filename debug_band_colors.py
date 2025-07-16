import cv2
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import our modules
from src.config.audio_config import COLOR_WHEEL_COLORS, BAND_COLORS
from src.video.color_utils import interpolate_color_wheel
from src.video.rotation import FrameGenerator

# Create output directory
os.makedirs("output/debug_colors", exist_ok=True)

# Load the image
image_path = "media/white_swirl.png"
frame_gen = FrameGenerator(image_path)

# Test positions
positions = [0.5, 0.75, 0.0, 0.25]  # Purple, Blue, Yellow, Orange
scales = [1.0, 0.75, 0.5, 0.25]  # Base scales for concentric circles (outer to inner)

# Create a frame with all bands
all_bands_frame = frame_gen.create_concentric_circles(
    scales=scales,
    colors=COLOR_WHEEL_COLORS,
    angle=0,
    color_wheel_positions=positions
)

# Save the full frame
cv2.imwrite("output/debug_colors/all_bands.png", cv2.cvtColor(all_bands_frame, cv2.COLOR_RGB2BGR))

# Create and save individual band frames
for i, (pos, scale) in enumerate(zip(positions, scales)):
    # Create a frame with just this band
    single_band_scales = [0] * len(scales)
    single_band_scales[i] = scale
    
    single_band_positions = [0] * len(positions)
    single_band_positions[i] = pos
    
    # Generate a frame with just this band
    band_frame = frame_gen.create_concentric_circles(
        scales=single_band_scales,
        colors=COLOR_WHEEL_COLORS,
        angle=0,
        color_wheel_positions=single_band_positions
    )
    
    # Save the band frame
    band_path = f"output/debug_colors/band_{i}.png"
    cv2.imwrite(band_path, cv2.cvtColor(band_frame, cv2.COLOR_RGB2BGR))
    
    # Log the band's color information
    wheel_idx = int(pos * len(COLOR_WHEEL_COLORS))
    color = COLOR_WHEEL_COLORS[wheel_idx % len(COLOR_WHEEL_COLORS)]
    logging.info(f"Band {i}: position={pos}, wheel_idx={wheel_idx}, color={color}")
    
print("Debug images saved to output/debug_colors/")
