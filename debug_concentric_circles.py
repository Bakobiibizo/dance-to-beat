import cv2
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import our modules
from src.config.audio_config import COLOR_WHEEL_COLORS, BAND_COLORS
from src.video.color_utils import interpolate_color_wheel

# Create output directory
os.makedirs("output/debug_circles", exist_ok=True)

# Test positions and scales
positions = [0.5, 0.75, 0.0, 0.25]  # Purple, Blue, Yellow, Orange
scales = [1.0, 0.75, 0.5, 0.25]  # Base scales for concentric circles (outer to inner)

# Create a blank canvas
height, width = 800, 800
center = (width // 2, height // 2)
frame = np.zeros((height, width, 3), dtype=np.uint8)

# Create separate layers for each circle
layers = []
masks = []

# Draw circles from outer to inner
for i, scale in enumerate(scales):
    # Create a mask for this circle
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = min(width, height) // 2 * scale
    cv2.circle(mask, center, int(radius), 255, -1)
    
    # If this is not the innermost circle, cut out the inner circle
    if i < len(scales) - 1:
        inner_radius = min(width, height) // 2 * scales[i+1]
        cv2.circle(mask, center, int(inner_radius), 0, -1)
    
    # Determine color for this band
    if i < len(positions):
        # Use color wheel interpolation
        position = positions[i]
        color = interpolate_color_wheel(COLOR_WHEEL_COLORS, position)
        logging.info(f"Layer {i}: position={position:.2f}, color={color}")
    else:
        # Fallback
        color = (255, 255, 255)
        logging.info(f"Layer {i}: using fallback white color")
    
    # Create a colored layer
    layer = np.full((height, width, 3), color, dtype=np.uint8)
    
    # Store the layer and mask for later composition
    layers.append(layer)
    masks.append(mask)

# Compose the final frame by applying each layer with its mask
for i in range(len(layers)):
    mask_3ch = np.dstack([masks[i], masks[i], masks[i]]) / 255.0
    frame = frame + (layers[i] * mask_3ch).astype(np.uint8)

# Save the frame
cv2.imwrite("output/debug_circles/concentric_circles.png", frame)

# Save individual band frames
for i in range(len(scales)):
    # Create a frame with just this band
    band_frame = np.zeros((height, width, 3), dtype=np.uint8)
    mask_3ch = np.dstack([masks[i], masks[i], masks[i]]) / 255.0
    band_frame = band_frame + (layers[i] * mask_3ch).astype(np.uint8)
    
    # Save the band frame
    band_path = f"output/debug_circles/band_{i}.png"
    cv2.imwrite(band_path, band_frame)
    
print("Debug images saved to output/debug_circles/")
