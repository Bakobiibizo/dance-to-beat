import os
import cv2
import numpy as np
import logging
from src.video.color_utils import interpolate_color_wheel
from src.config.audio_config import STARTING_COLORS, COLOR_WHEEL_COLORS, BAND_BASE_SCALES

class FrameGenerator:
    def __init__(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Failed to load image: {path}")
        if self.image.shape[2] == 4:
            logging.info("Alpha channel detected, dropping alpha")
            self.image = self.image[:, :, :3]
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def create_concentric_circles(self, scales, colors=None, angle=0, color_wheel_positions=None):
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Always use COLOR_WHEEL_COLORS for interpolation to ensure consistency
        colors = COLOR_WHEEL_COLORS
        layers, masks, band_colors = [], [], []

        for i, scale in enumerate(scales):
            if scale <= 0:
                layers.append(None)
                masks.append(None)
                band_colors.append(None)
                continue

            M = cv2.getRotationMatrix2D(center, angle, scale)
            scaled_image = cv2.warpAffine(self.image, M, (width, height), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            mask = np.zeros((height, width), dtype=np.uint8)
            radius = min(width, height) // 2 * scale
            cv2.circle(mask, center, int(radius), 255, -1)

            if i < len(scales) - 1 and scales[i+1] > 0:
                inner_radius = min(width, height) // 2 * scales[i+1]
                cv2.circle(mask, center, int(inner_radius), 0, -1)

            if color_wheel_positions is not None and i < len(color_wheel_positions):
                # Use color wheel interpolation
                position = color_wheel_positions[i]
                color = interpolate_color_wheel(COLOR_WHEEL_COLORS, position)
                logging.debug(f"Layer {i}: position={position:.2f}, color={color}")
            elif i < len(STARTING_COLORS):  # Use STARTING_COLORS for fixed colors
                # Use fixed color from STARTING_COLORS
                color = STARTING_COLORS[i]
                logging.debug(f"Layer {i}: using fixed color={color}")
            else:
                # Raise an error instead of silently falling back to white
                raise ValueError(f"No color specified for layer {i}. Check color_wheel_positions or STARTING_COLORS.")

            # Store the color used for this band
            band_colors.append(color)

            tinted = cv2.addWeighted(scaled_image, 0.3, np.full_like(scaled_image, color), 0.3, 0)
            layers.append(tinted)
            masks.append(mask)

        composite_frame = np.zeros_like(frame)
        
        # Process layers from inner to outer (reverse order)
        for i in range(len(layers) - 1, -1, -1):
            # Skip layers with None values (zero scale)
            if layers[i] is None or masks[i] is None:
                continue
                
            # Create a 3-channel mask
            mask_3ch = np.dstack([masks[i]] * 3) / 255.0
            
            # Apply the current layer to the composite frame
            # For areas where the mask is 1, use the layer's color
            # For areas where the mask is 0, keep the existing composite frame
            
            # Color tinting has already been applied when creating the layers
            # No need to apply color tint again here
            
            # Now apply the layer with its mask
            layer_contribution = (layers[i] * mask_3ch).astype(np.uint8)
            inverse_mask = 1.0 - mask_3ch
            existing_content = (composite_frame * inverse_mask).astype(np.uint8)
            composite_frame = layer_contribution + existing_content

        return composite_frame
