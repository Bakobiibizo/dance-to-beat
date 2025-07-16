"""Video Generation Module for Beat-Synchronized Rotating Animations.

This module provides the core functionality for creating beat-synchronized
rotating videos from images and audio files. It includes classes and functions for:

- Frame generation with various visual effects
- Color wheel management and interpolation
- Video creation with audio synchronization
- Effect application (rotation, color shifting, pulsing, beat markers)

The main entry point is the create_rotating_video function, which handles
the entire process of generating a video from an image and audio file.

This is the canonical implementation for video generation in the library.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.audio.beat_detection import get_beats
from src.audio.bpm_detection import detect_bpm
from src.audio.multi_band import get_multi_band_envelopes
from src.video.color_utils import interpolate_color_wheel, get_band_positions, ColorWheelCache
from src.config.audio_config import BAND_BASE_SCALES, PULSE_MIN_SCALE, PULSE_MAX_SCALE, PULSE_INTENSITY, MARKER_INTENSITY, COLOR_WHEEL_COLORS, STARTING_COLORS, OUTPUT_PATH

def apply_color_wheel_shift(frame, colors, position):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    
    # Create a mask for the center circle
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, 255, -1)
    
    # Interpolate color wheel position
    wheel_pos = position * len(colors)
    idx1 = int(wheel_pos) % len(colors)
    idx2 = (idx1 + 1) % len(colors)
    weight = wheel_pos - int(wheel_pos)
    c1 = np.array(colors[idx1])
    c2 = np.array(colors[idx2])
    color = tuple((c1 * (1 - weight) + c2 * weight).astype(int))
    
    # Apply color shift to the center circle
    tinted = cv2.addWeighted(frame, 0.3, np.full_like(frame, color), 0.7, 0)
    
    # Overlay the tinted image with the original mask
    mask_3ch = np.dstack([mask] * 3) / 255.0
    overlay = (tinted * mask_3ch).astype(np.uint8)
    inverse = (frame * (1.0 - mask_3ch)).astype(np.uint8)
    composite = overlay + inverse
    
    return composite

class FrameGenerator:
    """
    Class to handle image loading and preparation for video generation.
    """
    def __init__(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Failed to load image: {path}")
        if self.image.shape[2] == 4:
            logging.info("Alpha channel detected, dropping alpha")
            self.image = self.image[:, :, :3]
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.debug_frames_saved = 0
        self.output_path = Path.cwd() / OUTPUT_PATH 
        
    def create_concentric_circles(self, scales, colors=None, angle=0, color_wheel_positions=None, effects=None, band_envelopes=None):
        """
        Create concentric circles with different scales and colors.
        
        Args:
            scales: List of scale factors for each circle (outer to inner)
            colors: List of (R,G,B) colors for each circle, or color wheel for interpolation
            angle: Rotation angle in degrees
            color_wheel_positions: List of positions (0-1) in the color wheel for each circle
                                   If provided, will override the colors parameter
            
        Returns:
            Frame with concentric circles
        """
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create a blank canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Default colors if none provided
        if colors is None:
            colors = COLOR_WHEEL_COLORS  # Use the expanded color wheel for interpolation
        
        # Create separate layers for each circle
        layers = []
        masks = []
        BAND_STARTING_COLORS = []
        # Draw circles from outer to inner
        for i, scale in enumerate(scales):
            # Skip bands with zero scale (used for debugging)
            if scale <= 0:
                layers.append(None)
                masks.append(None)
                BAND_STARTING_COLORS.append(None)
                continue
                
            # Scale and rotate the image
            M = cv2.getRotationMatrix2D(center, angle, scale)
            scaled_image = cv2.warpAffine(self.image, M, (width, height), 
                                          flags=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_CONSTANT, 
                                          borderValue=(0, 0, 0))
            
            # Create a mask for this circle
            mask = np.zeros((height, width), dtype=np.uint8)
            radius = min(width, height) // 2 * scale
            cv2.circle(mask, center, int(radius), 255, -1)
            
            # If this is not the innermost circle, cut out the inner circle
            if i < len(scales) - 1 and scales[i+1] > 0:
                inner_radius = min(width, height) // 2 * scales[i+1]
                cv2.circle(mask, center, int(inner_radius), 0, -1)
            
            # Determine color for this band
            if color_wheel_positions is not None and i < len(color_wheel_positions):
                # Use color wheel interpolation
                position = color_wheel_positions[i]
                # Use standard interpolation since we don't have access to the cache here
                color = interpolate_color_wheel(colors, position)
                # logging.info(f"Layer {i}: position={position:.2f}, color={color}")
            elif i < len(STARTING_COLORS):  # Use STARTING_COLORS for fixed colors
                # Use fixed color from STARTING_COLORS
                color = STARTING_COLORS[i]
                # logging.info(f"Layer {i}: using fixed color={color}")
            else:
                # Raise an error instead of silently falling back to white
                raise ValueError(f"No color specified for layer {i}. Check color_wheel_positions or STARTING_COLORS.")
            
            # Store the color used for this band
            BAND_STARTING_COLORS.append(color)
            
            # Apply color tint to this band
            tinted = scaled_image.copy()
            # Use a stronger color tint to make the bands distinct while preserving image content
            color_overlay = np.full_like(tinted, color)
            # Blend with 70% color and 30% original image to maintain image visibility
            tinted = cv2.addWeighted(tinted, 0.3, color_overlay, 0.7, 0)
            
            # Store the layer and mask for later composition
            layers.append(tinted)
            masks.append(mask)
        
        # Create a composite frame by applying each layer with its mask
        # We need to apply masks in reverse order (inner to outer) to ensure proper layering
        # This ensures that each band maintains its distinct color
        composite_frame = np.zeros_like(frame)
        
        # Process layers from inner to outer (reverse order)
        for i in range(len(layers) - 1, -1, -1):
            # Skip layers with None values (zero scale)
            if layers[i] is None or masks[i] is None:
                continue
                
            # Create a 3-channel mask
            mask_3ch = np.dstack([masks[i], masks[i], masks[i]]) / 255.0
            
            # Apply the current layer to the composite frame
            # For areas where the mask is 1, use the layer's color
            # For areas where the mask is 0, keep the existing composite frame
            
            # Apply a strong color tint while preserving image content
            if STARTING_COLORS[i] is not None:
                # Create a color overlay
                color_overlay = np.full_like(layers[i], STARTING_COLORS[i])
                # Blend with 70% color and 30% original image
                layers[i] = cv2.addWeighted(layers[i], 0.3, color_overlay, 0.3, 0)
            
            # Now apply the layer with its mask
            layer_contribution = (layers[i] * mask_3ch).astype(np.uint8)
            inverse_mask = 1.0 - mask_3ch
            existing_content = (composite_frame * inverse_mask).astype(np.uint8)
            composite_frame = layer_contribution + existing_content
        
        # Return the composite frame
        frame = composite_frame
        
        # Debug: Save the first 10 frames to analyze colors
        if hasattr(self, 'debug_frames_saved') and self.debug_frames_saved < 10:
            frame_path = os.path.join(os.path.dirname(self.output_path), "debug_frames", f"frame_{self.debug_frames_saved:03d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Also save individual band frames for the first frame
            if self.debug_frames_saved == 0 and 'multi_band' in effects and band_envelopes:
                # Save each band separately
                for i, band_name in enumerate(["bass", "low_mid", "high_mid", "high"]):
                    if i < len(scales):
                        # Create a frame with just this band by setting all other scales to 0
                        single_band_scales = [0] * len(scales)
                        single_band_scales[i] = scales[i]
                        
                        single_band_positions = [0] * len(color_wheel_positions)
                        single_band_positions[i] = color_wheel_positions[i]
                        
                        band_frame = self.create_concentric_circles(
                            single_band_scales, 
                            colors=COLOR_WHEEL_COLORS,
                            angle=angle,
                            color_wheel_positions=single_band_positions
                        )
                        band_path = os.path.join(os.path.dirname(output_path), "debug_frames", f"band_{i}_{band_name}.png")
                        cv2.imwrite(band_path, cv2.cvtColor(band_frame, cv2.COLOR_RGB2BGR))
                        # logging.info(f"Saved band {i} ({band_name}) frame with position {color_wheel_positions[i]} and color {STARTING_COLORS[i] if i < len(STARTING_COLORS) else 'unknown'}")
            
            self.debug_frames_saved += 1
        
        return frame

def create_rotating_video(image_path, audio_path, output_path, effects=None, debug=False):
    """
    Create a video with a rotating image synchronized to audio beats.
    
    Args:
        image_path: Path to the image file
        audio_path: Path to the audio file
        output_path: Path for the output video
        effects: List of visual effects to apply ('pulse', 'color_shift', 'multi_band', etc.)
        debug: Enable debug mode
    """
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Image: {image_path}, Audio: {audio_path}, Output: {output_path}, Effects: {effects}")
    effects = effects or []
    temp_video = 'output/temp_video.mp4'

    try:
        with AudioFileClip(audio_path) as audio_clip:
            bpm = detect_bpm(audio_path)
            beat_frames = get_beats(audio_path)
            
            # Get multi-band envelopes for frequency-based effects
            band_envelopes = None
            if 'multi_band' in effects:
                band_envelopes, _ = get_multi_band_envelopes(audio_path)
                logging.info(f"Extracted envelopes for {len(band_envelopes)} frequency bands")
                
            # Initialize color wheel cache for performance optimization
            color_wheel_cache = ColorWheelCache()
            logging.info("Color wheel cache initialized for optimized color interpolation")
            
            if len(beat_frames) == 0:
                logging.warning("No beats detected")
                
            fps = 30
            duration = audio_clip.duration
            rotation_per_frame = (bpm / 60.0) * 360 / fps

            frame_gen = FrameGenerator(image_path)
            
            # Use a class to maintain state between frames
            class FrameState:
                def __init__(self):
                    # Store reference to the color wheel cache
                    self.color_cache = color_wheel_cache
                    # Initialize band positions for color wheel - persistent across frames
                    # These match the starting positions for each band color in the 12-point COLOR_WHEEL_COLORS
                    # For STARTING_COLORS: Purple, Blue, Yellow, Orange
                    # We need to find the exact positions in COLOR_WHEEL_COLORS that match these colors
                    
                    # Find the exact positions in COLOR_WHEEL_COLORS for each BAND_COLOR
                    self.start_positions = []
                    for band_color in STARTING_COLORS:
                        # Find the closest match in COLOR_WHEEL_COLORS
                        found = False
                        for i, wheel_color in enumerate(COLOR_WHEEL_COLORS):
                            if wheel_color == band_color:
                                # Exact match found
                                position = i / len(COLOR_WHEEL_COLORS)
                                self.start_positions.append(position)
                                found = True
                                logging.info(f"Found exact match for {band_color} at position {position} (index {i})")
                                break
                        
                        if not found:
                            # If no exact match, find the closest color in the wheel
                            logging.warning(f"No exact match found for {band_color} in COLOR_WHEEL_COLORS, finding closest match")
                            # Convert colors to numpy arrays for distance calculation
                            band_color_array = np.array(band_color)
                            min_distance = float('inf')
                            closest_index = 0
                            
                            for i, wheel_color in enumerate(COLOR_WHEEL_COLORS):
                                wheel_color_array = np.array(wheel_color)
                                distance = np.sum((band_color_array - wheel_color_array) ** 2)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_index = i
                            
                            position = closest_index / len(COLOR_WHEEL_COLORS)
                            self.start_positions.append(position)
                            logging.info(f"Using closest match for {band_color} at position {position} (index {closest_index})")
                    
                    # Ensure we have a position for each band
                    if len(self.start_positions) < len(BAND_BASE_SCALES):
                        logging.warning(f"Not enough start positions ({len(self.start_positions)}) for all bands ({len(BAND_BASE_SCALES)})")
                        # Add default positions for any missing bands
                        default_positions = [0.0, 0.25, 0.5, 0.75]  # N, E, S, W
                        while len(self.start_positions) < len(BAND_BASE_SCALES):
                            idx = len(self.start_positions) % len(default_positions)
                            self.start_positions.append(default_positions[idx])
                            logging.warning(f"Added default position {default_positions[idx]} for band {len(self.start_positions)-1}")
                    
                    logging.info(f"Calculated start positions: {self.start_positions}")
                    self.speeds = [0.002, 0.004, 0.008, 0.016]     # Each band moves at a slightly different speed
                    self.positions = self.start_positions.copy()    # Current positions, updated each frame
                    self.frame_count = 0
                    self.debug_frames_saved = 0
                
                def update_positions(self):
                    # Update each band's position independently based on its speed
                    for i in range(len(self.positions)):
                        if i < len(self.speeds):
                            self.positions[i] = (self.positions[i] + self.speeds[i]) % 1.0
                        else:
                            # If we don't have a speed for this band, use a default
                            self.positions[i] = (self.positions[i] + 0.003) % 1.0
                    
                    # Debug log the positions every 30 frames
                    if self.frame_count % 30 == 0:
                        logging.debug(f"Frame {self.frame_count}: Band positions = {[round(p, 3) for p in self.positions]}")
                    
                    self.frame_count += 1
            
            # Create a single instance to maintain state
            state = FrameState()
            
            # Create debug output directory if it doesn't exist
            debug_dir = os.path.join(os.path.dirname(output_path), "debug_frames")
            os.makedirs(debug_dir, exist_ok=True)

            def make_frame(t):
                height, width = frame_gen.image.shape[:2]
                center = (width // 2, height // 2)
                angle = (t * rotation_per_frame) % 360
                frame_idx = int(t * fps)
                
                # Update band positions for this frame
                state.update_positions()
                
                # Multi-band concentric circles effect
                if 'multi_band' in effects and band_envelopes:
                    # Calculate scales for each frequency band
                    scales = []
                    base_scales = BAND_BASE_SCALES  # Base scales for each circle (outer to inner)
                    band_names = ["bass", "low_mid", "high_mid", "high"]
                    # Initialize beat intensity for multi-band effect
                    multi_band_intensity = 0
                    for band_name, base_scale in zip(band_names, base_scales):
                        if band_name in band_envelopes:
                            env = band_envelopes[band_name]
                            if len(env) > 0:
                                # Map frame index to envelope index
                                env_idx = min(int(frame_idx * len(env) / (fps * duration)), len(env) - 1)
                                # Scale from base_scale to base_scale*1.3 based on envelope
                                intensity = env[env_idx]
                                multi_band_intensity = max(multi_band_intensity, intensity)
                                # Apply non-linear scaling for more dramatic effect
                                scale = base_scale + (intensity ** 2) * PULSE_INTENSITY
                                scales.append(scale)
                            else:
                                scales.append(base_scale)
                        else:
                            scales.append(base_scale)
                    
                    # If color_shift is also enabled, calculate color wheel positions for each layer
                    color_wheel_positions = None
                    
                    if 'color_shift' in effects:
                        # Calculate different positions for each layer
                        # Each layer will have a different starting position and speed
                        # to create a visually interesting effect
                        color_wheel_positions = []
                        
                        # Use the updated band positions for this frame
                        for i in range(len(scales)):
                            if i < len(state.positions):
                                pos = state.positions[i]
                                color_wheel_positions.append(pos)
                                
                                # Debug log for the first few frames
                                if state.frame_count < 5:
                                    color = state.color_cache.get_color(pos)
                                    logging.info(f"Frame {state.frame_count} - Band {i}: position={pos:.3f}, color={color}")
                            else:
                                # Raise an error instead of silently falling back
                                raise ValueError(f"Missing position for layer {i}. Check state.positions initialization.")
                    
                    # Add debug logging for the first frame
                    is_first_frame = (frame_idx == 0)
                    if is_first_frame and color_wheel_positions:
                        logging.info(f"First frame color wheel positions: {[round(p, 3) for p in color_wheel_positions]}")
                        for i, pos in enumerate(color_wheel_positions):
                            if i < len(color_wheel_positions):
                                color = state.color_cache.get_color(pos)
                                logging.info(f"First frame - Band {i}: position={pos:.3f}, color={color}")
                                
                                # Calculate the actual color from the 12-point wheel
                                wheel_pos = pos * len(COLOR_WHEEL_COLORS)
                                idx = int(wheel_pos) % len(COLOR_WHEEL_COLORS)
                                actual_color = COLOR_WHEEL_COLORS[idx]
                                logging.info(f"Position {pos:.3f} maps to wheel index {idx} with color {actual_color}")
                                
                                # Also log the interpolated color
                                interp_color = state.color_cache.get_color(pos)
                                logging.info(f"Interpolated color for position {pos:.3f}: {interp_color}")
                    
                    # Debug logging already handled above
                    
                    # Create frame with concentric circles
                    frame = frame_gen.create_concentric_circles(
                        scales,
                        colors=COLOR_WHEEL_COLORS,
                        angle=angle,
                        color_wheel_positions=color_wheel_positions,
                        effects=effects,
                        band_envelopes=multi_band_envelopes if 'multi_band' in effects else None
                    )
                
                # Standard single-image pulse effect
                elif 'pulse' in effects:
                    # Initialize scale for pulsing/bouncing effect
                    scale = PULSE_MIN_SCALE
                    current_frame = int(t * fps)
                    
                    # Strong pulse based on bass frequencies
                    if frame_idx < len(beat_frames):
                        bass_idx = int(beat_frames[frame_idx])
                        if 0 <= bass_idx < len(beat_frames):
                            # Apply stronger scaling effect for more visible bouncing
                            # Square the bass value to make peaks more pronounced
                            bass_value = beat_frames[bass_idx]
                            # Scale from min to max based on bass intensity
                            scale = PULSE_MIN_SCALE + (bass_value ** 2) * PULSE_INTENSITY
                    
                    # Also check for exact beat hits for extra emphasis
                    if not isinstance(beat_frames, np.ndarray):
                        beat_frames_array = np.array(beat_frames)
                    else:
                        beat_frames_array = beat_frames
                    
                    # Add extra emphasis on exact beat frames
                    if len(beat_frames_array) > 0:
                        distances = np.abs(beat_frames_array - current_frame)
                        min_distance = np.min(distances)
                        
                        # Extra boost exactly on beats
                        if min_distance == 0:  # Exactly on a beat
                            scale = max(scale, PULSE_MAX_SCALE)  # Ensure maximum scale on exact beat
                        elif min_distance == 1:  # One frame away from beat
                            scale = max(scale, PULSE_MAX_SCALE * 0.9)  # Strong emphasis near beat
                    
                    # Apply rotation and scaling to the image
                    frame = frame_gen.image.copy()
                    M = cv2.getRotationMatrix2D(center, angle, scale)
                    frame = cv2.warpAffine(frame, M, (width, height), flags=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    
                    # Initialize beat intensity for beat marker effect
                    beat_intensity = 0
                
                    # Apply color wheel shift using the STARTING_COLORS from config
                    if len(beat_frames_array) > 0:  # Make sure we have beats to check
                        # Find the closest beat to current frame
                        distances = np.abs(beat_frames_array - current_frame)
                        # Get indices of up to 5 closest beats (or fewer if we don't have 5)
                        num_beats_to_check = min(5, len(beat_frames_array))
                        closest_beats_idx = np.argsort(distances)[:num_beats_to_check]
                        closest_beats = beat_frames_array[closest_beats_idx]
                        
                        for beat_frame in closest_beats:
                            distance = abs(current_frame - beat_frame)
                            if distance < 3:  # Within 3 frames of a beat
                                # Linear falloff with distance
                                intensity = MARKER_INTENSITY * (1.0 - (distance / 3.0))
                                beat_intensity = max(intensity, beat_intensity)
                    
                    # Apply a very simple brightness boost if this is a beat frame
                    if beat_intensity > 0:
                        # White flash effect using configurable intensity
                        white_overlay = np.ones_like(frame) * 255
                        frame = cv2.addWeighted(frame, 1.0 - beat_intensity, white_overlay, beat_intensity, 0)

                frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Debug: Save the first few frames to analyze colors
                if state.frame_count < 5 and 'multi_band' in effects:
                    # Create debug directory if it doesn't exist
                    debug_dir = os.path.join(os.path.dirname(output_path), "debug_frames")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save the full frame
                    frame_path = os.path.join(debug_dir, f"frame_{state.frame_count:03d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # For the first frame, save each band separately to see their colors
                    if state.frame_count == 0:
                        logging.info(f"Saving individual band frames for debugging")
                        logging.info(f"Current band positions: {state.positions}")
                        
                        # Save each band separately
                        for i, band_name in enumerate(["bass", "low_mid", "high_mid", "high"]):
                            if i < len(scales):
                                # Create a frame with just this band
                                single_band_scales = [0, 0, 0, 0]  # Initialize all to 0
                                single_band_scales[i] = scales[i]  # Set only this band's scale
                                
                                # Create color wheel positions with just this band's position
                                single_band_positions = [0, 0, 0, 0]  # Initialize all to 0
                                single_band_positions[i] = state.positions[i]  # Set only this band's position
                                
                                # Generate a frame with just this band
                                band_frame = frame_gen.create_concentric_circles(
                                    single_band_scales,
                                    colors=COLOR_WHEEL_COLORS,  # Use the full color wheel
                                    angle=angle,
                                    color_wheel_positions=single_band_positions
                                )
                                
                                # Save the band frame
                                band_path = os.path.join(debug_dir, f"band_{i}_{band_name}.png")
                                cv2.imwrite(band_path, cv2.cvtColor(band_frame, cv2.COLOR_RGB2BGR))
                                
                                # Log the band's color information
                                wheel_idx = int(state.positions[i] * len(COLOR_WHEEL_COLORS))
                                color = COLOR_WHEEL_COLORS[wheel_idx % len(COLOR_WHEEL_COLORS)]
                                logging.info(f"Band {i} ({band_name}): position={state.positions[i]}, wheel_idx={wheel_idx}, color={color}")
                    
                    state.frame_count += 1
                
                return frame

            frame_size = (frame_gen.image.shape[1], frame_gen.image.shape[0])
            writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer. Ensure 'mp4v' codec is available or try .avi")

            for i in range(int(duration * fps)):
                t = i / fps
                frame = make_frame(t)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
                if i % 100 == 0:
                    logging.info(f"Frame {i}/{int(duration * fps)}")
            writer.release()
            if int(t * fps) < 3:
                cv2.imwrite(f"debug_frame_{int(t*fps)}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            with VideoFileClip(temp_video) as video:
                final = video.with_audio(audio_clip)
                final.write_videofile(output_path, fps=fps)

    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
            logging.info("Temporary video file removed")
