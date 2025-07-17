"""
Subtitle handling module for karaoke-style synchronized lyrics.

This module provides functionality to:
1. Load timestamped lyrics from various sources
2. Render subtitles on video frames with karaoke-style highlighting
3. Synchronize lyrics with audio timing
"""

import os
import json
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union

# Import GPU utilities
from ..utils.gpu_utils import to_gpu, to_cpu, HAS_CUPY

logger = logging.getLogger(__name__)

class Subtitle:
    """Represents a single subtitle with text and timing information."""
    
    def __init__(self, text: str, start_time: float, end_time: float):
        """
        Initialize a subtitle.
        
        Args:
            text: The subtitle text
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        
    def is_active(self, current_time: float) -> bool:
        """Check if this subtitle should be displayed at the given time."""
        return self.start_time <= current_time <= self.end_time
    
    def get_progress(self, current_time: float) -> float:
        """Get the progress through this subtitle (0.0 to 1.0)."""
        if current_time <= self.start_time:
            return 0.0
        if current_time >= self.end_time:
            return 1.0
        
        total_duration = self.end_time - self.start_time
        if total_duration <= 0:
            return 1.0
            
        elapsed = current_time - self.start_time
        return min(1.0, max(0.0, elapsed / total_duration))


class SubtitleManager:
    """Manages a collection of subtitles and renders them on video frames."""
    
    def __init__(self, font_scale: float = 1.0, font_thickness: int = 2, 
                 font_color: Tuple[int, int, int] = (255, 255, 255),
                 highlight_color: Tuple[int, int, int] = (255, 255, 0),
                 background_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                 background_alpha: float = 0.5,
                 position: str = "bottom",
                 max_width_ratio: float = 0.8):
        """
        Initialize the subtitle manager.
        
        Args:
            font_scale: Scale factor for font size
            font_thickness: Thickness of font strokes
            font_color: RGB color for normal text
            highlight_color: RGB color for highlighted (active) text
            background_color: RGB color for text background, None for no background
            background_alpha: Opacity of background (0.0-1.0)
            position: Vertical position ("top", "middle", "bottom")
            max_width_ratio: Maximum width of subtitle as ratio of frame width
        """
        self.subtitles: List[Subtitle] = []
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font_color = font_color
        self.highlight_color = highlight_color
        self.background_color = background_color
        self.background_alpha = background_alpha
        self.position = position
        self.max_width_ratio = max_width_ratio
        # Use a sans-serif font instead of the default serif font
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # Clean sans-serif font
        
    def load_from_json(self, json_path: Union[str, Path]) -> bool:
        """
        Load subtitles from a JSON file.
        
        Expected format:
        [
            {"text": "Lyric line 1", "start_time": 1.2, "end_time": 3.4},
            {"text": "Lyric line 2", "start_time": 3.5, "end_time": 5.6},
            ...
        ]
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.subtitles = []
            for item in data:
                self.subtitles.append(Subtitle(
                    text=item["text"],
                    start_time=float(item["start_time"]),
                    end_time=float(item["end_time"])
                ))
            
            # Sort by start time
            self.subtitles.sort(key=lambda s: s.start_time)
            logger.info(f"Loaded {len(self.subtitles)} subtitles from {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load subtitles from {json_path}: {e}")
            return False
    
    def get_active_subtitle(self, current_time: float) -> Optional[Subtitle]:
        """Get the subtitle that should be displayed at the given time."""
        for subtitle in self.subtitles:
            if subtitle.is_active(current_time):
                return subtitle
        return None
    
    def render_subtitle(self, frame: np.ndarray, current_time: float, use_gpu: bool = True) -> np.ndarray:
        """
        Render the current subtitle on a video frame with optional GPU acceleration.
        
        Args:
            frame: Input video frame (NumPy or CuPy array)
            current_time: Current time in seconds
            use_gpu: Whether to use GPU acceleration if available
            
        Returns:
            Frame with rendered subtitle (same type as input)
        """
        subtitle = self.get_active_subtitle(current_time)
        if subtitle is None:
            return frame
        
        # Check if input is GPU array and handle accordingly
        is_gpu_input = HAS_CUPY and use_gpu
        
        # We need to work with CPU for OpenCV text rendering
        # Transfer to CPU if needed
        if is_gpu_input:
            # Remember the original array type
            cpu_frame = to_cpu(frame)
        else:
            cpu_frame = frame
        
        # Get frame dimensions
        height, width = cpu_frame.shape[:2]
        
        # Calculate text size and position
        text = subtitle.text
        progress = subtitle.get_progress(current_time)
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        # Ensure text fits within frame
        max_width = int(width * self.max_width_ratio)
        if text_width > max_width:
            # Scale down font to fit
            scale_factor = max_width / text_width
            self.font_scale *= scale_factor
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, self.font_scale, self.font_thickness
            )
        
        # Calculate position
        x = (width - text_width) // 2  # Center horizontally
        
        if self.position == "top":
            y = text_height + 30
        elif self.position == "middle":
            y = (height + text_height) // 2
        else:  # bottom
            y = height - 30
        
        # Draw background if enabled
        if self.background_color is not None:
            bg_padding = 10
            bg_x1 = x - bg_padding
            bg_y1 = y - text_height - bg_padding
            bg_x2 = x + text_width + bg_padding
            bg_y2 = y + bg_padding
            
            # Create overlay for semi-transparent background
            overlay = cpu_frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                          self.background_color, -1)
            
            # Blend with original frame
            cpu_frame = cv2.addWeighted(overlay, self.background_alpha, cpu_frame, 
                                   1 - self.background_alpha, 0)
        
        # Calculate the split position for karaoke effect
        split_pos = int(len(text) * progress)
        
        # Draw the non-highlighted part
        if split_pos < len(text):
            non_highlighted = text[split_pos:]
            nh_width, _ = cv2.getTextSize(
                text[:split_pos], self.font, self.font_scale, self.font_thickness
            )[0]
            cv2.putText(cpu_frame, non_highlighted, (x + nh_width, y), self.font, 
                       self.font_scale, self.font_color, self.font_thickness)
        
        # Draw the highlighted part
        if split_pos > 0:
            highlighted = text[:split_pos]
            cv2.putText(cpu_frame, highlighted, (x, y), self.font, 
                       self.font_scale, self.highlight_color, self.font_thickness)
        
        # Transfer back to GPU if input was GPU array
        if is_gpu_input:
            return to_gpu(cpu_frame)
        
        return cpu_frame
