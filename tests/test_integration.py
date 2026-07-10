"""Integration tests for the video generation pipeline."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import cv2

from src.video.rotation import create_rotating_video


def _create_test_image(path, size=128):
    """Create a simple test image with a circle pattern."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Draw a white circle on black background
    cv2.circle(img, (size//2, size//2), size//3, (255, 255, 255), -1)
    cv2.imwrite(path, img)
    return path


def _create_test_audio(path, duration=3.0, sr=22050):
    """Create a simple test audio with regular beats."""
    import soundfile as sf
    
    # Create a simple click track with beats at 120 BPM
    samples = int(sr * duration)
    y = np.zeros(samples, dtype=np.float32)
    
    # Add clicks every 0.5 seconds (120 BPM)
    interval = int(sr * 0.5)
    for i in range(0, samples, interval):
        if i + 100 < samples:  # Ensure we don't go out of bounds
            y[i:i+100] = np.linspace(1.0, 0.0, 100)
    
    sf.write(path, y, sr)
    return path


@pytest.mark.integration
def test_create_rotating_video():
    """Test that the video generation pipeline works end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        img_path = os.path.join(tmpdir, "test_image.png")
        audio_path = os.path.join(tmpdir, "test_audio.wav")
        output_path = os.path.join(tmpdir, "test_output.mp4")
        
        _create_test_image(img_path)
        _create_test_audio(audio_path, duration=1.0)
        
        # Verify the test files were created successfully
        assert os.path.exists(img_path)
        assert os.path.exists(audio_path)
        
        create_rotating_video(img_path, audio_path, output_path, effects=["pulse"], use_gpu=False)
        assert os.path.getsize(output_path) > 1_000

        from moviepy.video.io.VideoFileClip import VideoFileClip
        with VideoFileClip(output_path) as clip:
            assert clip.duration == pytest.approx(1.0, abs=0.1)
            assert clip.audio is not None
            assert clip.size == [128, 128]
