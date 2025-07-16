"""Beat Detection Module for Audio Visualization.

This module provides functions for detecting beats in audio files,
particularly focused on finding strong beats in the bass frequency range
that are suitable for synchronizing visual effects.

The main functions include:
- get_beats: Detect beat-aligned frame indices (~30fps) based on low-frequency onset envelope.

This module uses the shared audio utilities for loading audio files and
extracting onset envelopes, focusing on the specific task of identifying
beat locations for visualization purposes.
"""

import numpy as np
import librosa
import logging
from src.config.audio_config import SAMPLE_RATE, HOP_LENGTH, RMS_THRESHOLD, BEAT_DELTA, BEAT_WAIT, FREQUENCY_BANDS
from src.audio.audio_utils import load_audio, extract_onset_envelope

def get_beat_detection_params(band_name=None):
    """
    Get centralized beat detection parameters.
    
    Args:
        band_name (str, optional): If provided, returns parameters for specific frequency band.
                                  If None, returns default parameters.
    
    Returns:
        dict: Dictionary of beat detection parameters
    """
    # Default parameters
    params = {
        'sr': SAMPLE_RATE,
        'hop_length': HOP_LENGTH,
        'rms_threshold': RMS_THRESHOLD,
        'beat_delta': BEAT_DELTA,
        'beat_wait': BEAT_WAIT,
        'fps': 30,  # Default frame rate
        'fmin': 20,
        'fmax': 200,
        'n_mels': 4
    }
    
    # If a specific band is requested, override with band-specific parameters
    if band_name:
        for band in FREQUENCY_BANDS:
            name, min_freq, max_freq, n_mels, rms_threshold = band
            if name == band_name:
                params['fmin'] = min_freq
                params['fmax'] = max_freq
                params['n_mels'] = n_mels
                params['rms_threshold'] = rms_threshold
                break
    
    return params


def get_beats(audio_path, band_name=None, fps=30, **kwargs):
    """
    Detect beat-aligned frame indices (~30fps) based on low-frequency onset envelope.

    Args:
        audio_path (str): Path to audio file
        band_name (str, optional): Name of frequency band to use for detection
        fps (int): Output frame rate
        **kwargs: Override any default parameters

    Returns:
        beat_frames (List[int]): 30fps-aligned frame indices where beats occur
    """
    # Get default parameters, potentially for a specific band
    params = get_beat_detection_params(band_name)
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Extract parameters
    sr = params['sr']
    hop_length = params['hop_length']
    fmin = params['fmin']
    fmax = params['fmax']
    n_mels = params['n_mels']
    rms_threshold = params['rms_threshold']
    beat_delta = params['beat_delta']
    beat_wait = params['beat_wait']
    try:
        # Load audio and extract masked onset envelope using shared utilities
        y, _ = load_audio(audio_path, sr)
        masked_env = extract_onset_envelope(
            y,
            sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_mels=n_mels,
            rms_threshold=rms_threshold,
        )

        # Find peaks in onset envelope (beats)
        peaks = librosa.util.peak_pick(
            masked_env,
            pre_max=1,
            post_max=1,
            pre_avg=1,
            post_avg=1,
            delta=beat_delta,  # Threshold for peak detection
            wait=beat_wait     # Minimum samples between peaks
        )
        
        # Convert to time and then to frame numbers at specified fps
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        beat_frames = np.round(peak_times * fps).astype(int)

        logging.info(f"Detected {len(beat_frames)} beats")
        return beat_frames

    except Exception as e:
        logging.error(f"Beat detection failed: {e}")
        return []