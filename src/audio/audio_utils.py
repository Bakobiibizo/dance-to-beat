"""Audio Utility Functions for Beat Detection and Envelope Extraction.

This module provides shared audio processing utilities used across the library,
including functions for loading audio files and extracting onset envelopes
for beat detection and visualization.

The main functions include:
- load_audio: Load and normalize audio from various file formats
- extract_onset_envelope: Extract a normalized onset strength envelope from audio

These utilities are used by both the beat_detection and multi_band modules
to provide consistent audio processing behavior.
"""

import logging
from typing import Tuple

import librosa
import numpy as np

# General-purpose helpers shared by beat_detection and multi_band modules.
# Keeping them here avoids duplicated logic and keeps audio-processing concerns
# in one location.

def load_audio(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Load and normalize audio from various file formats.
    
    Args:
        audio_path (str): Path to the audio file (mp3, wav, etc.)
        sr (int, optional): Target sample rate. Defaults to 22050.
        
    Returns:
        tuple: (audio_data, sample_rate) where audio_data is a numpy array
              normalized to [-1, 1] and sample_rate is the actual sample rate.
              Returns (None, None) if loading fails.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            raise ValueError("Audio file is empty or could not be read: %s" % audio_path)
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


def extract_onset_envelope(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    fmin: float = 20,
    fmax: float = 8000,
    n_mels: int = 128,
    rms_threshold: float = 0.01,
) -> np.ndarray:
    """Extract a normalized onset strength envelope from audio data.
    
    This function computes a mel spectrogram, converts it to log scale,
    extracts onset strength, and applies RMS-based masking to reduce noise
    in quiet sections. The result is normalized to the range [0, 1].
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        hop_length (int, optional): Hop length for STFT. Defaults to 512.
        fmin (int, optional): Minimum frequency for mel filterbank. Defaults to 20.
        fmax (int, optional): Maximum frequency for mel filterbank. Defaults to 8000.
        n_mels (int, optional): Number of mel bands. Defaults to 128.
        rms_threshold (float, optional): Threshold for RMS masking as a fraction
                                        of max RMS. Defaults to 0.01.
    
    Returns:
        np.ndarray: Normalized onset envelope in range [0, 1]
    """
    # Compute onset strength restricted to a frequency range.
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        aggregate=np.median,
        n_mels=n_mels,
    )

    # Energy masking via RMS (normalised).
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = rms / (np.max(rms) + 1e-6)
    mask = rms_norm > rms_threshold
    masked_env = onset_env * mask

    # Normalise if non-zero.
    max_val = np.max(masked_env)
    if max_val > 0:
        masked_env = masked_env / max_val

    return masked_env
