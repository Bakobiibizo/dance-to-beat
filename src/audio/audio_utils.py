"""Audio Utility Functions for Beat Detection and Envelope Extraction.

This module provides shared audio processing utilities used across the library,
including functions for loading audio files and extracting onset envelopes
for beat detection and visualization.

The main functions include:
- load_audio: Load and normalize audio from various file formats
- extract_onset_envelope: Extract a normalized onset strength envelope from audio

These utilities are used by both the beat_detection and multi_band modules
to provide consistent audio processing behavior.

GPU acceleration is used when available through CuPy.
"""

from typing import Tuple, Union, Optional

import librosa
import numpy as np

# Import GPU utilities
from src.utils.gpu_utils import to_gpu, to_cpu, HAS_CUPY, get_array_module
from src.utils.logging_config import setup_logging

# Initialize logger
logger = setup_logging()

# General-purpose helpers shared by beat_detection and multi_band modules.
# Keeping them here avoids duplicated logic and keeps audio-processing concerns
# in one location.

def load_audio(audio_path: str, sr: int = 22050, use_gpu: bool = True) -> Tuple[Union[np.ndarray, 'cp.ndarray'], int]:
    """Load and normalize audio from various file formats.
    
    Args:
        audio_path (str): Path to the audio file (mp3, wav, etc.)
        sr (int, optional): Target sample rate. Defaults to 22050.
        use_gpu (bool, optional): Whether to transfer audio data to GPU. Defaults to True.
        
    Returns:
        tuple: (audio_data, sample_rate) where audio_data is a numpy or cupy array
              normalized to [-1, 1] and sample_rate is the actual sample rate.
              Returns (None, None) if loading fails.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            raise ValueError("Audio file is empty or could not be read: %s" % audio_path)
            
        # Transfer to GPU if available and requested
        if use_gpu and HAS_CUPY:
            logger.debug(f"Transferring audio data to GPU (length: {len(y)})")
            y = to_gpu(y)
            
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return None, None


def extract_onset_envelope(
    y: Union[np.ndarray, 'cp.ndarray'],
    sr: int,
    *,
    hop_length: int = 512,
    fmin: float = 20,
    fmax: float = 8000,
    n_mels: int = 128,
    rms_threshold: float = 0.01,
    use_gpu: bool = True,
) -> Union[np.ndarray, 'cp.ndarray']:
    """Extract a normalized onset strength envelope from audio data.
    
    This function computes a mel spectrogram, converts it to log scale,
    extracts onset strength, and applies RMS-based masking to reduce noise
    in quiet sections. The result is normalized to the range [0, 1].
    
    GPU acceleration is used when available and requested.
    
    Args:
        y (np.ndarray or cp.ndarray): Audio time series
        sr (int): Sample rate
        hop_length (int, optional): Hop length for STFT. Defaults to 512.
        fmin (int, optional): Minimum frequency for mel filterbank. Defaults to 20.
        fmax (int, optional): Maximum frequency for mel filterbank. Defaults to 8000.
        n_mels (int, optional): Number of mel bands. Defaults to 128.
        rms_threshold (float, optional): Threshold for RMS masking as a fraction
                                        of max RMS. Defaults to 0.01.
        use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to True.
    
    Returns:
        np.ndarray or cp.ndarray: Normalized onset envelope in range [0, 1]
    """
    # Determine if input is on GPU
    is_gpu_input = HAS_CUPY and use_gpu and hasattr(y, 'device')
    
    # If input is on GPU, transfer to CPU for librosa processing
    if is_gpu_input:
        logger.debug("Transferring audio data from GPU to CPU for librosa processing")
        cpu_y = to_cpu(y)
    else:
        cpu_y = y
    
    # Compute onset strength restricted to a frequency range.
    logger.debug(f"Computing onset strength with mel bands: {n_mels}, freq range: {fmin}-{fmax} Hz")
    onset_env = librosa.onset.onset_strength(
        y=cpu_y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        aggregate=np.median,
        n_mels=n_mels,
    )

    # Energy masking via RMS (normalised).
    rms = librosa.feature.rms(y=cpu_y, hop_length=hop_length)[0]
    
    # Transfer results to GPU if requested
    if HAS_CUPY and use_gpu:
        logger.debug("Transferring onset envelope and RMS data to GPU for processing")
        onset_env = to_gpu(onset_env)
        rms = to_gpu(rms)
    
    # Get the appropriate array module (np or cp)
    xp = get_array_module(onset_env)
    
    # Apply normalization and masking on GPU if available
    rms_norm = rms / (xp.max(rms) + 1e-6)
    mask = rms_norm > rms_threshold
    masked_env = onset_env * mask

    # Normalise if non-zero.
    max_val = xp.max(masked_env)
    if max_val > 0:
        masked_env = masked_env / max_val

    return masked_env
