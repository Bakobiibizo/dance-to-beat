"""Multi-band Audio Envelope Extraction Module.

This module provides functions for extracting onset envelopes from multiple
frequency bands of an audio file. These envelopes can be used for creating
frequency-sensitive visualizations that respond differently to bass, mid, and
high frequency content.

The main functions include:
- get_multi_band_envelopes: Extract onset envelopes for multiple frequency bands

This module uses the shared audio utilities for loading audio files and
extracting onset envelopes, focusing on the specific task of separating
the audio signal into multiple frequency bands for visualization.
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple

from src.config.audio_config import SAMPLE_RATE, HOP_LENGTH, RMS_THRESHOLD, FREQUENCY_BANDS
from src.audio.audio_utils import load_audio, extract_onset_envelope
from src.audio.beat_detection import get_beat_detection_params

def get_multi_band_envelopes(audio_path, bands=FREQUENCY_BANDS, **kwargs):
    """
    Extract onset envelopes for multiple frequency bands.
    Uses the same core detection logic as get_beats but for multiple frequency bands.
    
    Args:
        audio_path (str): Path to audio file
        bands (list): List of frequency bands to process
        **kwargs: Override any default parameters
        
    Returns:
        tuple: (dict of band envelopes, audio time series)
    """
    try:
        # Get default parameters
        params = get_beat_detection_params()
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        # Extract parameters
        sr = params['sr']
        hop_length = params['hop_length']
        
        # Load audio
        y, _ = load_audio(audio_path, sr)
        if len(y) == 0:
            raise ValueError("Audio file is empty")
            
        # Process each band using shared envelope extraction utility
        band_envelopes = {}
        for band_info in bands:
            name, fmin, fmax, n_mels, rms_threshold = band_info
            logging.info(f"Band {name} ({fmin}-{fmax}Hz): using {n_mels} mel bands, RMS threshold {rms_threshold}")
            
            # Get onset envelope for this band
            band_env = extract_onset_envelope(
                y,
                sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                n_mels=n_mels,
                rms_threshold=rms_threshold,
            )
            
            band_envelopes[name] = band_env
            
        logging.info(f"Extracted onset envelopes for {len(band_envelopes)} frequency bands")
        return band_envelopes, y
        
    except Exception as e:
        logging.error(f"Multi-band envelope extraction failed: {e}")
        # Return empty envelopes
        empty_envelopes = {}
        for band in bands:
            name = band[0]  # Name is always the first element
            empty_envelopes[name] = np.array([])
            
        return empty_envelopes, None
