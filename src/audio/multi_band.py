import numpy as np
import librosa
import logging
from src.config.audio_config import FREQUENCY_BANDS
from src.audio.beat_detection import get_beats, get_beat_detection_params

def get_onset_envelope(y, sr, hop_length, fmin, fmax, n_mels, rms_threshold):
    """
    Extract onset envelope for a specific frequency band with RMS masking.
    This is a helper function that extracts the core functionality from get_beats
    without the peak picking and frame conversion.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
        n_mels: Number of mel bands
        rms_threshold: RMS threshold for masking
        
    Returns:
        Masked and normalized onset envelope
    """
    # Onset envelope for this frequency band
    onset_env = librosa.onset.onset_strength(
        y=y, 
        sr=sr, 
        hop_length=hop_length,
        fmin=fmin, 
        fmax=fmax,
        aggregate=np.median, 
        n_mels=n_mels
    )
    
    # Compute RMS and apply threshold mask
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms /= np.max(rms) + 1e-6  # Normalize
    mask = rms > rms_threshold
    
    # Apply mask to onset envelope
    masked_env = onset_env * mask
    
    # Normalize the envelope
    if np.max(masked_env) > 0:
        masked_env = masked_env / np.max(masked_env)
        
    return masked_env

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
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            raise ValueError("Audio file is empty")
            
        # Process each band
        band_envelopes = {}
        for band_info in bands:
            name, fmin, fmax, n_mels, rms_threshold = band_info
            logging.info(f"Band {name} ({fmin}-{fmax}Hz): using {n_mels} mel bands, RMS threshold {rms_threshold}")
            
            # Get onset envelope for this band
            band_env = get_onset_envelope(
                y, sr, hop_length, fmin, fmax, n_mels, rms_threshold
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
