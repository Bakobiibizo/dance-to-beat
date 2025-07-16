import numpy as np
import pytest
from src.audio.audio_utils import extract_onset_envelope


def _generate_click_track(sr: int, duration: float, bpm: float):
    """Generate a simple click track (impulse train) to simulate beats."""
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)
    interval = int(sr * 60.0 / bpm)
    y[::interval] = 1.0  # set impulses
    return y


def test_extract_onset_envelope_basic():
    sr = 22050
    duration = 5.0
    bpm = 120
    hop_length = 512
    y = _generate_click_track(sr, duration, bpm)

    envelope = extract_onset_envelope(
        y,
        sr,
        hop_length=hop_length,
        fmin=20,
        fmax=500,
        n_mels=4,
        rms_threshold=0.0,  # keep all
    )

    # Envelope length should match librosa frames count
    expected_len = int(np.ceil(len(y) / hop_length))
    assert len(envelope) == expected_len

    # Impulse track should yield non-zero envelope values at some frames
    assert np.max(envelope) > 0.0

    # Envelope should be normalised to [0,1]
    assert np.max(envelope) <= 1.0 + 1e-6
