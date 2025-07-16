import numpy as np
from src.audio.multi_band import get_multi_band_envelopes


def _generate_sine_mix(sr: int, duration: float):
    """Generate a mix of sine waves in different bands."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Bass 60Hz, low_mid 300Hz, high_mid 1000Hz, high 4000Hz
    y = (
        0.5 * np.sin(2 * np.pi * 60 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.2 * np.sin(2 * np.pi * 1000 * t) +
        0.1 * np.sin(2 * np.pi * 4000 * t)
    ).astype(np.float32)
    return y


def test_multi_band_envelopes(tmp_path):
    sr = 22050
    duration = 2.0
    y = _generate_sine_mix(sr, duration)
    wav_path = tmp_path / "mix.wav"

    # save using librosa
    import soundfile as sf
    sf.write(wav_path, y, sr)

    envelopes, _ = get_multi_band_envelopes(str(wav_path))
    # Expect four default bands present
    assert len(envelopes) == 4
    for env in envelopes.values():
        assert len(env) > 0
        assert np.max(env) <= 1.0 + 1e-6
