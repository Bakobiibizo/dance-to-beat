import librosa
import logging
import numpy as np

def detect_bpm(audio_path, sr=44100, freq_min=20, freq_max=150, hop_length=512):
    """
    Estimate BPM based on low-frequency onset envelope.

    Returns:
        bpm: Tempo in beats per minute (float)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            raise ValueError("Empty audio file")

        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=freq_min,
            fmax=freq_max,
            aggregate=np.median,
            n_mels=40
        )

        bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)[0]
        return float(bpm)

    except Exception as e:
        logging.error(f"BPM detection failed: {e}")
        return 1  # Fallback default
