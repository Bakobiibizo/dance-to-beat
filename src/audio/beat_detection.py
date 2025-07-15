import numpy as np
import librosa
import logging


def get_beats(
    audio_path,
    sr=44100,
    hop_length=512,
    fmin=20,
    fmax=200,
    n_mels=4,
    rms_threshold=0.4,
    fps=30
):
    """
    Detect beat-aligned frame indices (~30fps) based on low-frequency onset envelope.

    Args:
        audio_path (str): Path to audio file
        sr (int): Sample rate for loading
        hop_length (int): STFT hop length
        fmin/fmax (int): Frequency band for onset envelope
        n_mels (int): Mel bands to use (keep low for narrow bands)
        rms_threshold (float): Silence masking threshold
        fps (int): Output frame rate

    Returns:
        beat_frames (List[int]): 30fps-aligned frame indices where beats occur
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            raise ValueError("Audio file is empty")

        # Onset envelope in low frequencies
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

        # Peak picking on masked envelope
        peaks = librosa.util.peak_pick(
            masked_env, 
            pre_max=10, 
            post_max=10,
            pre_avg=50, 
            post_avg=50,
            delta=0.1, 
            wait=30
        )
        
        # Convert to time and then to frame numbers at specified fps
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        beat_frames = np.round(peak_times * fps).astype(int)

        logging.info(f"Detected {len(beat_frames)} beats")
        return beat_frames

    except Exception as e:
        logging.error(f"Beat detection failed: {e}")
        return []