import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def visualize_beats(audio_path, sr=44100, hop_length=512, fmin=20, fmax=100, fps=30):
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Onset envelope in low frequencies
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        aggregate=np.median,
        n_mels=2
    )

    # Compute RMS and apply threshold mask
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms /= np.max(rms) + 1e-6  # Normalize
    mask = rms > 0.55

    # Apply mask to onset envelope
    masked_env = onset_env * mask

    # Peak picking on masked envelope
    peaks = librosa.util.peak_pick(
        masked_env, pre_max=10, post_max=10,
        pre_avg=50, post_avg=50,
        delta=0.1, wait=30
    )
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    peak_frames_30fps = np.round(peak_times * fps).astype(int)

    # Plot everything
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1. Envelope + mask
    ax[0].plot(times, onset_env, label='Onset Envelope (Bass)', alpha=0.6)
    ax[0].plot(times, masked_env, label='Masked Envelope', color='orange')
    ax[0].vlines(peak_times, 0, np.max(masked_env), color='red', linestyle='--', alpha=0.7, label='Detected Beats')
    ax[0].set(title="Onset Envelope (20–150Hz) with Beat Detection", ylabel="Amplitude")
    ax[0].legend(loc="upper right")

    # 2. RMS energy
    ax[1].plot(times, rms, label="RMS Energy")
    ax[1].axhline(0.02, color='gray', linestyle='--', label='RMS Threshold')
    ax[1].set(ylabel="Normalized RMS", title="Energy Mask")
    ax[1].legend(loc="upper right")

    # 3. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[2])
    ax[2].set(title="Log-Frequency Spectrogram")
    fig.colorbar(img, ax=ax[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig("output/beat_visualization.png")

    return peak_frames_30fps


if __name__ == "__main__":
    visualize_beats("media/fleeting_drop2.wav")