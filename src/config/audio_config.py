"""
Configuration settings for audio processing and beat detection.
"""

# General audio processing settings
SAMPLE_RATE = 44100  # Hz
HOP_LENGTH = 512     # Number of samples between successive frames
N_FFT = 2048         # FFT window size

# Beat detection settings
BEAT_DELTA = 0.03    # Threshold for beat detection
BEAT_WAIT = 1        # Minimum number of frames between beats
BEAT_MIN_TEMPO = 60  # Minimum BPM to consider
BEAT_MAX_TEMPO = 180 # Maximum BPM to consider

# RMS energy threshold for masking silent sections
RMS_THRESHOLD = 0.12 # Normalized RMS threshold (0-1)

# Frequency bands (Hz) with per-band settings
# Format: (name, min_freq, max_freq, n_mels, rms_threshold)
FREQUENCY_BANDS = [
    ("bass", 20, 100, 2, 0.4),       # Bass frequencies (most sensitive)
    ("low_mid", 99, 299, 4, 0.4),  # Low-mid frequencies
    ("high_mid", 800, 1399, 8, 0.5), # High-mid frequencies
    ("high", 1400, 5000, 16, 0.6)    # High frequencies (least sensitive)
]

# Visualization settings
BAND_BASE_SCALES = [1.0, 0.75, 0.5, 0.25]  # Base scales for concentric circles (outer to inner)
STARTING_COLORS = [
    (139, 32, 186),    # Purple (bass)
    (0, 179, 212),    # Blue (low-mid)
    (255, 251, 0),    # Yellow (high-mid)
    (255, 126, 1)   # Orange (high)
]

# Color wheel
COLOR_WHEEL_COLORS = [
    (255, 251, 0), # North
    (255, 207, 0),
    (255, 168, 0),
    (255, 126, 1), # East
    (255, 33, 1),
    (255, 34, 149),
    (139, 32, 186), # South
    (0, 35, 185),
    (0, 122, 199),
    (0, 179, 212), # West
    (0, 184, 1),
    (132, 206, 1),
]

BAND_STARTING_POSITIONS = [0.5, 0.75, 0.0, 0.25]
BAND_SPEEDS = [0.003, 0.004, 0.005, 0.006]
    

# Pulse effect settings
PULSE_MIN_SCALE = 1.0   # Minimum scale factor
PULSE_MAX_SCALE = 1.5   # Maximum scale factor
PULSE_INTENSITY = 0.4   # Intensity of pulse effect (higher = stronger pulse)

# Beat marker effect settings
MARKER_INTENSITY = 0.5  # Intensity of beat marker flash (higher = brighter)
MARKER_DECAY = 0.3      # Decay rate of beat marker flash (higher = faster decay)

# Loop settings
LOOP_FRAMES = 240
BAND_SPEEDS = [1 / LOOP_FRAMES * f for f in [1, 2, 3, 4]]

OUTPUT_PATH = "output/"