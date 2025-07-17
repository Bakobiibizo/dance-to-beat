"""
Configuration settings for audio processing and beat detection.
"""

# General audio processing settings
SAMPLE_RATE = 44100  # Hz
HOP_LENGTH = 512     # Number of samples between successive frames
N_FFT = 2048         # Size of the Fast Fourier Transform (FFT) window. The FFT is an algorithm for efficiently calculating the discrete Fourier transform (DFT) of a sequence. It is commonly used to convert a signal from the time domain to the frequency domain.

# Beat detection settings
BEAT_DELTA = 0.02    # Threshold for beat detection
BEAT_WAIT = 1        # Minimum number of frames between beats
BEAT_MIN_TEMPO = 60  # Minimum BPM to consider
BEAT_MAX_TEMPO = 180 # Maximum BPM to consider

# RMS energy threshold for masking silent sections
RMS_THRESHOLD = 0.4 # Normalized RMS threshold (0-1)

# Frequency bands (Hz) with per-band settings
# Format: (name, min_freq, max_freq, n_mels, rms_threshold)
FREQUENCY_BANDS = [
    ("bass", 50, 250, 2, 0.15),     # Bass frequencies (most sensitive)
    ("low_mid", 250, 1200, 4, 0.25),  # Low-mid frequencies
    ("high_mid", 1200, 2500, 8, 0.35), # High-mid frequencies
    ("high", 2500, 5000, 16, 0.45),    # High frequencies (least sensitive)
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
BAND_SPEEDS = [0.002, 0.004, 0.008, 0.016]
    

# Pulse effect settings
PULSE_MIN_SCALE = 1.0   # Minimum scale factor
PULSE_MAX_SCALE = 1.5   # Maximum scale factor
PULSE_INTENSITY = 0.6   # Intensity of pulse effect (higher = stronger pulse)

# Beat marker effect settings
MARKER_INTENSITY = 0.6  # Intensity of beat marker flash (higher = brighter)
MARKER_DECAY = 0.3      # Decay rate of beat marker flash (higher = faster decay)

# Edge detection effect settings
EDGE_INTENSITY = 0.7     # Blend intensity for edge effect (0.0-1.0)
EDGE_LOW_THRESHOLD = 50  # Lower threshold for Canny edge detection
EDGE_HIGH_THRESHOLD = 150 # Higher threshold for Canny edge detection

# Subtitle settings
SUBTITLE_FONT_SCALE = 1.4       # Font size scale factor (increased by ~40%)
SUBTITLE_FONT_THICKNESS = 2     # Font thickness
SUBTITLE_FONT_COLOR = (255, 255, 255)       # White text
SUBTITLE_HIGHLIGHT_COLOR = (255, 255, 0)    # Yellow highlight
SUBTITLE_BG_COLOR = (0, 0, 0)               # Black background
SUBTITLE_BG_ALPHA = 0.5                     # Background opacity
SUBTITLE_POSITION = "bottom"                # Vertical position
SUBTITLE_MAX_WIDTH_RATIO = 0.8              # Maximum width as ratio of frame width

# Loop settings
LOOP_FRAMES = 240
BAND_SPEEDS = [1 / LOOP_FRAMES * f for f in [1, 2, 3, 4]]

OUTPUT_PATH = "output/"