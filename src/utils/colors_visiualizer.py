import numpy as np
import json
import matplotlib.pyplot as plt

# 12-point color wheel (as RGB tuples)
COLOR_WHEEL = [
    (255, 251, 0), (255, 207, 0), (255, 168, 0), (255, 126, 1),
    (255, 33, 1), (255, 34, 149), (139, 32, 186), (0, 35, 185),
    (0, 122, 199), (0, 179, 212), (0, 184, 1), (132, 206, 1)
]

# Starting band positions and speeds
start_positions = [0.5, 0.75, 0.0, 0.25]   # Purple, Blue, Yellow, Orange
speeds = [0.002, 0.004, 0.008, 0.016]      # independent rotation rates
frame_count = 100
num_bands = len(start_positions)

# Interpolation function
def interpolate_color(colors, position):
    position %= 1.0
    wheel_pos = position * len(colors)
    idx1 = int(wheel_pos) % len(colors)
    idx2 = (idx1 + 1) % len(colors)
    weight = wheel_pos - int(wheel_pos)
    c1 = np.array(colors[idx1])
    c2 = np.array(colors[idx2])
    return tuple((c1 * (1 - weight) + c2 * weight).astype(int))

# Compute RGB colors for each band over each frame
color_array = np.zeros((frame_count, num_bands, 3), dtype=int)

for frame in range(frame_count):
    for band in range(num_bands):
        pos = (start_positions[band] + frame * speeds[band]) % 1.0
        rgb = interpolate_color(COLOR_WHEEL, pos)
        color_array[frame, band] = rgb

# Save as .npy for use in render loop
np.save("output/band_color_transitions.npy", color_array)

# Save as .json for inspection
with open("output/band_color_transitions.json", "w") as f:
    json.dump(color_array.tolist(), f, indent=2)

# Optional: Show color transitions with matplotlib
plt.figure(figsize=(10, 3))
for band in range(num_bands):
    band_colors = color_array[:, band] / 255.0
    plt.imshow([band_colors], aspect='auto', extent=[0, frame_count, band, band + 1])
plt.yticks(np.arange(0.5, num_bands + 0.5), [f"Band {i+1}" for i in range(num_bands)])
plt.xlabel("Frame")
plt.title("Color Wheel Positions Over Time")
plt.tight_layout()
plt.savefig("output/band_color_transitions.png")
# plt.show()  # Uncomment to preview if running locally with a GUI
