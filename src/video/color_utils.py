from src.config.audio_config import COLOR_WHEEL_COLORS, STARTING_COLORS, BAND_STARTING_POSITIONS, BAND_SPEEDS
import numpy as np
import logging

def find_color_index(color_list, target_color):
    try:
        return color_list.index(target_color)
    except ValueError:
        raise ValueError(f"Color {target_color} not found in wheel.")

def calculate_normalized_positions(starting_colors, wheel_colors):
    wheel_size = len(wheel_colors)
    positions = []
    for color in starting_colors:
        idx = find_color_index(wheel_colors, color)
        position = idx / wheel_size
        positions.append(round(position, 4))  # Round for readability
    return positions

def get_band_positions(frame_number):
    return [(start + frame_number * speed) % 1.0 for start, speed in zip(BAND_STARTING_POSITIONS, BAND_SPEEDS)]

def interpolate_color_wheel(colors, position):
    position %= 1.0
    wheel_pos = position * len(colors)
    idx1 = int(wheel_pos) % len(colors)
    idx2 = (idx1 + 1) % len(colors)
    weight = wheel_pos - int(wheel_pos)
    c1 = np.array(colors[idx1])
    c2 = np.array(colors[idx2])
    return tuple((c1 * (1 - weight) + c2 * weight).astype(int))

class ColorWheelCache:
    """
    A cache for precomputed color wheel positions to optimize performance.
    Instead of recalculating interpolated colors for each frame, we precompute
    them for all possible positions and store them in a lookup table.
    """
    def __init__(self, colors=COLOR_WHEEL_COLORS, precision=1000):
        """
        Initialize the color wheel cache.
        
        Args:
            colors: List of RGB tuples defining the color wheel
            precision: Number of positions to precompute (higher = more accurate but more memory)
        """
        self.colors = colors
        self.precision = precision
        self.cache = {}
        self._build_cache()
        logging.info(f"Color wheel cache built with {len(self.cache)} entries at precision {precision}")
    
    def _build_cache(self):
        """
        Build the color wheel cache by precomputing interpolated colors
        for all possible positions at the specified precision.
        """
        for i in range(self.precision + 1):
            position = i / self.precision
            self.cache[position] = interpolate_color_wheel(self.colors, position)
    
    def get_color(self, position):
        """
        Get the interpolated color for a given position from the cache.
        
        Args:
            position: A float between 0 and 1 representing the position on the color wheel
            
        Returns:
            RGB tuple of the interpolated color
        """
        # Normalize position to 0-1 range
        position = position % 1.0
        
        # Find the closest cached position
        cache_position = round(position * self.precision) / self.precision
        
        # Return the cached color
        return self.cache.get(cache_position, interpolate_color_wheel(self.colors, position))


if __name__ == "__main__":
    positions = calculate_normalized_positions(STARTING_COLORS, COLOR_WHEEL_COLORS)
    print("Normalized color wheel positions:")
    for i, pos in enumerate(positions):
        print(f"Band {i+1}: {pos}")

    band_positions = get_band_positions(0)
    print("Band positions at frame 0:")
    for i, pos in enumerate(band_positions):
        print(f"Band {i+1}: {pos}")

    # Test the standard interpolation
    colors = interpolate_color_wheel(COLOR_WHEEL_COLORS, band_positions[0])
    print("Colors for band 1 at frame 0 (standard):", colors)
    
    # Test the cached interpolation
    cache = ColorWheelCache()
    cached_colors = cache.get_color(band_positions[0])
    print("Colors for band 1 at frame 0 (cached):", cached_colors)
    