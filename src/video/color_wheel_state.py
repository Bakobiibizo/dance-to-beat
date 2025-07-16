import logging
from src.config.audio_config import BAND_STARTING_POSITIONS, COLOR_WHEEL_COLORS

class ColorWheelState:
    def __init__(self):
        self.start_positions = []
        for band_color in BAND_STARTING_POSITIONS:
            found = False
            for i, wheel_color in enumerate(COLOR_WHEEL_COLORS):
                if wheel_color == band_color:
                    pos = i / len(COLOR_WHEEL_COLORS)
                    self.start_positions.append(pos)
                    found = True
                    break
            if not found:
                logging.warning(f"No match for {band_color}, falling back to default position")
                fallback_map = {
                    (139, 32, 186): 0.5,  # Purple
                    (0, 179, 212): 0.75,  # Blue
                    (255, 251, 0): 0.0,   # Yellow
                    (255, 126, 1): 0.25   # Orange
                }
                self.start_positions.append(fallback_map.get(band_color, 0.0))

        self.positions = self.start_positions.copy()
        self.speeds = [0.002, 0.004, 0.008, 0.016]
        self.frame_count = 0

    def update(self):
        self.positions = [(p + s) % 1.0 for p, s in zip(self.positions, self.speeds)]
        self.frame_count += 1

if __name__ == "__main__":
    state = ColorWheelState()
    for i in range(10):
        print(f"Frame {i}: {state.positions}")
        state.update()