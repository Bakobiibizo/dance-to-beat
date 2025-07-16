import unittest
import numpy as np
import pytest

pytest.skip("Legacy detect_bass API removed; skipping obsolete test.", allow_module_level=True)

class TestAudioFunctions(unittest.TestCase):
    def test_detect_bass(self):
        # Create a simple test signal
        sr = 44100
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Generate a simple sine wave at 60 Hz (bass frequency)
        y = 0.5 * np.sin(2 * np.pi * 60 * t)
        
        # Test the detect_bass function
        onset_env = detect_bass(y, sr)
        
        # Basic checks
        self.assertIsNotNone(onset_env)
        self.assertTrue(len(onset_env) > 0)
        self.assertTrue(np.all(onset_env >= 0))
        self.assertTrue(np.all(onset_env <= 1))

if __name__ == '__main__':
    unittest.main()
