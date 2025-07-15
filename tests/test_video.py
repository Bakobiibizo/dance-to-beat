import unittest
import os
import cv2
import numpy as np
from src.video.rotation import FrameGenerator

class TestVideoFunctions(unittest.TestCase):
    def setUp(self):
        # Create a simple test image
        self.test_image_path = "test_image.png"
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        # Add a simple pattern to make rotation visible
        img[40:60, 40:60] = [0, 0, 255]  # Red square in the middle
        cv2.imwrite(self.test_image_path, img)
        
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_frame_generator(self):
        # Test the FrameGenerator class
        frame_gen = FrameGenerator(self.test_image_path)
        
        # Basic checks
        self.assertIsNotNone(frame_gen.image)
        self.assertEqual(frame_gen.image.shape, (100, 100, 3))
        # Check that the image was converted to RGB (OpenCV loads as BGR)
        self.assertTrue(np.array_equal(frame_gen.image[40:60, 40:60, 0], np.ones((20, 20)) * 255))  # R channel

if __name__ == '__main__':
    unittest.main()
