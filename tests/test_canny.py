import unittest
import numpy as np
import cv2
from canny import Canny

class TestCannyEdgeDetection(unittest.TestCase):
"""
The test will:
- Load a synthetic grayscale image (a black square with a white rectangle).
- Run your Canny edge detector.Assert that the output:
    -Is the same shape as the input.
    - Has only expected edge intensity values (0, weak, strong).
    - Contains some non-zero edges if edges are expected? this is good enough?
"""
    def setUp(self):
        # Create a simple synthetic image: black background with white rectangle
        self.img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(self.img, (30, 30), (70, 70), 255, -1)

    def test_output_shape(self):
        canny = Canny(self.img)
        output = canny.canny()
        self.assertEqual(output.shape, self.img.shape, "Output shape should match input image")

    def test_output_values(self):
        canny = Canny(self.img)
        output = canny.canny()
        unique_vals = np.unique(output)
        # Should only contain 0, weak (50), or strong (255)
        for val in unique_vals:
            self.assertIn(val, [0, 50, 255], f"Unexpected pixel value {val} in output")

    def test_detects_edges(self):
        canny = Canny(self.img)
        output = canny.canny()
        num_edges = np.sum(output > 0)
        self.assertGreater(num_edges, 0, "Should detect some edges in synthetic image")

if __name__ == '__main__':
    unittest.main()
