import unittest
import numpy as np
import cv2
from src.sobel import Sobel

class TestSobel(unittest.TestCase):
    """
    Unit test class for the `Sobel` class, using Python's built-in unittest framework.

    The test will:
    - Load a synthetic grayscale image (a black square with a white rectangle).
    - Run the Sobel edge detector.
    - Assert that the output:
        -Is the same shape as the input.
        - Has only expected edges.
        - Contains some non-zero edges if edges are expected
        - Contains no edges in response to an all-zero image
    """
    def setUp(self):
        """Create synthetic test images."""
        # A 5x5 black image with a white square in the center
        self.image = np.zeros((5, 5), dtype=np.float32)
        self.image[1:4, 1:4] = 255  # White square in the center

    def test_output_shape(self):
        """Check that the output shape matches the input."""
        sobel = Sobel(self.image)
        output = sobel.find_edges()
        self.assertEqual(output.shape, self.image.shape, "Output shape should match input shape.")

    def test_detects_edges(self):
        """Check that edges are detected for a known case."""
        sobel = Sobel(self.image)
        output = sobel.find_edges()
        self.assertGreater(np.sum(output > 0), 0, "There should be nonzero values representing edges.")

    def test_all_zeros_input(self):
        """An all-black image should return an all-zero edge image."""
        black_image = np.zeros((5, 5), dtype=np.float32)
        sobel = Sobel(black_image)
        output = sobel.find_edges()
        self.assertTrue(np.all(output == 0), "An all-black image should produce no edges.")

    def test_single_bright_pixel(self):
        """A single bright pixel should produce a minimal gradient response."""
        img = np.zeros((5, 5), dtype=np.float32)
        img[2, 2] = 255  # A single bright pixel
        sobel = Sobel(img)
        output = sobel.find_edges()
        self.assertGreater(np.sum(output > 0), 0, "A single bright pixel should produce some gradient response.")

    def test_output_value_range(self):
        """Ensure the output values are within a reasonable range."""
        sobel = Sobel(self.image)
        output = sobel.find_edges()
        self.assertTrue(np.all(output >= 0), "Sobel magnitude should be non-negative.")