import numpy as np

"""
Code adapted from: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py
    Modified by Emon Sarker (emondsarker) on March 26, 2025
    Changes: 
    - Kernel Definition: The Sobel kernels (sobelx and sobely) have been replaced with the 2x2 Roberts Cross kernels (robertsx and robertsy).
    - Convolution Operation: The Sobel filter operates on 3x3 patches, whereas the Roberts filter operates on 2x2 patches.
    - Loop Range: The loop for applying the filter is adjusted to iterate over smaller 2x2 regions instead of 3x3 regions.
    - Edge Magnitude Calculation: In the find_edges method, the gradient magnitudes are calculated based on the Roberts filter's results using the 2x2 kernels.
"""

class Roberts:
    """
    Class implementing the Roberts Cross algorithm for edge detection

    The Roberts Cross operator performs a simple, quick to compute, 2-D spatial gradient measurement on an image.
    It was one of the first edge detectors and was initially proposed by Lawrence Roberts in 1963.
    
    The operator consists of a pair of 2×2 convolution kernels:
    Gx = [[ 1, 0],      Gy = [[ 0, -1],
          [ 0,-1]]            [ 1,  0]]
    """

    def __init__(self, image):
        """
        Initialize Roberts Cross algorithm parameters
        
        Parameters:
        image: array representation of the input image in greyscale
        """
        # Roberts Cross kernels for x and y directions
        self.robertsx = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.robertsy = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.image = image

    def find_edges(self):
        """
        Apply Roberts Cross operator to detect edges in the image.
        
        Returns:
        numpy array: Edge-detected image
        """
        [rows, columns] = np.shape(self.image) 
        roberts_filtered_image = np.zeros(shape=(rows, columns))

        for i in range(rows - 1): 
            for j in range(columns - 1):
                gx = np.sum(np.multiply(self.robertsx, self.image[i:i + 2, j:j + 2]))  
                gy = np.sum(np.multiply(self.robertsy, self.image[i:i + 2, j:j + 2]))  
                # Calculate the magnitude of the gradient
                roberts_filtered_image[i, j] = np.sqrt(gx ** 2 + gy ** 2)

        return roberts_filtered_image
