import numpy as np

class Roberts:
    """
    Class implementing the Roberts Cross algorithm for edge detection

    The Roberts Cross operator performs a simple, quick to compute, 2-D spatial gradient measurement on an image.
    It was one of the first edge detectors and was initially proposed by Lawrence Roberts in 1963.
    
    The operator consists of a pair of 2×2 convolution kernels:
    Gx = [[ 1, 0],      Gy = [[ 0, -1],
          [ 0,-1]]            [ 1,  0]]

    Code adapted from: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py
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
