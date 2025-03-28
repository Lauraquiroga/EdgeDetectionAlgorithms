import numpy as np

class Sobel:
    """
    Class implementing the Sobel edge detection algorithm.

    The Sobel operator is used in image processing to detect edges by computing 
    the gradient magnitude of an image using two 3x3 convolution kernels: one for 
    detecting changes in the horizontal direction (x) and another for the vertical 
    direction (y).

    Code adapted from: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py
    """

    def __init__(self, image):
        """
        Initialize the Sobel edge detector.

        Parameters:
        image (numpy array): A 2D array representing the input image in grayscale.

        Attributes:
        sobelx (numpy array): Kernel for detecting horizontal edges.
        sobely (numpy array): Kernel for detecting vertical edges.
        image (numpy array): The input grayscale image.
        """
        self.sobelx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        self.sobely = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.image = image

    def find_edges(self):
        """
        Apply the Sobel operator to detect edges in the image.

        The function computes the gradient magnitude at each pixel by convolving 
        the image with the Sobel kernels. The final edge strength at each pixel 
        is determined by the Euclidean norm of the gradients in the x and y directions.

        Returns:
        numpy array: A 2D array representing the edge-detected image.
        """
        [rows, columns] = np.shape(self.image)  # Get the dimensions of the input image
        sobel_filtered_image = np.zeros(shape=(rows, columns))  # Initialize the output image with zeros

        # Iterate through each pixel in the image (excluding border pixels)
        for i in range(rows - 2):
            for j in range(columns - 2):
                # Compute the gradient in the x-direction by applying the Sobel x kernel
                gx = np.sum(np.multiply(self.sobelx, self.image[i:i + 3, j:j + 3]))

                 # Compute the gradient in the y-direction by applying the Sobel y kernel
                gy = np.sum(np.multiply(self.sobely, self.image[i:i + 3, j:j + 3])) 

                # Compute the gradient magnitude
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2) 

        return sobel_filtered_image