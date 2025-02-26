import numpy as np

class Sobel:
    """
    Class implementing the Sobel algorithm for edge detection

    The Sobel algorithm was originally described in:


    Code adapted from: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py
    """

    def __init__(self, image):
        """
        Initialize Sobel algorithm parameters
        
        Parameters:
        image: array representation of the input image in greyscale
        
        """
        self.sobelx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        self.sobely = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.image = image

    def find_edges(self):
        [rows, columns] = np.shape(self.image)  # we need to know the shape of the input grayscale image
        sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

        # Now we "sweep" the image in both x and y directions and compute the output
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(self.sobelx, self.image[i:i + 3, j:j + 3]))  # x direction
                gy = np.sum(np.multiply(self.sobely, self.image[i:i + 3, j:j + 3]))  # y direction
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the magnitude of the gradient