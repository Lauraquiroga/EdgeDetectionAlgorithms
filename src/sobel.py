from matplotlib.image import imread
import matplotlib.pyplot as plt
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
        Sobelx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Sobely = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])