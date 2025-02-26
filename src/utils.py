from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import time

class Helper:
    """
    A utility class containing helper methods for handling images and timing functions.
    
    The `Helper` class includes static methods to:
    - Measure the execution time of functions.
    - Convert images to greyscale
    - Display images
    """

    @staticmethod
    def timing_decorator(func):
        """
        A decorator that measures the execution time of a function.
        
        This decorator wraps around a function, calculates the time taken to execute the function,
        and returns both the result of the function and the elapsed time.

        Parameters:
        func (function): The function to be wrapped and timed.

        Returns:
        tuple: A tuple containing the result of the function and the execution time in seconds.
        """
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            return result, elapsed_time  # Return both the function result and execution time
        return wrapper
    
    @staticmethod
    def display_original_filtered(input_image, filtered_image):
        """
        Displays the original image and the filtered image

        Parameters:
        input_image: the array representation of the input image (original)
        filtered_image: the array representation of the filtered image (edges)
        """
        fig2 = plt.figure(2)
        ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
        ax1.imshow(input_image)
        ax2.imshow(filtered_image, cmap=plt.get_cmap('gray'))
        fig2.show()

        # Show both images
        plt.show()

    @staticmethod
    def convert_greyscale(input_image):
        """
        Convert an image into greyscale

        Parameters:
        input_image: array representation of input (coloured) image

        Return:
        array representation of the greyscale image
        """
        # Extracting each one of the RGB components
        r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]

        # The following operation will take weights and parameters to convert the color image to grayscale
        gamma = 1.400  # a parameter
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        return  r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma


    @staticmethod
    def read_image(file_path):
        """
        Returns an array representation of an image given its file path.

        Parameters:
        file_path (str): the path to the image file
        """
        return imread(file_path)