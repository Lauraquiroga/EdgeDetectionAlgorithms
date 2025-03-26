import cv2
import numpy as np
from src.utils import Helper
from src.sobel import Sobel
from src.canny import Canny
from src.roberts import Roberts

def main():
    """
    Main function to demonstrate edge detection algorithms.
    Loads an image, applies different edge detection methods,
    and displays the results side by side.
    """
    
    # TODO: Replace this dummy image with a real one with a pipeline
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)  # White rectangle
    cv2.circle(img, (100, 100), 30, 0, -1)  # Black circle

    print("Applying edge detection algorithms...")

    # Sobel
    sobel = Sobel(img)
    sobel_edges = sobel.find_edges()
    print("Sobel edge detection completed")

    # Roberts
    roberts = Roberts(img)
    roberts_edges = roberts.find_edges()
    print("Roberts Cross edge detection completed")

    # Canny
    canny = Canny(img)
    canny_edges = canny.canny()
    print("Canny edge detection completed")

    # Display results
    print("\nDisplaying results...")
    print("Close each figure window to see the next result")
    
    # Show original vs each edge detection result
    Helper.display_original_filtered(img, sobel_edges)
    Helper.display_original_filtered(img, roberts_edges)
    Helper.display_original_filtered(img, canny_edges)

if __name__ == '__main__':
    main()
