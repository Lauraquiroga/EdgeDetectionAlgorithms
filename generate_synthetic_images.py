import os
import cv2
import numpy as np

def generate_synthetic_image(size):
    """
    Generate a synthetic image with a white rectangle and black circle.
    The shapes are scaled proportionally to the image size.
    
    Parameters:
    size (tuple): (width, height) of the image
    
    Returns:
    numpy.ndarray: Generated image
    """
    # Create black background
    img = np.zeros(size, dtype=np.uint8)
    
    # Calculate rectangle dimensions (50% of image size)
    rect_width = int(size[1] * 0.5)
    rect_x = (size[1] - rect_width) // 2
    rect_y = (size[0] - rect_width) // 2
    
    # Draw white rectangle
    cv2.rectangle(img, 
                 (rect_x, rect_y), 
                 (rect_x + rect_width, rect_y + rect_width), 
                 255, 
                 -1)
    
    # Calculate circle dimensions (30% of rectangle size)
    circle_radius = int(rect_width * 0.3)
    circle_center = (size[1] // 2, size[0] // 2)
    
    # Draw black circle
    cv2.circle(img, circle_center, circle_radius, 0, -1)
    
    return img

def main():
    """
    Generate 100 synthetic images with sizes from 20x20 to 2000x2000.
    Save them in a 'synthetic_images' directory.
    """
    # Create output directory if it doesn't exist
    output_dir = "synthetic_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Calculate size increments for 100 images
    sizes = np.linspace(20, 2000, 100, dtype=int)
    
    # Generate and save images
    for i, size in enumerate(sizes):
        # Generate square image
        img = generate_synthetic_image((size, size))
        
        # Save image with descriptive filename
        filename = f"synthetic_{size}x{size}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        
        # Progress update
        print(f"Generated image {i+1}/100: {filename}")

if __name__ == "__main__":
    main()
