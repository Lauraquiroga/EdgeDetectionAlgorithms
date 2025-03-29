import os
import csv
import cv2
import numpy as np
from src.roberts import Roberts
from src.sobel import Sobel
from src.canny import Canny
from src.utils import Helper

def process_image(image_path, output_dir):
    """
    Process a single image with all three edge detection algorithms.
    
    Parameters:
    image_path (str): Path to input image
    output_dir (str): Directory to save edge-detected images
    
    Returns:
    dict: Dictionary containing runtime results for each algorithm
    """
    # Read and convert image to grayscale
    img = Helper.read_image(image_path)
    if len(img.shape) == 3:  # If RGB image
        img = Helper.convert_greyscale(img)
    
    # Get image resolution from filename
    resolution = os.path.basename(image_path).split('_')[1].split('.')[0]  # e.g., "100x100"
    
    results = []
    
    # Roberts Edge Detection
    roberts = Roberts(img)
    roberts_edges, roberts_time = Helper.timing_decorator(roberts.find_edges)()
    cv2.imwrite(os.path.join(output_dir, f"roberts_{resolution}.png"), roberts_edges)
    results.append(("Roberts", resolution, roberts_time))
    
    # Sobel Edge Detection
    sobel = Sobel(img)
    sobel_edges, sobel_time = Helper.timing_decorator(sobel.find_edges)()
    cv2.imwrite(os.path.join(output_dir, f"sobel_{resolution}.png"), sobel_edges)
    results.append(("Sobel", resolution, sobel_time))
    
    # Canny Edge Detection
    canny = Canny(img)
    canny_edges, canny_time = Helper.timing_decorator(canny.canny)()
    cv2.imwrite(os.path.join(output_dir, f"canny_{resolution}.png"), canny_edges)
    results.append(("Canny", resolution, canny_time))
    
    return results

def main():
    """
    Process all synthetic images and save runtime results.
    """
    # Create output directory if it doesn't exist
    output_dir = "synthetic_image_edges"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get all synthetic images
    input_dir = "synthetic_images"
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
    # Prepare CSV file
    csv_file = "runtime.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "image_resolution", "time_taken"])
        
        # Process each image
        total_images = len(image_files)
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing image {i}/{total_images}: {image_file}")
            
            image_path = os.path.join(input_dir, image_file)
            results = process_image(image_path, output_dir)
            
            # Write results to CSV
            writer.writerows(results)
            
            # Flush CSV after each image to save progress
            f.flush()
    
    print(f"\nProcessing complete. Results saved to {csv_file}")
    print(f"Edge-detected images saved to {output_dir}")

if __name__ == "__main__":
    main()
