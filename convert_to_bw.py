import os
from src.utils import Helper
import cv2

def convert_images_to_bw():
    """
    Convert all images in input_images directory to black and white
    and save them to bw_input_images directory.
    """
    # Create output directory if it doesn't exist
    output_dir = "bw_input_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Get all images from input directory
    input_dir = "input_images"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory: {input_dir}")
        print(f"Please place your images in the {input_dir} directory")
        return

    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # Process each image
    for filename in image_files:
        try:
            # Read image
            input_path = os.path.join(input_dir, filename)
            img = Helper.read_image(input_path)
            
            # Convert to grayscale if image is colored
            if len(img.shape) == 3:
                img = Helper.convert_greyscale(img)
            
            # Save black and white image
            output_path = os.path.join(output_dir, f"bw_{filename}")
            cv2.imwrite(output_path, img)
            print(f"Converted: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print(f"\nProcessing complete. Black and white images saved to {output_dir}")

if __name__ == "__main__":
    convert_images_to_bw()
