import os
import cv2
from src.utils import Helper
from src.roberts import Roberts
from src.sobel import Sobel
from src.canny import Canny

class ImageProcessor:
    """
    Process images with multiple edge detection algorithms.
    Allows easy configuration of input and output directories.
    """
    def __init__(self, input_dir="bw_input_images", output_dir="bw_output_images"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.algorithms = {
            'roberts': self._apply_roberts,
            'sobel': self._apply_sobel,
            'canny': self._apply_canny
        }

    def _create_output_dirs(self):
        """Create main output directory and algorithm-specific subdirectories."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

        # Create subdirectory for each algorithm
        for algo_name in self.algorithms.keys():
            algo_dir = os.path.join(self.output_dir, algo_name)
            if not os.path.exists(algo_dir):
                os.makedirs(algo_dir)
                print(f"Created directory: {algo_dir}")

    def _apply_roberts(self, img):
        """Apply Roberts Cross edge detection."""
        detector = Roberts(img)
        return detector.find_edges()

    def _apply_sobel(self, img):
        """Apply Sobel edge detection."""
        detector = Sobel(img)
        return detector.find_edges()

    def _apply_canny(self, img):
        """Apply Canny edge detection."""
        detector = Canny(img)
        return detector.canny()

    def process_images(self):
        """
        Process all images in input directory with all algorithms.
        Save results to algorithm-specific subdirectories in output directory.
        """
        # Check input directory exists
        if not os.path.exists(self.input_dir):
            print(f"Input directory '{self.input_dir}' not found.")
            return

        # Create output directories
        self._create_output_dirs()

        # Get all image files
        image_files = [f for f in os.listdir(self.input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print(f"No images found in {self.input_dir}")
            return

        # Process each image with each algorithm
        total_images = len(image_files)
        for idx, filename in enumerate(image_files, 1):
            try:
                print(f"\nProcessing image {idx}/{total_images}: {filename}")
                
                # Read image
                input_path = os.path.join(self.input_dir, filename)
                img = Helper.read_image(input_path)

                # Apply each algorithm
                for algo_name, algo_func in self.algorithms.items():
                    # Process image
                    edges = algo_func(img)
                    
                    # Save result
                    output_path = os.path.join(self.output_dir, algo_name, f"{algo_name}_{filename}")
                    cv2.imwrite(output_path, edges)
                    print(f"- {algo_name.title()} edge detection completed")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        print(f"\nProcessing complete. Results saved to {self.output_dir}")

def main():
    """
    Main function to demonstrate usage.
    Can be modified to accept different input/output directories.
    """
    # Example usage with default directories
    processor = ImageProcessor()
    processor.process_images()

    # Example usage with custom directories
    # processor = ImageProcessor("custom_input_dir", "custom_output_dir")
    # processor.process_images()

if __name__ == "__main__":
    main()
