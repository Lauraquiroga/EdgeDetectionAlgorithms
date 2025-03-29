import os
import numpy as np
import cv2
from scipy.io import loadmat
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import pandas as pd

class AccuracyEvaluator:
    """
    Evaluate edge detection accuracy by comparing with ground truth data.
    Calculates precision, recall, F1 score, and Jaccard score.
    """
    def __init__(self, ground_truth_dir="mat_bin_ground_truth", predictions_dir="bw_output_images"):
        self.ground_truth_dir = ground_truth_dir
        self.predictions_dir = predictions_dir
        self.algorithms = ['roberts', 'sobel', 'canny']
        
    def _extract_image_number(self, filename):
        """Extract the image number from filename."""
        # Extract numbers from filename
        numbers = ''.join(c for c in filename if c.isdigit())
        return numbers if numbers else None
        
    def _load_ground_truth(self, mat_file):
        """Load ground truth from .mat file and convert to binary format."""
        try:
            mat_data = loadmat(mat_file)
            # BSD500 format typically has multiple ground truths, we'll use the first one
            if 'groundTruth' in mat_data:
                ground_truth = mat_data['groundTruth'][0, 0]['Boundaries'][0, 0]
            else:
                # Try alternative field names
                field_names = list(mat_data.keys())
                print(f"Available fields in mat file: {field_names}")
                # You might need to adjust this based on actual mat file structure
                ground_truth = mat_data[field_names[-1]]
            
            return (ground_truth > 0).astype(np.uint8)
        except Exception as e:
            print(f"Error loading ground truth from {mat_file}: {str(e)}")
            return None

    def _load_prediction(self, image_path):
        """Load prediction image and convert to binary format."""
        if not os.path.exists(image_path):
            return None
        pred = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return (pred > 0).astype(np.uint8)

    def _calculate_metrics(self, ground_truth, prediction):
        """Calculate all metrics for a single image."""
        # Flatten arrays for metric calculation
        gt_flat = ground_truth.flatten()
        pred_flat = prediction.flatten()
        
        return {
            'precision': precision_score(gt_flat, pred_flat, zero_division=0),
            'recall': recall_score(gt_flat, pred_flat, zero_division=0),
            'f1': f1_score(gt_flat, pred_flat, zero_division=0),
            'jaccard': jaccard_score(gt_flat, pred_flat, zero_division=0)
        }

    def evaluate(self):
        """
        Evaluate all algorithms against ground truth data.
        Returns a DataFrame with results.
        """
        results = []
        
        # Get all ground truth files
        mat_files = [f for f in os.listdir(self.ground_truth_dir) if f.endswith('.mat')]
        
        if not mat_files:
            print(f"No .mat files found in {self.ground_truth_dir}")
            return
        
        total_files = len(mat_files)
        print(f"Found {total_files} ground truth files")
        
        # Process each ground truth file
        for idx, mat_file in enumerate(mat_files, 1):
            print(f"\nProcessing file {idx}/{total_files}: {mat_file}")
            
            # Get base name without extension
            base_name = os.path.splitext(mat_file)[0]
            
            try:
                # Load ground truth
                gt_path = os.path.join(self.ground_truth_dir, mat_file)
                ground_truth = self._load_ground_truth(gt_path)
                
                # Process each algorithm
                for algo in self.algorithms:
                    # Extract number from ground truth filename
                    img_number = self._extract_image_number(mat_file)
                    if not img_number:
                        print(f"Could not extract image number from {mat_file}")
                        continue

                    # Look for prediction file with this number
                    pred_path = os.path.join(self.predictions_dir, algo, f"{algo}_bw_{img_number}.jpg")
                    prediction = self._load_prediction(pred_path)
                    
                    if prediction is None:
                        print(f"Warning: No prediction found at {pred_path}")
                        continue
                        
                    # Ensure shapes match
                    if ground_truth.shape != prediction.shape:
                        print(f"Warning: Shape mismatch for {mat_file} with {algo}")
                        continue
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(ground_truth, prediction)
                    metrics.update({
                        'algorithm': algo,
                        'image': base_name
                    })
                    results.append(metrics)
                    
            except Exception as e:
                print(f"Error processing {mat_file}: {str(e)}")
        
        # Convert results to DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Calculate and display average metrics per algorithm
            print("\nAverage Metrics by Algorithm:")
            avg_metrics = df.groupby('algorithm').mean()
            print(avg_metrics.round(4))
            
            # Save detailed results to CSV
            csv_path = "accuracy_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nDetailed results saved to {csv_path}")
            
            return df
        else:
            print("No results to report")
            return None

def main():
    """Run the evaluation."""
    evaluator = AccuracyEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
