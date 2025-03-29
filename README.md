# Edge Detection Algorithms

Advanced Algorithms (COSC 520) - Project  
Authors: Ladan Tazik, Laura Quiroga, Emon Sarker

Edge detection is a fundamental technique in image processing and computer vision. This usually corresponds to object boundaries. This project implements and compares numerically three widely used edge detection algorithms: Sobel, Roberts Cross, and Canny.

## Project Structure

```
EdgeDetectionAlgorithms/
├── src/                           # Core algorithm implementations
│   ├── sobel.py                  # Sobel edge detection algorithm
│   ├── roberts.py                # Roberts Cross edge detection algorithm
│   ├── canny.py                  # Canny edge detection algorithm
│   └── utils.py                  # Helper utilities for image processing
├── tests/                        # Unit tests for algorithms
│   ├── test_sobel.py
│   ├── test_roberts.py
│   └── test_canny.py
├── input_images/                 # Directory for input images
├── bw_output_images/            # Output directory for processed images
│   ├── sobel/
│   ├── roberts/
│   └── canny/
├── mat_bin_ground_truth/        # Ground truth data for evaluation
├── synthetic_images/            # Synthetic test images
├── templates/                   # Web interface templates
│   └── index.html
└── uploads/                    # Temporary storage for web uploads
```

### Core Scripts

- **main.py**: Demonstrates edge detection algorithms on a sample image, displaying results side by side.

- **process_images.py**: Batch processes images using all three edge detection algorithms:

  - Takes images from `input_images/`
  - Applies Sobel, Roberts, and Canny edge detection
  - Saves results to algorithm-specific directories in `bw_output_images/`

- **evaluate_accuracy.py**: Evaluates algorithm accuracy against ground truth data:

  - Compares edge detection results with ground truth from BSD500 dataset
  - Calculates precision, recall, F1 score, and Jaccard index
  - Outputs detailed results to `accuracy_results.csv`

- **app.py**: Flask web application for interactive edge detection:
  - Allows users to upload images
  - Processes images with all three algorithms
  - Displays original and edge-detected results

### Edge Detection Implementations

- **src/sobel.py**: Implements the Sobel operator using 3x3 convolution kernels for detecting horizontal and vertical edges.

- **src/roberts.py**: Implements the Roberts Cross operator using 2x2 convolution kernels for detecting diagonal edges.

- **src/canny.py**: Implements the Canny edge detection algorithm with five steps:
  1. Gaussian blur for noise reduction
  2. Intensity gradient computation
  3. Non-maximum suppression
  4. Double thresholding
  5. Edge tracking by hysteresis

### Utility Scripts

- **convert_to_bw.py**: Converts input images to black and white format.

- **generate_synthetic_images.py**: Creates synthetic test images with known edge patterns.

- **process_synthetic_images.py**: Processes synthetic images with edge detection algorithms.

- **plot_performance.py**: Generates performance comparison visualizations.

## Setup instructions

Follow these steps to run the project:

1. **Clone the project**

   ```bash
   git clone https://github.com/Lauraquiroga/EdgeDetectionAlgorithms.git
   cd EdgeDetectionAlgorithms
   ```

2. **Install dependencies**  
   Ensure that you have Python installed on your system.  
   Create and activate a virtual environment (recommended version for the virtual environment: Python 3.12.7).

Install requirements mentioned in the requirements.txt file.

```bash
pip install -r requirements.txt
```

3.  **Run the project**  
    You can use the project in several ways:

- Process a single image:

```bash
python main.py
```

- Batch process multiple images:

```bash
python process_images.py
```

- Run the web interface:

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

- Evaluate algorithm accuracy:

```bash
python evaluate_accuracy.py
```

4.  **Run the unit tests**  
    To run the test cases use the following command.

```bash
python -m unittest discover -s tests
```

## References

The Sobel edge detection algorithm was first described by Irwin Sobel and Gary Feldman in 1968 as part of the Stanford Artificial Intelligence Laboratory (SAIL)
project:  
I. Sobel and G. Feldman, "A 3×3 isotropic gradient operator for image
processing," Pattern Classification and Scene Analysis, pp. 271–272, 01 1973.

The Roberts Cross operator was first described in:  
Lawrence G. Roberts, "Machine Perception of Three-Dimensional Solids," Ph.D. Dissertation, Massachusetts Institute of Technology, 1963.

The Canny algorithm was first described in:  
J. F. Canny, "A Computational Approach to Edge Detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679-698, Nov. 1986.

Code in Sobel and Roberts classes was adapted from:  
adamiao. sobel-filter-tutorial. Accessed: 2025-02-25. 2019. url: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py  
Which is licensed under the GNU General Public License v3.0 (GPL-3.0)
