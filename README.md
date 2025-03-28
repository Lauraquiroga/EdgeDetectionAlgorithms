# Edge Detection Algorithms
Advanced Algorithms (COSC 520) - Project     
Authors: Ladan Tazik, Laura Quiroga, Emon Sarker     
      
Edge detection is a fundamental technique in image processing and computer vision. This usually corresponds to object boundaries. This project implements and compares numerically three widely used edge detection algorithms: Sobel, Roberts Cross, and Canny. 


## Project Structure

## Setup instructions   
Follow these steps to run the project:
1. **Clone the project**      
   ```bash
   git clone https://github.com/Lauraquiroga/EdgeDetectionAlgorithms.git
   cd EdgeDetectionAlgorithms
   ```

2.  **Install dependencies**     
   Ensure that you have Python installed on your system.     
   Create and activate a virtual environment (recommended version for the virtual environment: Python 3.12.7).       

   Install requirements mentioned in the requirements.txt file.       
   
   ```bash
   pip install -r requirements.txt
   ```


3.  **Run the project**      
   Run the project from the root folder.       
   ```bash
   python main.py # or use python3 main.py if needed
   ```
   This will    
   TO-DO    
   
4.  **Run the unit tests**       
   To run the test cases use the following command.       
   ```bash
   python -m unittest discover -s tests # or use python3 -m unittest discover -s tests if needed
   ```

## References         

The Sobel edge detection algorithm was first described by Irwin Sobel and Gary Feldman in 1968 as part of the Stanford Artificial Intelligence Laboratory (SAIL)
project:          
I. Sobel and G. Feldman, “A 3×3 isotropic gradient operator for image
processing,” Pattern Classification and Scene Analysis, pp. 271–272, 01
1973.           

The Roberts Cross operator was first described in:        
Lawrence G. Roberts, "Machine Perception of Three-Dimensional Solids," Ph.D. Dissertation, Massachusetts Institute of Technology, 1963.      

The Canny algorithm was first described in:        
J. F. Canny, "A Computational Approach to Edge Detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679-698, Nov. 1986.         

Code in Sobel and Roberts classes was adapted from:            
adamiao. sobel-filter-tutorial. Accessed: 2025-02-25. 2019. url: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py      
