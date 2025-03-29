import numpy as np

class Canny:
    """
    The Canny Edge Detection algorithm. 
    This version of Canny works on greyscale images only.
    """

    def __init__(self, image, low_threshold=None, high_threshold=None):
        """
        Initialize Canny edge detection parameters.
        
        Parameters:
        - image: np.array, grayscale input image.
        - low_threshold: int, lower threshold for hysteresis (optional).
        - high_threshold: int, upper threshold for hysteresis (optional).
        """
        
        self.image = image
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.strong_pixel = None  # Strong edges (default: 255)
        self.weak_pixel = None    # Weak edges (default: 50)

        if len(self.image.shape) == 3:
            raise ValueError("Input image must be grayscale")
    
    def create_gaussian_kernel(self, size, sigma):
        """
        Create a Gaussian kernel for blurring.
        
        Parameters:
        - size: int, kernel size (must be odd)
        - sigma: float, standard deviation for Gaussian kernel
        
        Returns:
        - np.array: 2D Gaussian kernel
        """
        # Ensure size is odd
        if size % 2 == 0:
            size += 1
            
        # Create coordinate grid
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        
        # Calculate kernel values
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        
        # Normalize kernel
        return kernel / np.sum(kernel)
    
    def apply_gaussian_blur(self, k=(5, 5), sigma=1.4):
        """
        Step 1: Apply Gaussian blur to smooth the image and reduce noise.
        
        Parameters:
        - k: tuple, kernel size for Gaussian filter.
        - sigma: float, standard deviation for Gaussian kernel.
        
        Returns:
        - np.array, blurred image.
        """
        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(k[0], sigma)
        
        # Apply convolution
        rows, cols = self.image.shape
        pad = k[0] // 2
        padded = np.pad(self.image, ((pad, pad), (pad, pad)), mode='reflect')
        blurred = np.zeros_like(self.image, dtype=np.float64)
        
        for i in range(rows):
            for j in range(cols):
                # Extract window and apply kernel
                window = padded[i:i + k[0], j:j + k[0]]
                blurred[i, j] = np.sum(window * kernel)
                
        return blurred
    
    def compute_intensity_gradient(self, blurred_image):
        """
        Step 2: Compute the gradient magnitude and direction using our Sobel implementation.

        Parameters:
        - blurred_image: np.array, smoothed image after Gaussian blur.

        Returns:
        - magnitude: np.array, gradient magnitude.
        - direction: np.array, gradient direction in radians.
        """
        # Define Sobel kernels for x and y directions
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        rows, cols = blurred_image.shape
        grad_x = np.zeros((rows, cols))
        grad_y = np.zeros((rows, cols))
        
        # Apply Sobel kernels
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                grad_x[i, j] = np.sum(sobel_x * blurred_image[i-1:i+2, j-1:j+2])
                grad_y[i, j] = np.sum(sobel_y * blurred_image[i-1:i+2, j-1:j+2])
        
        # Calculate magnitude and direction
        magnitude = np.hypot(grad_x, grad_y)  # More stable than sqrt(x**2 + y**2)
        direction = np.arctan2(grad_y, grad_x)  # Radians
        
        return magnitude, direction
    
    def non_maximum_suppression(self, magnitude, direction):
        """
        Step 3: Perform non-maximum suppression to thin the edges.
        
        Parameters:
        - magnitude: np.array, gradient magnitude.
        - direction: np.array, gradient direction.
        
        Returns:
        - np.array, suppressed image with thinned edges.
        """
        rows, cols = magnitude.shape
        suppressed = np.zeros((rows, cols), dtype=np.float32)
        
        
        # Convert direction from radians to degrees (0 to 180 range)
        angle = direction * (180.0 / np.pi) % 180  
        angle = np.nan_to_num(angle)  # Replaces NaN with 0, inf with large finite number
        for i in range(1, rows - 1):  # Ignore image borders
            for j in range(1, cols - 1):  
                q = 255
                r = 255

                # Horizontal edge
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Diagonal edge (45 degrees)
                elif 22.5 <= angle[i, j] < 67.5: 
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Vertical edge (90 degrees)
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Diagonal edge (135 degrees)
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]
                
                # Suppress non-maximum values
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0
        
        return suppressed
    
    def adaptive_thresholding(self, magnitude):
        """
        Compute robust adaptive thresholds using percentile values.

        Parameters:
        - magnitude: np.array, gradient magnitude.

        Returns:
        - (low, high): tuple, lower and upper thresholds.
        """
        # Flatten and remove zeros (since most are zero)
        flat = magnitude[magnitude > 0].ravel()

        if len(flat) == 0:
            return 0, 0  # fallback

        low = np.percentile(flat, 20)
        high = np.percentile(flat, 50)

        return low, high

    def double_thresholding(self, suppressed):
        """
        Step 4: Apply double thresholding to classify strong and weak edges.
        
        Parameters:
        - suppressed: np.array, image after non-maximum suppression.
        
        Returns:
        - np.array, thresholded edges with strong and weak edges labeled.
        """
        if self.low_threshold is None or self.high_threshold is None:
            self.low_threshold, self.high_threshold = self.adaptive_thresholding(suppressed)
        
        strong = self.strong_pixel if self.strong_pixel is not None else 255
        weak = self.weak_pixel if self.weak_pixel is not None else 50
        edges = np.zeros_like(suppressed)
        
        # Strong edges
        strong_i, strong_j = np.where(suppressed >= self.high_threshold)
        edges[strong_i, strong_j] = strong
        
        # Weak edges
        weak_i, weak_j = np.where((suppressed >= self.low_threshold) & (suppressed < self.high_threshold))
        edges[weak_i, weak_j] = weak
        
        return edges
    
    def edge_tracking_by_hysteresis(self, edges):
        """
        Step 5: Perform edge tracking by hysteresis to preserve weak edges 
        that are connected to strong edges.
        
        Parameters:
        - edges: np.array, thresholded edge image.
        
        Returns:
        - np.array, final edge-detected image.
        """
        strong = self.strong_pixel if self.strong_pixel is not None else 255
        weak = self.weak_pixel if self.weak_pixel is not None else 50
        
        rows, cols = edges.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if edges[i, j] == weak:
                    # Check if any neighboring pixel is a strong edge
                    if (edges[i + 1, j - 1] == strong or edges[i + 1, j] == strong or edges[i + 1, j + 1] == strong
                            or edges[i, j - 1] == strong or edges[i, j + 1] == strong
                            or edges[i - 1, j - 1] == strong or edges[i - 1, j] == strong or edges[i - 1, j + 1] == strong):
                        edges[i, j] = strong  # Promote weak edge to strong
                    else:
                        edges[i, j] = 0  # Suppress weak edge
        
        return edges
    
    def canny(self, k=(5, 5), sigma=1.4):
        """
        Perform Canny edge detection step by step.
        
        Parameters:
        - k: tuple, kernel size for Gaussian blur.
        - sigma: float, standard deviation for Gaussian kernel.
        
        Returns:
        - np.array, final edge-detected image.
        """
        blurred = self.apply_gaussian_blur(k, sigma)
        magnitude, direction = self.compute_intensity_gradient(blurred)
        suppressed = self.non_maximum_suppression(magnitude, direction)
        edges = self.double_thresholding(suppressed)
        final_edges = self.edge_tracking_by_hysteresis(edges)
        
        return final_edges
