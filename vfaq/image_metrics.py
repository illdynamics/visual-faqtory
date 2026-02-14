import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple
from scipy.ndimage import convolve
from scipy.ndimage import convolve # Import convolve

def calculate_blur(image_path: str) -> float:
    """
    Calculate the blurriness of an image using the Laplacian variance method.
    The result is normalized to a [0, 1] range.
    A higher value indicates more blur.
    """
    try:
        with Image.open(image_path) as img:
            gray = img.convert('L')  # Convert to grayscale
            # Apply a Gaussian blur as a pre-processing step to reduce noise
            gray_blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
            
            np_image = np.array(gray_blurred)
            
            # Manually apply Laplacian filter using convolution
            # Laplacian kernel
            laplacian_kernel = np.array([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]])
            
            laplacian_image = convolve(np_image, laplacian_kernel)
            
            # Variance of the Laplacian
            variance = laplacian_image.var()

            # Normalize to [0, 1] range.
            # These min/max values are empirical and might need tuning.
            # A common range for Laplacian variance for natural images is 0 to ~1000.
            # A completely flat image (all same color) would have 0 variance.
            # A very sharp image could have variance > 1000.
            # For normalization, we clamp and scale.
            min_variance = 0.0
            max_variance = 1000.0  # Empirical max for typical images, adjust if needed
            
            normalized_blur = 1.0 - np.clip((variance - min_variance) / (max_variance - min_variance), 0.0, 1.0)
            return normalized_blur
            
    except Exception as e:
        print(f"Error calculating blur for {image_path}: {e}")
        return 0.0 # Return 0 blur (sharp) on error

def calculate_entropy(image_path: str) -> float:
    """
    Calculate the Shannon entropy of an image's grayscale histogram.
    Higher entropy indicates more detail/randomness in the image.
    """
    try:
        with Image.open(image_path) as img:
            gray = img.convert('L')  # Convert to grayscale
            hist = gray.histogram()
            hist = [h for h in hist if h > 0] # Filter out zero-frequency bins
            
            # Normalize histogram to get probabilities
            total_pixels = sum(hist)
            probabilities = [h / total_pixels for h in hist]
            
            # Calculate Shannon entropy
            entropy = -sum(p * np.log2(p) for p in probabilities)
            
            # Normalize entropy to a reasonable range if needed.
            # Max entropy for 256 grayscale levels is log2(256) = 8.
            # Normalizing by 8 gives a [0, 1] range.
            normalized_entropy = entropy / 8.0
            return normalized_entropy
            
    except Exception as e:
        print(f"Error calculating entropy for {image_path}: {e}")
        return 0.0 # Return 0 entropy on error