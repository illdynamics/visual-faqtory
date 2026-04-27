import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple
from scipy.ndimage import convolve


def calculate_frame_similarity(
    image_a_path: str,
    image_b_path: str,
    *,
    size: Tuple[int, int] = (128, 72),
) -> float:
    """
    Compute a lightweight continuity score in the range [0, 1].

    The original metric leaned heavily on grayscale structure, which made it too
    forgiving for "same silhouette, totally soupified texture/color" failures.
    For reinjected animation endpoints we want something stricter, so the score
    now blends:
      - luminance correlation
      - edge correlation
      - RGB color similarity
      - a tiny perceptual dHash agreement check

    1.0 means "very similar", 0.0 means "wildly different".
    """
    try:
        with Image.open(image_a_path) as img_a, Image.open(image_b_path) as img_b:
            a_rgb = img_a.convert('RGB').resize(size, Image.Resampling.BICUBIC)
            b_rgb = img_b.convert('RGB').resize(size, Image.Resampling.BICUBIC)

            a = a_rgb.convert('L')
            b = b_rgb.convert('L')

            a_arr = np.asarray(a, dtype=np.float32) / 255.0
            b_arr = np.asarray(b, dtype=np.float32) / 255.0

            # Base luminance correlation.
            a_vec = a_arr.flatten() - float(a_arr.mean())
            b_vec = b_arr.flatten() - float(b_arr.mean())
            denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) + 1e-8
            luminance_corr = float(np.dot(a_vec, b_vec) / denom)

            # Edge correlation helps catch structure drift even when colors mutate.
            a_edges = np.asarray(a.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
            b_edges = np.asarray(b.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
            ae_vec = a_edges.flatten() - float(a_edges.mean())
            be_vec = b_edges.flatten() - float(b_edges.mean())
            edge_denom = (np.linalg.norm(ae_vec) * np.linalg.norm(be_vec)) + 1e-8
            edge_corr = float(np.dot(ae_vec, be_vec) / edge_denom)

            # Color drift matters for continuity too. Soup can preserve outline while
            # melting the palette, so include a lightweight RGB mean absolute error.
            a_rgb_arr = np.asarray(a_rgb.resize((64, 36), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            b_rgb_arr = np.asarray(b_rgb.resize((64, 36), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            color_mae = float(np.mean(np.abs(a_rgb_arr - b_rgb_arr)))
            color_score = max(0.0, min(1.0, 1.0 - color_mae))

            # Tiny perceptual hash agreement for extra protection against obvious
            # texture/layout soup while staying dependency-light.
            def _dhash_bits(img: Image.Image, hash_size: int = 8) -> np.ndarray:
                gray = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BICUBIC)
                arr = np.asarray(gray, dtype=np.float32)
                return (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).flatten()

            hash_a = _dhash_bits(a_rgb)
            hash_b = _dhash_bits(b_rgb)
            hash_score = float(np.mean(hash_a == hash_b))

            # Convert [-1, 1] correlations to [0, 1].
            luminance_score = max(0.0, min(1.0, (luminance_corr + 1.0) / 2.0))
            edge_score = max(0.0, min(1.0, (edge_corr + 1.0) / 2.0))

            # Blend. Keep structure primary, but make room for color/hash so the
            # continuity guard stops rubber-stamping chromatic soup.
            return float(
                (0.35 * luminance_score)
                + (0.25 * edge_score)
                + (0.25 * color_score)
                + (0.15 * hash_score)
            )
    except Exception as e:
        print(f"Error calculating frame similarity for {image_a_path} vs {image_b_path}: {e}")
        return 0.0

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