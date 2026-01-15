"""
Image Preprocessing Module for Chromosome Classification

Functions for resizing, normalizing, and preprocessing chromosome images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from .. import config


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single chromosome image.
    
    Steps:
    1. Resize to target size (maintains aspect ratio if possible)
    2. Normalize to [0, 1] range
    
    Args:
        image: Input grayscale image (numpy array)
        target_size: Target size (height, width). Default: config.IMAGE_SIZE
        normalize: Whether to normalize to [0, 1]. Default: True
    
    Returns:
        Preprocessed image as float32 array in [0, 1] range
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE
    
    # Resize image
    if image.shape != target_size:
        # Use INTER_AREA for downsampling (better quality)
        # Use INTER_CUBIC for upsampling
        if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        
        resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    else:
        resized = image.copy()
    
    # Convert to float32
    resized = resized.astype(np.float32)
    
    # Normalize to [0, 1]
    if normalize:
        if resized.max() > 1.0:
            resized = resized / 255.0
    
    return resized


def normalize_image(image: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize image using zero-mean, unit-variance (z-score normalization).
    
    Args:
        image: Input image (float array)
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        Normalized image
    """
    if method == 'zscore':
        # Zero-mean, unit-variance
        mean = np.mean(image)
        std = np.std(image)
        
        if std > 1e-8:  # Avoid division by zero
            normalized = (image - mean) / std
        else:
            normalized = image - mean
        
        return normalized
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        image: Input grayscale image (uint8 or float in [0, 1])
    
    Returns:
        Histogram equalized image
    """
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img_uint8)
    
    # Convert back to float32 in [0, 1]
    return equalized.astype(np.float32) / 255.0

