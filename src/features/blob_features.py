"""
Blob Feature Extraction Module for Chromosome Classification

Extracts morphological, texture, and histogram features from chromosome images.
"""

import numpy as np
import cv2
from typing import List, Optional
from skimage import measure, feature
from skimage.morphology import binary_closing, binary_opening
from scipy import stats

from .. import config


def extract_blob_features(image: np.ndarray) -> np.ndarray:
    """
    Extract morphological blob features from chromosome image.
    
    Features extracted:
    - Area, Perimeter, Aspect Ratio
    - Major/Minor axis length, Eccentricity
    - Solidity, Extent, Orientation
    - Hu moments (7)
    
    Args:
        image: Preprocessed grayscale image (float in [0, 1] or uint8)
    
    Returns:
        Feature vector as numpy array
    """
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image.copy()
    
    # Threshold to create binary mask
    # Use Otsu's method for automatic thresholding
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # If no contours found, return zero features
        return np.zeros(17, dtype=np.float32)
    
    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate basic properties
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Fit ellipse
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        aspect_ratio = major_axis / (minor_axis + 1e-8)
        eccentricity = np.sqrt(1 - (minor_axis / (major_axis + 1e-8)) ** 2)
    else:
        major_axis = 0.0
        minor_axis = 0.0
        aspect_ratio = 0.0
        eccentricity = 0.0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox_area = w * h
    extent = area / (bbox_area + 1e-8)
    
    # Convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-8)
    
    # Orientation (angle of major axis)
    if len(largest_contour) >= 5:
        orientation = angle
    else:
        orientation = 0.0
    
    # Hu moments (7 invariant moments)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform for Hu moments (they can be very small)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # Combine all features
    features = np.array([
        area,
        perimeter,
        aspect_ratio,
        major_axis,
        minor_axis,
        eccentricity,
        solidity,
        extent,
        orientation,
        *hu_moments  # 7 Hu moments
    ], dtype=np.float32)
    
    return features


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Local Binary Pattern (LBP).
    
    Args:
        image: Preprocessed grayscale image
    
    Returns:
        Feature vector with LBP histogram
    """
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image.copy()
    
    # Compute LBP
    # Parameters: radius=3, n_points=24 (uniform LBP)
    lbp = feature.local_binary_pattern(img_uint8, P=24, R=3, method='uniform')
    
    # Compute histogram (bins = number of uniform patterns + 1 for non-uniform)
    n_bins = 26  # 25 uniform patterns + 1 for non-uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist.astype(np.float32)


def extract_histogram_features(image: np.ndarray) -> np.ndarray:
    """
    Extract histogram-based statistical features.
    
    Features:
    - Mean, Std, Skewness, Kurtosis
    - Percentiles (10th, 25th, 50th, 75th, 90th)
    
    Args:
        image: Preprocessed grayscale image
    
    Returns:
        Feature vector with histogram statistics
    """
    # Flatten image
    pixels = image.flatten()
    
    # Basic statistics
    mean = np.mean(pixels)
    std = np.std(pixels)
    skewness = stats.skew(pixels)
    kurtosis = stats.kurtosis(pixels)
    
    # Percentiles
    percentiles = np.percentile(pixels, [10, 25, 50, 75, 90])
    
    features = np.array([
        mean,
        std,
        skewness,
        kurtosis,
        *percentiles
    ], dtype=np.float32)
    
    return features


def extract_all_features(
    image: np.ndarray,
    include_texture: bool = False,
    include_histogram: bool = True
) -> np.ndarray:
    """
    Extract all available features from chromosome image.
    
    Args:
        image: Preprocessed grayscale image
        include_texture: Whether to include texture features (LBP)
        include_histogram: Whether to include histogram features
    
    Returns:
        Combined feature vector
    """
    features_list = []
    
    # Blob features (always included)
    blob_feat = extract_blob_features(image)
    features_list.append(blob_feat)
    
    # Texture features (optional)
    if include_texture:
        texture_feat = extract_texture_features(image)
        features_list.append(texture_feat)
    
    # Histogram features (optional)
    if include_histogram:
        hist_feat = extract_histogram_features(image)
        features_list.append(hist_feat)
    
    # Concatenate all features
    combined = np.concatenate(features_list)
    
    return combined

