"""
Feature extraction module for chromosome classification.

Contains functions for preprocessing images, extracting blob features,
applying PCA, and data augmentation.
"""

from .preprocessing import preprocess_image, normalize_image
from .blob_features import extract_blob_features, extract_texture_features, extract_histogram_features
from .pca import FeatureExtractor, vectorize_image
from .augmentation import ChromosomeAugmentation, create_augmentation_pipeline

__all__ = [
    'preprocess_image',
    'normalize_image',
    'extract_blob_features',
    'extract_texture_features',
    'extract_histogram_features',
    'FeatureExtractor',
    'vectorize_image',
    'ChromosomeAugmentation',
    'create_augmentation_pipeline',
]

