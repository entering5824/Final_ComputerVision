"""
PCA Feature Extraction Module for Chromosome Classification

Combines PCA dimensionality reduction with blob features for chromosome classification.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .preprocessing import preprocess_image
from .blob_features import extract_all_features
from .. import config


def vectorize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to flattened vector.
    
    Args:
        image: Preprocessed image (already resized)
    
    Returns:
        Flattened image vector
    """
    return image.flatten().astype(np.float32)


class FeatureExtractor:
    """
    Feature extractor that combines PCA and blob features.
    
    Pipeline:
    1. Preprocess images (resize, normalize)
    2. Extract blob features (morphological, texture, histogram)
    3. Vectorize images and apply PCA
    4. Combine PCA features + blob features
    """
    
    def __init__(self, pca_variance_threshold: float = None, pca_n_components: int = None):
        """
        Initialize feature extractor.
        
        Args:
            pca_variance_threshold: Variance threshold for PCA (default: config.PCA_VARIANCE_THRESHOLD)
            pca_n_components: Fixed number of PCA components (alternative to variance threshold)
        """
        self.pca_variance_threshold = pca_variance_threshold or config.PCA_VARIANCE_THRESHOLD
        self.pca_n_components = pca_n_components or config.PCA_N_COMPONENTS
        
        self.pca = None
        self.image_scaler = None  # Scaler for image vectors before PCA
        self.blob_scaler = None  # Scaler for blob features
        
        self.pca_fitted = False
        self.scalers_fitted = False
    
    def _preprocess_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess all images."""
        preprocessed = []
        for img in images:
            preprocessed.append(preprocess_image(img, target_size=config.IMAGE_SIZE))
        return np.array(preprocessed)
    
    def _extract_blob_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract blob features for all images."""
        features_list = []
        for img in images:
            # Preprocess first
            preprocessed = preprocess_image(img, target_size=config.IMAGE_SIZE)
            # Extract features
            features = extract_all_features(
                preprocessed,
                include_texture=config.INCLUDE_TEXTURE_FEATURES,
                include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
            )
            features_list.append(features)
        return np.array(features_list)
    
    def _vectorize_images(self, images: np.ndarray) -> np.ndarray:
        """Vectorize preprocessed images."""
        n_samples = images.shape[0]
        image_vectors = images.reshape(n_samples, -1)
        return image_vectors
    
    def fit_pca(self, images: List[np.ndarray], fit_scaler: bool = True):
        """
        Fit PCA on training images.
        
        IMPORTANT: This should only be called on training data to avoid data leakage.
        
        Args:
            images: List of training images
            fit_scaler: Whether to fit scalers
        """
        print(f"Fitting PCA on {len(images)} training images...")
        
        # Preprocess images
        preprocessed = self._preprocess_images(images)
        
        # Vectorize images
        image_vectors = self._vectorize_images(preprocessed)
        
        # Fit scaler for image vectors
        if fit_scaler:
            self.image_scaler = StandardScaler()
            image_vectors_scaled = self.image_scaler.fit_transform(image_vectors)
        else:
            image_vectors_scaled = image_vectors
        
        # Fit PCA
        if self.pca_variance_threshold is not None:
            # Use variance threshold
            self.pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        else:
            # Use fixed number of components
            n_components = min(self.pca_n_components, image_vectors_scaled.shape[0], image_vectors_scaled.shape[1])
            self.pca = PCA(n_components=n_components, svd_solver='full')
        
        self.pca.fit(image_vectors_scaled)
        
        # Print PCA info
        n_components = self.pca.n_components_
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA fitted: {n_components} components, {explained_variance*100:.2f}% variance explained")
        
        self.pca_fitted = True
        
        # Fit blob feature scaler
        if fit_scaler:
            blob_features = self._extract_blob_features_batch(images)
            self.blob_scaler = StandardScaler()
            self.blob_scaler.fit(blob_features)
            self.scalers_fitted = True
    
    def get_combined_features(
        self,
        images: List[np.ndarray],
        fit_pca: bool = False,
        fit_scaler: bool = False,
        extended: bool = None,
        include_texture: bool = None,
        include_histogram: bool = None
    ) -> np.ndarray:
        """
        Extract combined features (PCA + blob) for images.
        
        Args:
            images: List of images
            fit_pca: Whether to fit PCA (only True for training set)
            fit_scaler: Whether to fit scalers (only True for training set)
            extended: Whether to use extended features (default: config.USE_EXTENDED_FEATURES)
            include_texture: Whether to include texture features (default: config.INCLUDE_TEXTURE_FEATURES)
            include_histogram: Whether to include histogram features (default: config.INCLUDE_HISTOGRAM_FEATURES)
        
        Returns:
            Combined feature matrix [n_samples, n_features]
        """
        if extended is None:
            extended = config.USE_EXTENDED_FEATURES
        if include_texture is None:
            include_texture = config.INCLUDE_TEXTURE_FEATURES
        if include_histogram is None:
            include_histogram = config.INCLUDE_HISTOGRAM_FEATURES
        
        # Fit PCA if requested (training phase)
        if fit_pca:
            self.fit_pca(images, fit_scaler=fit_scaler)
        
        if not self.pca_fitted:
            raise ValueError("PCA not fitted. Call fit_pca() first or set fit_pca=True")
        
        # Preprocess images
        preprocessed = self._preprocess_images(images)
        
        # Extract blob features
        blob_features = self._extract_blob_features_batch(images)
        
        # Vectorize images
        image_vectors = self._vectorize_images(preprocessed)
        
        # Scale image vectors
        if self.image_scaler is not None:
            image_vectors_scaled = self.image_scaler.transform(image_vectors)
        else:
            image_vectors_scaled = image_vectors
        
        # Apply PCA
        pca_features = self.pca.transform(image_vectors_scaled)
        
        # Scale blob features
        if self.blob_scaler is not None:
            blob_features_scaled = self.blob_scaler.transform(blob_features)
        else:
            blob_features_scaled = blob_features
        
        # Combine features
        combined_features = np.hstack([pca_features, blob_features_scaled])
        
        return combined_features
    
    def save(self, filepath: str):
        """Save PCA model and scalers to file."""
        save_dict = {
            'pca': self.pca,
            'image_scaler': self.image_scaler,
            'blob_scaler': self.blob_scaler,
            'pca_variance_threshold': self.pca_variance_threshold,
            'pca_n_components': self.pca_n_components,
            'pca_fitted': self.pca_fitted,
            'scalers_fitted': self.scalers_fitted
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved feature extractor to {filepath}")
    
    def load(self, filepath: str):
        """Load PCA model and scalers from file."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.pca = save_dict['pca']
        self.image_scaler = save_dict['image_scaler']
        self.blob_scaler = save_dict['blob_scaler']
        self.pca_variance_threshold = save_dict.get('pca_variance_threshold', config.PCA_VARIANCE_THRESHOLD)
        self.pca_n_components = save_dict.get('pca_n_components', config.PCA_N_COMPONENTS)
        self.pca_fitted = save_dict.get('pca_fitted', True)
        self.scalers_fitted = save_dict.get('scalers_fitted', True)
        
        print(f"Loaded feature extractor from {filepath}")

