"""
Inference Module for Chromosome Classification

Provides inference pipeline for predicting on unlabeled chromosome images.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd

from ..models.mlp import ChromosomeMLP
from ..features.pca import FeatureExtractor
from ..training.supervised import predict_with_confidence
from ..utils.model_utils import get_device


class ChromosomeClassifier:
    """
    Complete inference pipeline for chromosome classification.
    
    Loads trained model and feature extractor, provides prediction interface.
    """
    
    def __init__(
        self,
        model_path: str,
        pca_model_path: str,
        num_classes: int = 23,
        hidden_dims: List[int] = None,
        input_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model checkpoint
            pca_model_path: Path to PCA model
            num_classes: Number of classes (23)
            hidden_dims: Hidden layer dimensions (must match training)
            input_dim: Input dimension (if not in checkpoint metadata)
            device: Device (CPU or GPU)
        """
        device = get_device(device)
        self.device = device
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [128, 64]
        
        # Load feature extractor (PCA)
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load(pca_model_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            if 'input_dim' in metadata:
                input_dim = metadata['input_dim']
            elif input_dim is None:
                raise ValueError("Model metadata must contain 'input_dim' or pass input_dim parameter")
        elif input_dim is None:
            raise ValueError("Model checkpoint must contain metadata with 'input_dim' or pass input_dim parameter")
        
        self.model = ChromosomeMLP(input_dim, num_classes, self.hidden_dims).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Class names
        self.class_names = [str(i) for i in range(1, 23)] + ['X', 'Y']
    
    def predict(
        self,
        images: List[np.ndarray],
        batch_size: int = 32,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Predict chromosome classes for images.
        
        Args:
            images: List of grayscale chromosome images
            batch_size: Batch size for prediction
            return_probabilities: Whether to return full probability distributions
        
        Returns:
            Dictionary with predictions, class names, and optionally probabilities
        """
        # Extract features
        from .. import config
        features = self.feature_extractor.get_combined_features(
            images,
            fit_pca=False,
            fit_scaler=False,
            extended=config.USE_EXTENDED_FEATURES,
            include_texture=config.INCLUDE_TEXTURE_FEATURES,
            include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
        )
        
        # Predict
        predictions, confidences = predict_with_confidence(
            self.model,
            features,
            device=self.device,
            batch_size=batch_size
        )
        
        # Get class names
        predicted_classes = [self.class_names[pred] for pred in predictions]
        
        result = {
            'predictions': predicted_classes,
            'prediction_indices': predictions.tolist(),
            'confidences': confidences.tolist()
        }
        
        if return_probabilities:
            # Get full probability distributions
            from torch.utils.data import DataLoader
            from ..training.supervised import ChromosomeDataset
            
            dataset = ChromosomeDataset(features, np.zeros(len(features)))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            all_probs = []
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
            
            probabilities = np.vstack(all_probs)
            result['probabilities'] = probabilities.tolist()
            result['class_names'] = self.class_names
        
        return result
    
    def predict_single(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict for a single image.
        
        Args:
            image: Single grayscale chromosome image
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        result = self.predict([image], return_probabilities=False)
        return result['predictions'][0], result['confidences'][0]
    
    def predict_batch_from_directory(
        self,
        image_dir: str,
        output_path: Optional[str] = None,
        image_extensions: List[str] = None
    ) -> pd.DataFrame:
        """
        Predict for all images in a directory and save results.
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save predictions CSV (optional)
            image_extensions: List of image extensions to look for
        
        Returns:
            DataFrame with predictions
        """
        import cv2
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        image_dir = Path(image_dir)
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
        
        images = []
        filenames = []
        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(img_path.name)
        
        if len(images) == 0:
            print(f"No images found in {image_dir}")
            return pd.DataFrame()
        
        print(f"Processing {len(images)} images...")
        results = self.predict(images, return_probabilities=False)
        
        # Create DataFrame
        df = pd.DataFrame({
            'filename': filenames,
            'predicted_class': results['predictions'],
            'confidence': results['confidences']
        })
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Saved predictions to {output_path}")
        
        return df

