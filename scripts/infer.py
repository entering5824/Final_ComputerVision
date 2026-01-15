"""
Inference Script for Chromosome Classification

Predicts chromosome classes for unlabeled images.
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.classifier import ChromosomeClassifier
from src import config


def main():
    """Main inference pipeline."""
    
    print("=" * 80)
    print("CHROMOSOME CLASSIFICATION - INFERENCE")
    print("=" * 80)
    
    # Load classifier
    print("\nLoading trained model and feature extractor...")
    
    # First, get input_dim from a test image (or from metadata if available)
    import cv2
    import numpy as np
    from src.features.pca import FeatureExtractor
    from src.data.loader import load_unlabeled_data
    
    # Load extractor to get feature dimension
    extractor = FeatureExtractor()
    extractor.load(config.PCA_MODEL_PATH)
    
    # Try to infer input_dim from a sample image or use a default
    # For now, we'll load model metadata
    checkpoint = torch.load(config.SEMI_SUPERVISED_MODEL_PATH, map_location='cpu')
    if 'metadata' in checkpoint and 'input_dim' in checkpoint['metadata']:
        input_dim = checkpoint['metadata']['input_dim']
    else:
        # Fallback: extract features from a test image if available
        test_images = load_unlabeled_data("data/unlabeled")
        if len(test_images) > 0:
            test_features = extractor.get_combined_features(
                test_images[:1],
                fit_pca=False,
                fit_scaler=False,
                extended=config.USE_EXTENDED_FEATURES,
                include_texture=config.INCLUDE_TEXTURE_FEATURES,
                include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
            )
            input_dim = test_features.shape[1]
        else:
            raise ValueError("Cannot determine input_dim. Please ensure model metadata contains 'input_dim' or add a test image.")
    
    classifier = ChromosomeClassifier(
        model_path=config.SEMI_SUPERVISED_MODEL_PATH,
        pca_model_path=config.PCA_MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        hidden_dims=config.HIDDEN_DIMS,
        input_dim=input_dim
    )
    
    # Predict on unlabeled data
    print(f"\nPredicting on images in: data/unlabeled/")
    
    results_df = classifier.predict_batch_from_directory(
        image_dir="data/unlabeled",
        output_path=os.path.join(config.RESULTS_DIR, "predictions.csv")
    )
    
    if len(results_df) > 0:
        print(f"\nPredictions:")
        print(results_df.head(10))
        print(f"\nTotal predictions: {len(results_df)}")
        print(f"\nClass distribution:")
        print(results_df['predicted_class'].value_counts())
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Predictions saved to: {config.RESULTS_DIR}/predictions.csv")


if __name__ == "__main__":
    main()

