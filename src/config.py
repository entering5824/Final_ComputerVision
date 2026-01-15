"""
Configuration Module for Semi-Supervised Chromosome Classification

Centralized configuration for all paths, hyperparameters, and feature options.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
# Labeled data (D_L): folders 1-22, X, Y nằm trực tiếp trong data/
LABELED_DATA_DIR = DATA_DIR  # data/1/, data/2/, ..., data/22/, data/X/, data/Y/
UNLABELED_DATA_DIR = DATA_DIR / "unlabeled"  # data/unlabeled/ chứa ảnh chưa gán nhãn

# Model paths
PCA_MODEL_PATH = str(MODELS_DIR / "pca_model.pkl")
SUPERVISED_MODEL_PATH = str(MODELS_DIR / "supervised_model.pth")
SEMI_SUPERVISED_MODEL_PATH = str(MODELS_DIR / "semi_supervised_model.pth")

# Results paths
RESULTS_JSON_PATH = str(RESULTS_DIR / "comparison.json")
CONFUSION_MATRIX_PATH = str(RESULTS_DIR / "confusion_matrix.png")
PER_CLASS_PERFORMANCE_PATH = str(RESULTS_DIR / "per_class_performance.png")
TRAINING_CURVES_PATH = str(RESULTS_DIR / "training_curves.png")
PCA_VARIANCE_PATH = str(RESULTS_DIR / "pca_variance.png")

# Model architecture
NUM_CLASSES = 23  # 1-22, X, Y
HIDDEN_DIMS = [512, 256, 128]  # MLP hidden layer dimensions (as per plan: [512 → 256 → 128])

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
USE_LR_SCHEDULER = True
EARLY_STOPPING_PATIENCE = 10

# Semi-supervised learning parameters
MIN_TOTAL_EPOCHS = 300  # Minimum total epochs (required by assignment)
EPOCHS_PER_ITERATION = 25  # Epochs per self-training iteration (as per plan: 10-25)
MAX_ITERATIONS = 20  # Maximum self-training iterations
CONFIDENCE_THRESHOLDS = [0.85, 0.90, 0.95, 0.98]  # T_conf values to try (annealing: 0.98 → 0.85)
TOP_K_PER_CLASS = 100  # Maximum pseudo-labels per class per iteration

# Data splitting
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
UNLABELED_RATIO = 0.35  # Proportion of training data to convert to unlabeled
RANDOM_STATE = 42

# Feature extraction parameters
# Image preprocessing
IMAGE_SIZE = (128, 64)  # (height, width) - maintain aspect ratio for chromosomes
NORMALIZE_IMAGE = True

# PCA parameters
PCA_VARIANCE_THRESHOLD = 0.95  # Keep components that explain ≥95% variance
PCA_N_COMPONENTS = 128  # Alternative: fixed number of components

# Blob features
USE_EXTENDED_FEATURES = True  # Include extended morphological features
INCLUDE_TEXTURE_FEATURES = False  # LBP, GLCM (can be slow)
INCLUDE_HISTOGRAM_FEATURES = True  # Histogram statistics

# Augmentation parameters
USE_AUGMENTATION = True
ROTATION_RANGE = 15  # ±degrees
SHIFT_RANGE = 0.05  # ±5% of image size
SCALE_RANGE = 0.05  # ±5% scaling

# Device
USE_GPU = True

# Logging
VERBOSE = True

# Class names mapping
CLASS_NAMES = [str(i) for i in range(1, 23)] + ['X', 'Y']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

