"""
Training Module for Chromosome Classification

Provides supervised and semi-supervised training functionality.
"""

from .supervised import train_supervised, predict_with_confidence, ChromosomeDataset
from .semi_supervised import self_training_loop, select_pseudo_labels

__all__ = [
    'train_supervised',
    'predict_with_confidence',
    'ChromosomeDataset',
    'self_training_loop',
    'select_pseudo_labels',
]

