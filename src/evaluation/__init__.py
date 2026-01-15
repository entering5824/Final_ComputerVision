"""
Evaluation module for chromosome classification.

Contains functions for calculating metrics, calibration, and visualization.
"""

from .metrics import calculate_metrics, per_class_metrics, compare_models, print_comparison_table
from .calibration import TemperatureScaling, plot_reliability_diagram, expected_calibration_error
from .visualization import (
    plot_confusion_matrix, plot_training_curves, 
    plot_pca_variance, plot_per_class_performance
)

__all__ = [
    'calculate_metrics',
    'per_class_metrics',
    'compare_models',
    'print_comparison_table',
    'TemperatureScaling',
    'plot_reliability_diagram',
    'expected_calibration_error',
    'plot_confusion_matrix',
    'plot_training_curves',
    'plot_pca_variance',
    'plot_per_class_performance',
]

