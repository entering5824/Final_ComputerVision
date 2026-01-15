"""
Metrics Module for Chromosome Classification

Computes evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from ..training.supervised import predict_with_confidence
from ..utils.model_utils import get_device


def calculate_metrics(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    device: Optional[torch.device] = None,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate model and compute metrics.
    
    Args:
        model: Trained model
        features: Feature matrix
        labels: True labels
        class_names: List of class names
        device: Device (CPU or GPU)
        batch_size: Batch size
    
    Returns:
        Dictionary with metrics
    """
    device = get_device(device)
    
    # Predict
    predictions, confidences = predict_with_confidence(
        model, features, device=device, batch_size=batch_size
    )
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    # Macro F1 (important for imbalanced classes as per requirements)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(len(class_names)):
        mask = labels == i
        if np.sum(mask) > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
        else:
            acc = 0.0
        per_class_acc.append(acc)
    
    return {
        'accuracy': accuracy,
        'precision': precision_weighted,
        'recall': recall_weighted,
        'f1_score': f1_weighted,
        'f1_macro': f1_macro,  # Macro F1 (important for imbalanced classes)
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': labels.tolist(),
        'confidences': confidences.tolist()
    }


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    metrics = {}
    for i, class_name in enumerate(class_names):
        mask_true = y_true == i
        mask_pred = y_pred == i
        
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        tn = np.sum((y_true != i) & (y_pred != i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(mask_true)
        
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(support)
        }
    
    return metrics


def compare_models(
    supervised_results: Dict[str, Any],
    semi_supervised_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare supervised and semi-supervised models.
    
    Args:
        supervised_results: Results from supervised model
        semi_supervised_results: Results from semi-supervised model
        save_path: Path to save comparison table (JSON)
    
    Returns:
        Comparison dictionary
    """
    import json
    from pathlib import Path
    
    comparison = {
        'supervised': {
            'accuracy': supervised_results['accuracy'],
            'precision': supervised_results['precision'],
            'recall': supervised_results['recall'],
            'f1_score': supervised_results['f1_score'],
            'f1_macro': supervised_results.get('f1_macro', 0.0)
        },
        'semi_supervised': {
            'accuracy': semi_supervised_results['accuracy'],
            'precision': semi_supervised_results['precision'],
            'recall': semi_supervised_results['recall'],
            'f1_score': semi_supervised_results['f1_score'],
            'f1_macro': semi_supervised_results.get('f1_macro', 0.0)
        },
        'improvement': {
            'accuracy': semi_supervised_results['accuracy'] - supervised_results['accuracy'],
            'precision': semi_supervised_results['precision'] - supervised_results['precision'],
            'recall': semi_supervised_results['recall'] - supervised_results['recall'],
            'f1_score': semi_supervised_results['f1_score'] - supervised_results['f1_score'],
            'f1_macro': semi_supervised_results.get('f1_macro', 0.0) - supervised_results.get('f1_macro', 0.0)
        }
    }
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved comparison to {save_path}")
    
    return comparison


def print_comparison_table(comparison: Dict[str, Any]):
    """Print comparison table in a readable format."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Supervised':<15} {'Semi-Supervised':<15} {'Improvement':<15}")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'f1_macro']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score (Weighted)', 'F1-Score (Macro)']
    
    for metric, name in zip(metrics, metric_names):
        sup_val = comparison['supervised'][metric]
        semi_val = comparison['semi_supervised'][metric]
        improvement = comparison['improvement'][metric]
        
        print(f"{name:<20} {sup_val*100:>6.2f}%       {semi_val*100:>6.2f}%       {improvement*100:>+6.2f}%")
    
    print("=" * 60)




