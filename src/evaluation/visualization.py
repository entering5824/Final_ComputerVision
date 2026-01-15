"""
Visualization Module for Chromosome Classification

Generates plots for training curves, confusion matrix, PCA variance, and per-class performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from pathlib import Path


def plot_training_curves(
    supervised_history: Dict[str, List[float]],
    semi_supervised_history: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training curves (loss and accuracy).
    
    Args:
        supervised_history: Training history from supervised model
        semi_supervised_history: Training history from semi-supervised model (optional)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_sup = range(1, len(supervised_history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs_sup, supervised_history['train_loss'], 
                 label='Supervised Train', marker='o', markersize=3)
    axes[0].plot(epochs_sup, supervised_history['val_loss'], 
                 label='Supervised Val', marker='s', markersize=3)
    
    if semi_supervised_history and 'train_loss' in semi_supervised_history and len(semi_supervised_history['train_loss']) > 0:
        epochs_semi = range(1, len(semi_supervised_history['train_loss']) + 1)
        axes[0].plot(epochs_semi, semi_supervised_history['train_loss'], 
                     label='Semi-Supervised Train', marker='^', markersize=3)
        axes[0].plot(epochs_semi, semi_supervised_history['val_loss'], 
                     label='Semi-Supervised Val', marker='v', markersize=3)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs_sup, supervised_history['train_acc'], 
                 label='Supervised Train', marker='o', markersize=3)
    axes[1].plot(epochs_sup, supervised_history['val_acc'], 
                 label='Supervised Val', marker='s', markersize=3)
    
    if semi_supervised_history and 'train_acc' in semi_supervised_history and len(semi_supervised_history['train_acc']) > 0:
        epochs_semi = range(1, len(semi_supervised_history['train_acc']) + 1)
        axes[1].plot(epochs_semi, semi_supervised_history['train_acc'], 
                     label='Semi-Supervised Train', marker='^', markersize=3)
        axes[1].plot(epochs_semi, semi_supervised_history['val_acc'], 
                     label='Semi-Supervised Val', marker='v', markersize=3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.close()


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot PCA explained variance.
    
    Args:
        explained_variance_ratio: Explained variance ratio for each component
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_components = len(explained_variance_ratio)
    components = range(1, n_components + 1)
    
    # Explained variance per component
    axes[0].plot(components, explained_variance_ratio, marker='o', markersize=4)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Explained Variance by Component')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumulative = np.cumsum(explained_variance_ratio)
    axes[1].plot(components, cumulative, marker='o', markersize=4, color='orange')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA variance plot to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_per_class_performance(
    metrics: Dict[str, Any],
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot per-class performance metrics.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        class_names: List of class names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # Precision
    axes[0].bar(x, metrics['precision_per_class'], width)
    axes[0].set_xlabel('Chromosome')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Recall
    axes[1].bar(x, metrics['recall_per_class'], width, color='orange')
    axes[1].set_xlabel('Chromosome')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # F1-Score
    axes[2].bar(x, metrics['f1_per_class'], width, color='green')
    axes[2].set_xlabel('Chromosome')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class performance to {save_path}")
    
    plt.close()




