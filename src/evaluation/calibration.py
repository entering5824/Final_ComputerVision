"""
Calibration Module for Chromosome Classification

Provides confidence calibration using Temperature Scaling and evaluation metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional


class TemperatureScaling:
    """
    Temperature Scaling for confidence calibration.
    
    Learns a single temperature parameter T to calibrate model probabilities:
    P_calibrated = softmax(logits / T)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaler.
        
        Args:
            temperature: Initial temperature (default: 1.0)
        """
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature)
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit temperature parameter on validation set.
        
        Args:
            logits: Model logits [n_samples, n_classes]
            labels: True labels [n_samples]
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
    
    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits [n_samples, n_classes]
        
        Returns:
            Calibrated probabilities [n_samples, n_classes]
        """
        return torch.softmax(logits / self.temperature, dim=1)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probabilities: Predicted probabilities [n_samples, n_classes]
        labels: True labels [n_samples]
        n_bins: Number of bins for calibration
    
    Returns:
        Tuple of (ECE, dict with bin statistics)
    """
    # Get predicted class and confidence
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == labels).astype(float)
    
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = {
        'bin_lowers': bin_lowers,
        'bin_uppers': bin_uppers,
        'accuracies': np.zeros(n_bins),
        'confidences': np.zeros(n_bins),
        'counts': np.zeros(n_bins)
    }
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Accuracy and average confidence in this bin
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            bin_stats['accuracies'][i] = accuracy_in_bin
            bin_stats['confidences'][i] = avg_confidence_in_bin
            bin_stats['counts'][i] = np.sum(in_bin)
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece), bin_stats


def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        probabilities: Predicted probabilities [n_samples, n_classes]
        labels: True labels [n_samples]
        n_bins: Number of bins
        save_path: Path to save figure
    
    Returns:
        Dictionary with calibration statistics
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Compute ECE and bin statistics
    ece, bin_stats = expected_calibration_error(probabilities, labels, n_bins)
    
    # Create reliability diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    
    bin_centers = (bin_stats['bin_lowers'] + bin_stats['bin_uppers']) / 2
    bin_width = bin_stats['bin_uppers'] - bin_stats['bin_lowers']
    
    # Plot bars
    accuracies = bin_stats['accuracies']
    confidences = bin_stats['confidences']
    counts = bin_stats['counts']
    
    # Only plot bins with samples
    valid_mask = counts > 0
    
    ax.bar(bin_centers[valid_mask], accuracies[valid_mask], 
           width=bin_width[valid_mask], alpha=0.7, 
           label='Accuracy', color='steelblue', edgecolor='black')
    ax.plot(bin_centers[valid_mask], confidences[valid_mask], 
           marker='o', linestyle='--', label='Confidence', 
           color='red', linewidth=2, markersize=8)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")
    
    plt.close()
    
    return {
        'ece': ece,
        'bin_stats': {
            'centers': bin_centers.tolist(),
            'accuracies': accuracies.tolist(),
            'confidences': confidences.tolist(),
            'counts': counts.tolist()
        }
    }




