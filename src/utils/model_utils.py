"""
Model Utilities for Chromosome Classification

Functions for saving and loading models with metadata.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from .. import config


def get_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Get device (GPU or CPU) based on availability and config.
    
    Args:
        device: Optional device to use. If None, determines automatically.
    
    Returns:
        torch.device: CUDA device if available and USE_GPU=True, else CPU
    """
    if device is not None:
        return device
    
    if config.USE_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def save_model(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model to file with metadata.
    
    Args:
        model: PyTorch model
        filepath: Path to save model
        metadata: Optional metadata dictionary
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }, filepath)
    print(f"Saved model to {filepath}")


def load_model(
    filepath: str,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model from file.
    
    Args:
        filepath: Path to model file
        model: Model instance to load weights into
        device: Device to load model on
    
    Returns:
        Dictionary with loaded metadata
    """
    device = get_device(device)
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = checkpoint.get('metadata', {})
    print(f"Loaded model from {filepath}")
    
    return metadata
