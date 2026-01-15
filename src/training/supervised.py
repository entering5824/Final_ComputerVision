"""
Supervised Training Module for Chromosome Classification

Provides supervised training functions for MLP model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

from ..models.mlp import ChromosomeMLP
from .. import config
from ..utils.model_utils import get_device


class ChromosomeDataset(Dataset):
    """
    PyTorch Dataset for chromosome features and labels.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix [n_samples, n_features]
            labels: Label array [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_supervised(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 23,
    hidden_dims: List[int] = None,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    weight_decay: float = 1e-4,
    use_lr_scheduler: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train supervised MLP model.
    
    Args:
        train_features: Training features [n_train, n_features]
        train_labels: Training labels [n_train]
        val_features: Validation features [n_val, n_features]
        val_labels: Validation labels [n_val]
        num_classes: Number of classes
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        early_stopping_patience: Early stopping patience
        weight_decay: L2 regularization weight
        use_lr_scheduler: Whether to use learning rate scheduler
        device: Device (CPU or GPU)
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    device = get_device(device)
    
    if hidden_dims is None:
        hidden_dims = config.HIDDEN_DIMS
    
    # Create datasets
    train_dataset = ChromosomeDataset(train_features, train_labels)
    val_dataset = ChromosomeDataset(val_features, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    input_dim = train_features.shape[1]
    model = ChromosomeMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    if verbose:
        print(f"\nTraining on {device}")
        print(f"Model: MLP with input_dim={input_dim}, hidden_dims={hidden_dims}")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"\nTraining completed. Best validation accuracy: {max(history['val_acc']):.2f}%")
    
    return model, history


def predict_with_confidence(
    model: nn.Module,
    features: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    model_type: str = "mlp",
    images: Optional[List[np.ndarray]] = None,
    blob_features: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with confidence scores.
    
    Args:
        model: Trained model
        features: Feature matrix [n_samples, n_features] (for MLP)
        device: Device (CPU or GPU)
        batch_size: Batch size
        model_type: Model type ("mlp", "cnn", "hybrid") - for compatibility
        images: Images (not used for MLP, for compatibility)
        blob_features: Blob features (not used for MLP, for compatibility)
    
    Returns:
        Tuple of (predictions, confidences) where:
            - predictions: Predicted class indices [n_samples]
            - confidences: Max probability (confidence) [n_samples]
    """
    device = get_device(device)
    
    model.eval()
    
    # Create dataset and dataloader
    # For MLP, we use features directly
    if model_type == "mlp":
        dataset = ChromosomeDataset(features, np.zeros(len(features)))  # Dummy labels
    else:
        # For other model types (not implemented yet)
        raise ValueError(f"Model type {model_type} not supported")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for features_batch, _ in dataloader:
            features_batch = features_batch.to(device)
            
            # Get predictions
            outputs = model(features_batch)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidences, predictions = torch.max(probs, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    confidences = np.concatenate(all_confidences)
    
    return predictions, confidences

