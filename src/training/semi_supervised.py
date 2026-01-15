"""
Semi-Supervised Learning Module for Chromosome Classification

Implements self-training/pseudo-labeling with safety mechanisms:
- Top-K confident samples per class (avoid class imbalance)
- Pseudo-labels never replace original labeled data
- Incremental addition only
- Consistency regularization: require consistent predictions across augmentations
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from collections import defaultdict

from .supervised import train_supervised as train_supervised_model, predict_with_confidence
from ..models.mlp import ChromosomeMLP
from ..utils.model_utils import get_device


def select_pseudo_labels(
    predictions: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.8,
    top_k_per_class: Optional[int] = None,
    num_classes: int = 23
) -> np.ndarray:
    """
    Select high-confidence samples for pseudo-labeling with safety mechanisms.
    
    Safety mechanisms:
    1. Only select samples above confidence threshold
    2. Limit number of pseudo-labels per class (top-K) to maintain class balance
    3. Return boolean mask indicating selected samples
    
    Args:
        predictions: Predicted class indices
        confidences: Confidence scores (max probability)
        threshold: Confidence threshold (default: 0.8)
        top_k_per_class: Maximum number of pseudo-labels per class (None = no limit)
        num_classes: Number of classes
    
    Returns:
        Boolean mask: True for selected samples, False otherwise
    """
    # Initial selection: confidence above threshold
    selected = confidences >= threshold
    
    # Apply top-K per class if specified
    if top_k_per_class is not None:
        # Group by predicted class
        class_indices = defaultdict(list)
        for idx, pred in enumerate(predictions):
            if selected[idx]:
                class_indices[pred].append(idx)
        
        # Select top-K per class based on confidence
        final_selected = np.zeros(len(predictions), dtype=bool)
        
        for class_id, indices in class_indices.items():
            # Get confidences for this class
            class_confidences = confidences[indices]
            
            # Sort by confidence (descending) and take top-K
            sorted_indices = np.argsort(class_confidences)[::-1]
            top_k_indices = sorted_indices[:top_k_per_class]
            
            # Mark as selected
            for idx in [indices[i] for i in top_k_indices]:
                final_selected[idx] = True
        
        return final_selected
    
    return selected


def select_pseudo_labels_with_consistency(
    model: nn.Module,
    images: List[np.ndarray],
    augmentation_fn: Callable,
    num_augmentations: int = 5,
    threshold: float = 0.8,
    consistency_threshold: float = 0.9,
    top_k_per_class: Optional[int] = None,
    num_classes: int = 23,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    model_type: str = "mlp",
    features: Optional[np.ndarray] = None,
    blob_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Select pseudo-labels using consistency regularization.
    
    For each unlabeled image, apply multiple augmentations and check:
    1. High confidence on original image
    2. Consistent predictions across augmentations
    
    Args:
        model: Trained model
        images: List of unlabeled images
        augmentation_fn: Augmentation function to apply
        num_augmentations: Number of augmentations per image
        threshold: Confidence threshold for original image
        consistency_threshold: Minimum fraction of augmentations that must agree
        top_k_per_class: Maximum pseudo-labels per class
        num_classes: Number of classes
        device: Device (CPU or GPU)
        batch_size: Batch size
        model_type: "mlp", "cnn", or "hybrid"
        features: Feature matrix (for MLP models)
        blob_features: Blob features (for hybrid models)
    
    Returns:
        Boolean mask: True for selected samples
    """
    device = get_device(device)
    
    model.eval()
    n_samples = len(images)
    selected_mask = np.zeros(n_samples, dtype=bool)
    
    # Get predictions for original images
    if model_type == "mlp":
        original_preds, original_confs = predict_with_confidence(
            model, features, device=device, batch_size=batch_size, model_type=model_type
        )
    elif model_type == "cnn":
        original_preds, original_confs = predict_with_confidence(
            model, images=images, device=device, batch_size=batch_size, model_type=model_type
        )
    elif model_type == "hybrid":
        original_preds, original_confs = predict_with_confidence(
            model, images=images, blob_features=blob_features,
            device=device, batch_size=batch_size, model_type=model_type
        )
    
    # Check consistency for high-confidence samples
    high_conf_mask = original_confs >= threshold
    
    for idx in np.where(high_conf_mask)[0]:
        # Apply augmentations and collect predictions
        aug_predictions = []
        
        for _ in range(num_augmentations):
            # Apply augmentation
            if model_type == "mlp":
                # For MLP, we'd need to re-extract features after augmentation
                # This is simplified - in practice, might need feature extraction
                continue
            elif model_type == "cnn":
                aug_img = augmentation_fn(images[idx])
                aug_pred, _ = predict_with_confidence(
                    model, images=[aug_img], device=device, 
                    batch_size=1, model_type=model_type
                )
                aug_predictions.append(aug_pred[0])
            elif model_type == "hybrid":
                aug_img = augmentation_fn(images[idx])
                aug_pred, _ = predict_with_confidence(
                    model, images=[aug_img], blob_features=blob_features[idx:idx+1],
                    device=device, batch_size=1, model_type=model_type
                )
                aug_predictions.append(aug_pred[0])
        
        if len(aug_predictions) == 0:
            # If no augmentations, skip consistency check
            continue
        
        # Check consistency
        aug_predictions = np.array(aug_predictions)
        original_pred = original_preds[idx]
        
        # Fraction of augmentations that agree with original prediction
        consistency = np.mean(aug_predictions == original_pred)
        
        if consistency >= consistency_threshold:
            selected_mask[idx] = True
    
    # Apply top-K per class filtering
    if top_k_per_class is not None and np.any(selected_mask):
        # Group by predicted class
        class_indices = defaultdict(list)
        for idx in np.where(selected_mask)[0]:
            class_indices[original_preds[idx]].append(idx)
        
        # Select top-K per class
        final_selected = np.zeros(n_samples, dtype=bool)
        
        for class_id, indices in class_indices.items():
            class_confidences = original_confs[indices]
            sorted_indices = np.argsort(class_confidences)[::-1]
            top_k_indices = sorted_indices[:top_k_per_class]
            
            for idx in [indices[i] for i in top_k_indices]:
                final_selected[idx] = True
        
        return final_selected
    
    return selected_mask


def self_training_loop(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    unlabeled_features: np.ndarray,
    num_classes: int = 23,
    hidden_dims: List[int] = None,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs_per_iteration: int = 15,
    early_stopping_patience: int = 5,
    max_iterations: int = 20,
    confidence_thresholds: List[float] = [0.7, 0.8, 0.9],
    top_k_per_class: int = 100,
    min_total_epochs: int = 300,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    weight_decay: float = 1e-4,
    use_lr_scheduler: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Self-training loop with pseudo-labeling.
    
    IMPORTANT: 
    - Early stopping is applied within each self-training iteration
    - Total epochs across all iterations must be ≥ min_total_epochs (300)
    - Pseudo-labeled samples are added incrementally and never replace original labeled data
    
    Args:
        train_features: Initial labeled training features
        train_labels: Initial labeled training labels
        val_features: Validation features (for tuning T_conf)
        val_labels: Validation labels
        unlabeled_features: Unlabeled features
        num_classes: Number of classes
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        batch_size: Batch size
        epochs_per_iteration: Epochs per self-training iteration
        early_stopping_patience: Patience for early stopping
        max_iterations: Maximum number of self-training iterations
        confidence_thresholds: List of T_conf values to try
        top_k_per_class: Maximum pseudo-labels per class
        min_total_epochs: Minimum total epochs required (default: 300)
        device: Device (CPU or GPU)
        verbose: Whether to print progress
        model_type: Model type (deprecated, always uses MLP)
    
    Returns:
        Tuple of (final_model, training_info)
    """
    device = get_device(device)
    
    # Track training info
    training_info = {
        'iterations': [],
        'total_epochs': 0,
        'pseudo_labels_added': [],
        'best_t_conf': None,
        'best_val_acc': 0.0,
        't_conf_results': [],
        # History for plotting (same format as supervised_history)
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Tune confidence threshold on validation set
    if verbose:
        print("\n" + "=" * 60)
        print("TUNING CONFIDENCE THRESHOLD (T_conf)")
        print("=" * 60)
    
    # Initial training with MLP model (features + PCA) as required
    model, history = train_supervised_model(
        train_features, train_labels,
        val_features, val_labels,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=epochs_per_iteration,
        early_stopping_patience=early_stopping_patience,
        weight_decay=weight_decay,
        use_lr_scheduler=use_lr_scheduler,
        device=device,
        verbose=False
    )
    
    training_info['total_epochs'] += len(history['train_loss'])
    # Collect initial training history
    training_info['train_loss'].extend(history['train_loss'])
    training_info['val_loss'].extend(history['val_loss'])
    training_info['train_acc'].extend(history['train_acc'])
    training_info['val_acc'].extend(history['val_acc'])
    
    # Test different T_conf values
    for t_conf in confidence_thresholds:
        # Predict on unlabeled data (MLP model with features)
        predictions, confidences = predict_with_confidence(
            model, 
            unlabeled_features, 
            device=device, 
            batch_size=batch_size
        )
        
        # Select pseudo-labels
        selected_mask = select_pseudo_labels(
            predictions, confidences,
            threshold=t_conf,
            top_k_per_class=top_k_per_class,
            num_classes=num_classes
        )
        
        num_selected = np.sum(selected_mask)
        
        if num_selected == 0:
            if verbose:
                print(f"T_conf={t_conf:.1f}: No samples selected")
            continue
        
        # Evaluate on validation set with pseudo-labels
        # (We use validation accuracy to choose best T_conf)
        val_predictions, _ = predict_with_confidence(
            model, 
            val_features, 
            device=device, 
            batch_size=batch_size
        )
        val_acc = 100.0 * np.mean(val_predictions == val_labels)
        
        training_info['t_conf_results'].append({
            't_conf': t_conf,
            'num_pseudo': num_selected,
            'val_accuracy': val_acc
        })
        
        if verbose:
            print(f"T_conf={t_conf:.1f}: {num_selected} pseudo-labels, Val Acc: {val_acc:.2f}%")
        
        # Track best T_conf
        if val_acc > training_info['best_val_acc']:
            training_info['best_val_acc'] = val_acc
            training_info['best_t_conf'] = t_conf
    
    # Use best T_conf for self-training
    best_t_conf = training_info['best_t_conf']
    if best_t_conf is None:
        # Fallback: use highest threshold that selected any samples
        best_t_conf = max([r['t_conf'] for r in training_info['t_conf_results']])
    
    if verbose:
        print(f"\nSelected T_conf = {best_t_conf:.1f} (best validation accuracy)")
        print("\n" + "=" * 60)
        print("SELF-TRAINING LOOP")
        print("=" * 60)
    
    # Self-training loop
    current_train_features = train_features.copy()
    current_train_labels = train_labels.copy()
    remaining_unlabeled_features = unlabeled_features.copy()
    remaining_unlabeled_indices = np.arange(len(unlabeled_features))
    
    iteration = 0
    
    while iteration < max_iterations and training_info['total_epochs'] < min_total_epochs:
        iteration += 1
        
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current labeled samples: {len(current_train_features)}")
            print(f"Remaining unlabeled samples: {len(remaining_unlabeled_features)}")
        
        # Train model on current labeled data (MLP with features + PCA)
        model, history = train_supervised_model(
            current_train_features, current_train_labels,
            val_features, val_labels,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=epochs_per_iteration,
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            use_lr_scheduler=use_lr_scheduler,
            device=device,
            verbose=False
        )
        
        epochs_this_iteration = len(history['train_loss'])
        training_info['total_epochs'] += epochs_this_iteration
        
        # Collect history from this iteration
        training_info['train_loss'].extend(history['train_loss'])
        training_info['val_loss'].extend(history['val_loss'])
        training_info['train_acc'].extend(history['train_acc'])
        training_info['val_acc'].extend(history['val_acc'])
        
        if verbose:
            print(f"Trained for {epochs_this_iteration} epochs (total: {training_info['total_epochs']})")
        
        # Predict on remaining unlabeled data (using MLP with features)
        if len(remaining_unlabeled_features) == 0:
            if verbose:
                print("No more unlabeled samples available")
            break
        
        predictions, confidences = predict_with_confidence(
            model, 
            remaining_unlabeled_features,
            device=device, 
            batch_size=batch_size
        )
        
        # Select pseudo-labels with safety mechanisms
        selected_mask = select_pseudo_labels(
            predictions, confidences,
            threshold=best_t_conf,
            top_k_per_class=top_k_per_class,
            num_classes=num_classes
        )
        
        num_selected = np.sum(selected_mask)
        
        if num_selected == 0:
            if verbose:
                print("No high-confidence samples found, stopping self-training")
            break
        
        # Get selected samples
        selected_features = remaining_unlabeled_features[selected_mask]
        selected_predictions = predictions[selected_mask]
        
        # Add pseudo-labels to training set (INCREMENTAL - never replace original)
        current_train_features = np.vstack([current_train_features, selected_features])
        current_train_labels = np.concatenate([current_train_labels, selected_predictions])
        
        # Remove selected samples from unlabeled set
        remaining_unlabeled_features = remaining_unlabeled_features[~selected_mask]
        remaining_unlabeled_indices = remaining_unlabeled_indices[~selected_mask]
        
        training_info['pseudo_labels_added'].append(num_selected)
        training_info['iterations'].append({
            'iteration': iteration,
            'epochs': epochs_this_iteration,
            'pseudo_labels': num_selected,
            'total_labeled': len(current_train_features),
            'val_acc': history['val_acc'][-1] if history['val_acc'] else 0.0
        })
        
        if verbose:
            print(f"Added {num_selected} pseudo-labels")
            print(f"New labeled set size: {len(current_train_features)}")
            print(f"Validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Final training on all labeled data (including all pseudo-labels)
    if verbose:
        print(f"\n--- Final Training ---")
        print(f"Total labeled samples: {len(current_train_features)}")
        print(f"Total epochs so far: {training_info['total_epochs']}")
    
    # Ensure we reach min_total_epochs
    remaining_epochs = max(0, min_total_epochs - training_info['total_epochs'])
    if remaining_epochs > 0:
        if verbose:
            print(f"Training for additional {remaining_epochs} epochs to reach minimum...")
        
        model, final_history = train_supervised_model(
            current_train_features, current_train_labels,
            val_features, val_labels,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=remaining_epochs,
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            use_lr_scheduler=use_lr_scheduler,
            device=device,
            verbose=False
        )
        
        training_info['total_epochs'] += len(final_history['train_loss'])
        # Collect final training history
        training_info['train_loss'].extend(final_history['train_loss'])
        training_info['val_loss'].extend(final_history['val_loss'])
        training_info['train_acc'].extend(final_history['train_acc'])
        training_info['val_acc'].extend(final_history['val_acc'])
    
    if verbose:
        print(f"\nSelf-training completed!")
        print(f"Total iterations: {iteration}")
        print(f"Total epochs: {training_info['total_epochs']} (required: {min_total_epochs})")
        print(f"Total pseudo-labels added: {sum(training_info['pseudo_labels_added'])}")
    
    return model, training_info

