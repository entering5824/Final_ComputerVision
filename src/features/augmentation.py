"""
Data Augmentation Module for Chromosome Classification

Provides augmentation functions for training data augmentation.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import random

from .. import config


class ChromosomeAugmentation:
    """
    Augmentation class for chromosome images.
    
    Provides rotation, shift, flip, and scaling augmentations.
    """
    
    def __init__(
        self,
        rotation_range: float = None,
        shift_range: float = None,
        scale_range: float = None,
        flip_prob: float = 0.0,  # Be careful with vertical flip for chromosomes
        random_seed: Optional[int] = None
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range: Maximum rotation angle in degrees (±)
            shift_range: Maximum shift as fraction of image size (±)
            scale_range: Maximum scale change as fraction (±)
            flip_prob: Probability of vertical flip (default: 0.0, be careful)
            random_seed: Random seed for reproducibility
        """
        self.rotation_range = rotation_range or config.ROTATION_RANGE
        self.shift_range = shift_range or config.SHIFT_RANGE
        self.scale_range = scale_range or config.SCALE_RANGE
        self.flip_prob = flip_prob
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def rotate(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate image by random angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (if None, random)
        
        Returns:
            Rotated image
        """
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def shift(self, image: np.ndarray, dx: Optional[float] = None, dy: Optional[float] = None) -> np.ndarray:
        """
        Shift image by random amount.
        
        Args:
            image: Input image
            dx: Horizontal shift as fraction of width (if None, random)
            dy: Vertical shift as fraction of height (if None, random)
        
        Returns:
            Shifted image
        """
        h, w = image.shape[:2]
        
        if dx is None:
            dx = np.random.uniform(-self.shift_range, self.shift_range) * w
        else:
            dx = dx * w
        
        if dy is None:
            dy = np.random.uniform(-self.shift_range, self.shift_range) * h
        else:
            dy = dy * h
        
        # Get translation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Apply translation
        shifted = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return shifted
    
    def scale(self, image: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
        """
        Scale image by random factor.
        
        Args:
            image: Input image
            scale_factor: Scale factor (if None, random)
        
        Returns:
            Scaled image (cropped/padded to original size)
        """
        h, w = image.shape[:2]
        
        if scale_factor is None:
            scale_factor = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        
        # Scale image
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if new_h > h or new_w > w:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            scaled = scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad with reflection
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                       cv2.BORDER_REFLECT)
        
        return scaled
    
    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """
        Flip image vertically.
        
        WARNING: Use with caution for chromosomes as orientation may have biological meaning.
        
        Args:
            image: Input image
        
        Returns:
            Vertically flipped image
        """
        return cv2.flip(image, 0)
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally.
        
        Args:
            image: Input image
        
        Returns:
            Horizontally flipped image
        """
        return cv2.flip(image, 1)
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image
        
        Returns:
            Augmented image
        """
        aug_image = image.copy()
        
        # Random rotation
        if self.rotation_range > 0:
            aug_image = self.rotate(aug_image)
        
        # Random shift
        if self.shift_range > 0:
            aug_image = self.shift(aug_image)
        
        # Random scale
        if self.scale_range > 0:
            aug_image = self.scale(aug_image)
        
        # Random vertical flip (with probability)
        if self.flip_prob > 0 and np.random.random() < self.flip_prob:
            aug_image = self.flip_vertical(aug_image)
        
        return aug_image


def create_augmentation_pipeline(
    rotation_range: float = None,
    shift_range: float = None,
    scale_range: float = None,
    flip_prob: float = 0.0
) -> ChromosomeAugmentation:
    """
    Create augmentation pipeline.
    
    Args:
        rotation_range: Maximum rotation angle
        shift_range: Maximum shift fraction
        scale_range: Maximum scale change fraction
        flip_prob: Probability of vertical flip
    
    Returns:
        ChromosomeAugmentation instance
    """
    return ChromosomeAugmentation(
        rotation_range=rotation_range,
        shift_range=shift_range,
        scale_range=scale_range,
        flip_prob=flip_prob
    )

