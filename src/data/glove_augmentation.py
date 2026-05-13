"""
Glove region augmentation for training.
Applies realistic augmentations to glove regions using masks.
Ensures temporal consistency: same augmentation parameters for entire video clip.
"""

import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from pathlib import Path
import sys

# Import augmentation functions from scripts
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'data_processing'))
from glove_augmentation_realistic import (
    AUGMENTATION_FUNCTIONS,
    SKIN_TONES,
    FABRIC_COLORS,
)


class GloveAugmentation:
    """
    Video-level glove region augmentation for training.
    
    Features:
    - Applies augmentation to entire video clip (temporal consistency)
    - Only augments samples with valid glove masks
    - Supports multiple augmentation types
    - Random selection of augmentation method per clip
    """
    
    def __init__(
        self,
        enabled: bool = True,
        augmentation_prob: float = 0.5,
        augmentation_types: Optional[list] = None,
    ):
        """
        Args:
            enabled: Whether to enable glove augmentation
            augmentation_prob: Probability of applying augmentation to a clip
            augmentation_types: List of augmentation types to use. If None, uses default set.
        """
        self.enabled = enabled
        self.augmentation_prob = augmentation_prob
        
        # Default augmentation types (balanced mix of realistic augmentations)
        if augmentation_types is None:
            self.augmentation_types = [
                # Solid skin colors (solid-color replacement) - 7 types
                'skin_color_very_light',
                'skin_color_light',
                'skin_color_medium_light',
                'skin_color_medium',
                'skin_color_medium_dark',
                'skin_color_dark',
                'skin_color_very_dark',
                
                # Skin tones (preserve texture) - choose 3 representative tones
                'skin_tone_light',
                'skin_tone_medium',
                'skin_tone_dark',
                
                # Fabric colors - choose 4 common colors
                'fabric_black',
                'fabric_white',
                'fabric_blue',
                'fabric_brown',
                
                # Patterns - 4 types
                'horizontal_stripes',
                'vertical_stripes',
                'checkerboard',
                'dots_pattern',
                
                # Natural variations - 2 types
                'brightness_variation',
                'subtle_noise',
            ]
        else:
            self.augmentation_types = augmentation_types
        
        # Validate augmentation types
        for aug_type in self.augmentation_types:
            if aug_type not in AUGMENTATION_FUNCTIONS:
                raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def __call__(
        self,
        frames: np.ndarray,
        masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply glove augmentation to video frames.
        
        Args:
            frames: (T, H, W, 3) numpy array, range [0, 255], uint8
            masks: (T, H, W) numpy array, binary mask (0 or 255), uint8
                   If None, no augmentation is applied
        
        Returns:
            augmented_frames: (T, H, W, 3) numpy array, range [0, 255], uint8
        """
        # Skip if disabled or no masks
        if not self.enabled or masks is None:
            return frames
        
        # Skip with probability
        if np.random.rand() > self.augmentation_prob:
            return frames
        
        # Check if masks are valid (at least one frame has non-zero mask)
        if not np.any(masks > 0):
            return frames
        
        # Randomly select augmentation type
        aug_type = np.random.choice(self.augmentation_types)
        
        # Try vectorized augmentation first (fast path for simple augmentations)
        vectorized_result = self._try_vectorized_augmentation(frames, masks, aug_type)
        if vectorized_result is not None:
            return vectorized_result
        
        # Fall back to per-frame augmentation for complex patterns
        # Fix random seed for this clip to ensure temporal consistency
        seed = np.random.randint(0, 2**31)
        
        # Apply augmentation to each frame
        augmented_frames = frames.copy()
        T = frames.shape[0]
        
        for t in range(T):
            # Reset random seed for each frame to ensure consistent augmentation
            np.random.seed(seed)
            
            frame = frames[t]
            mask = masks[t]
            
            # Skip if no mask in this frame
            if not np.any(mask > 0):
                continue
            
            # Apply augmentation
            try:
                aug_func = AUGMENTATION_FUNCTIONS[aug_type]
                augmented_frame = aug_func(frame, mask)
                augmented_frames[t] = augmented_frame
            except Exception as e:
                # If augmentation fails, keep original frame
                print(f"Warning: Glove augmentation failed for frame {t}: {e}")
                continue
        
        return augmented_frames
    
    def _try_vectorized_augmentation(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        aug_type: str,
    ) -> Optional[np.ndarray]:
        """
        Try to apply vectorized augmentation (fast path).
        Returns None if augmentation type doesn't support vectorization.
        
        Vectorized augmentations (10-20x faster):
        - Solid color replacements (skin_color_*, fabric_*)
        - Brightness/noise variations
        
        Non-vectorized (need per-frame processing):
        - Patterns with random generation (stripes, checkerboard, dots)
        - Skin tone with LAB color space conversion
        """
        # Solid skin color augmentations (vectorized)
        if aug_type.startswith('skin_color_'):
            return self._vectorized_skin_color(frames, masks, aug_type)
        
        # Fabric color augmentations (vectorized)
        if aug_type.startswith('fabric_'):
            return self._vectorized_fabric_color(frames, masks, aug_type)
        
        # Brightness variation (vectorized)
        if aug_type == 'brightness_variation':
            return self._vectorized_brightness(frames, masks)
        
        # Subtle noise (vectorized)
        if aug_type == 'subtle_noise':
            return self._vectorized_noise(frames, masks)
        
        # Other augmentations need per-frame processing
        return None
    
    def _vectorized_skin_color(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        aug_type: str,
    ) -> np.ndarray:
        """Vectorized solid skin color replacement."""
        # Map augmentation type to skin tone index
        skin_tone_map = {
            'skin_color_very_light': 0,
            'skin_color_light': 1,
            'skin_color_medium_light': 2,
            'skin_color_medium': 3,
            'skin_color_medium_dark': 4,
            'skin_color_dark': 5,
            'skin_color_very_dark': 6,
        }
        
        if aug_type == 'skin_color_random':
            tone_idx = np.random.randint(0, len(SKIN_TONES))
        else:
            tone_idx = skin_tone_map.get(aug_type, 3)
        
        # Get color
        color = np.array(SKIN_TONES[tone_idx], dtype=np.uint8)
        
        # Vectorized replacement: (T, H, W, 3)
        augmented = frames.copy()
        mask_expanded = (masks > 0)[..., None]  # (T, H, W, 1)
        augmented = np.where(mask_expanded, color, augmented)
        
        return augmented.astype(np.uint8)
    
    def _vectorized_fabric_color(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        aug_type: str,
    ) -> np.ndarray:
        """Vectorized fabric color replacement."""
        # Map augmentation type to fabric color index
        fabric_color_map = {
            'fabric_black': 0,
            'fabric_white': 4,
            'fabric_blue': 6,
            'fabric_brown': 9,
        }
        
        color_idx = fabric_color_map.get(aug_type, 0)
        color = np.array(FABRIC_COLORS[color_idx], dtype=np.uint8)
        
        # Vectorized replacement
        augmented = frames.copy()
        mask_expanded = (masks > 0)[..., None]  # (T, H, W, 1)
        augmented = np.where(mask_expanded, color, augmented)
        
        return augmented.astype(np.uint8)
    
    def _vectorized_brightness(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Vectorized brightness adjustment."""
        # Random brightness factor (same for all frames)
        factor = np.random.uniform(0.6, 1.4)
        
        # Vectorized brightness adjustment
        augmented = frames.copy()
        mask_expanded = (masks > 0)[..., None]  # (T, H, W, 1)
        
        # Apply brightness only to masked regions
        masked_region = frames.astype(np.float32) * factor
        masked_region = np.clip(masked_region, 0, 255)
        
        augmented = np.where(mask_expanded, masked_region, frames)
        
        return augmented.astype(np.uint8)
    
    def _vectorized_noise(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Vectorized noise addition."""
        # Generate noise for all frames at once
        T, H, W, C = frames.shape
        noise = np.random.normal(0, 5, (T, H, W, C)).astype(np.float32)
        
        # Vectorized noise addition
        augmented = frames.copy()
        mask_expanded = (masks > 0)[..., None]  # (T, H, W, 1)
        
        # Add noise only to masked regions
        noisy_region = frames.astype(np.float32) + noise
        noisy_region = np.clip(noisy_region, 0, 255)
        
        augmented = np.where(mask_expanded, noisy_region, frames)
        
        return augmented.astype(np.uint8)


def load_glove_masks_from_hdf5(hdf5_file, frame_indices: list) -> Optional[np.ndarray]:
    """
    Load glove masks from HDF5 file.
    
    Args:
        hdf5_file: Open h5py.File object
        frame_indices: List of frame indices to load
    
    Returns:
        masks: (T, H, W) numpy array, binary mask (0 or 255), uint8
               Returns None if no masks available
    """
    import h5py
    
    # Check if masks exist in HDF5
    if 'masks' not in hdf5_file or 'glove_masks' not in hdf5_file['masks']:
        return None
    
    glove_masks_group = hdf5_file['masks/glove_masks']
    
    # Check if valid_frames exists
    if 'valid_frames' not in glove_masks_group:
        return None
    
    valid_frames = glove_masks_group['valid_frames'][:]
    
    # If no valid frames, return None
    if not np.any(valid_frames):
        return None
    
    # Load masks for requested frames
    masks_dataset = glove_masks_group['masks']
    T = len(frame_indices)
    H, W = masks_dataset.shape[1], masks_dataset.shape[2]
    
    masks = np.zeros((T, H, W), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx < len(valid_frames) and valid_frames[frame_idx]:
            masks[i] = masks_dataset[frame_idx]
    
    return masks


def get_glove_augmentation(config: dict, is_training: bool = True):
    """
    Get glove augmentation instance based on config.
    
    Args:
        config: Configuration dictionary
        is_training: Whether in training mode
    
    Returns:
        GloveAugmentation instance or None
    """
    if not is_training:
        return None
    
    # Check if glove augmentation is enabled in config
    glove_aug_config = config['data'].get('glove_augmentation', {})
    enabled = glove_aug_config.get('enabled', False)
    
    if not enabled:
        return None
    
    return GloveAugmentation(
        enabled=True,
        augmentation_prob=glove_aug_config.get('prob', 0.5),
        augmentation_types=glove_aug_config.get('types', None),
    )
