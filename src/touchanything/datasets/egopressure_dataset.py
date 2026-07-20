"""
EgoPressure Dataset Loader for TouchAnything Training Framework.

Adapts EgoPressure dataset to our training pipeline:
- Input: Egocentric RGB video + Hand pose (21 joints x 3D per hand)
- Output: Pressure maps in UV space (224x224 → 16x16)

Pose format: Bilateral (42, 3) = [left_21, right_21]
  - Active hand: original 21-joint 3D coordinates from EgoPressure
  - Missing hand: filled with MISSING_JOINT_VALUE (-999.0) sentinel
  - Model should learn to ignore sentinel-valued joints

Pressure format: Bilateral (2, 16, 16) = [left, right]
  - UV pressure_map (224x224) downsampled to 16x16 via INTER_AREA
  - pressure_map is per-frame normalized by EgoPressure (~[0, 1.3])
  - We clip negatives (interpolation artifacts) but keep values >1
"""
import os
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EgoPressureDataset(Dataset):
    """
    EgoPressure dataset loader compatible with TouchAnything training framework.
    
    Dataset structure:
        egopressure/
        ├── p_001/
        │   ├── p_001_gesture_name_hand.tar (extracted to folders)
        │   └── ...
        └── ...
    
    Each sequence contains:
        - {frame_id}.anno.pkl: Hand pose (21 joints) + MANO params
        - {frame_id}.force.bin: Pressure map (185x105)
        - {frame_id}.cam-d_rgb.jpeg: Egocentric RGB (1920x1080)
        - {frame_id}.cam-d_mask.png: Hand mask
    """
    
    def __init__(
        self,
        data_root: str,
        sequences: List[str],
        clip_length: int = 16,
        image_size: int = 224,
        pressure_size: int = 16,
        stride: int = 1,
        require_annotation: bool = True,
        require_pressure: bool = True,
        augmentation: bool = False,
    ):
        """
        Args:
            data_root: Root directory of EgoPressure dataset
            sequences: List of sequence names (e.g., ['p_001/gesture1', 'p_002/gesture2'])
            clip_length: Number of frames per clip
            image_size: Target image size (default: 224 for DINOv2)
            pressure_size: Target pressure map size (default: 16)
            stride: Stride for sliding window
            require_annotation: Only use frames with has_annotation=True
            require_pressure: Only use frames with non-zero pressure
            augmentation: Apply data augmentation
        """
        self.data_root = Path(data_root)
        self.clip_length = clip_length
        self.image_size = image_size
        self.pressure_size = pressure_size
        self.stride = stride
        self.require_annotation = require_annotation
        self.require_pressure = require_pressure
        self.augmentation = augmentation
        
        # Build clip index
        self.clips = self._build_clip_index(sequences)
        
        logger.info(f"EgoPressureDataset initialized:")
        logger.info(f"  Sequences: {len(sequences)}")
        logger.info(f"  Total clips: {len(self.clips)}")
        logger.info(f"  Clip length: {clip_length}")
        logger.info(f"  Image size: {image_size}")
        logger.info(f"  Pressure size: {pressure_size}x{pressure_size}")
    
    def _build_clip_index(self, sequences: List[str]) -> List[Dict]:
        """Build index of all valid clips."""
        clips = []
        
        for seq_name in sequences:
            seq_path = self.data_root / seq_name
            
            if not seq_path.exists():
                logger.warning(f"Sequence not found: {seq_path}")
                continue
            
            # Get all frame IDs
            anno_files = sorted(seq_path.glob("*.anno.pkl"))
            frame_ids = [f.stem.split('.')[0] for f in anno_files]
            
            if len(frame_ids) < self.clip_length:
                logger.warning(f"Sequence {seq_name} too short: {len(frame_ids)} frames")
                continue
            
            # Filter frames if needed
            if self.require_annotation or self.require_pressure:
                valid_frames = self._filter_valid_frames(seq_path, frame_ids)
            else:
                valid_frames = frame_ids
            
            if len(valid_frames) < self.clip_length:
                logger.warning(f"Sequence {seq_name} has only {len(valid_frames)} valid frames")
                continue
            
            # Create clips with sliding window
            for start_idx in range(0, len(valid_frames) - self.clip_length + 1, self.stride):
                clip_frames = valid_frames[start_idx:start_idx + self.clip_length]
                
                clips.append({
                    'sequence': seq_name,
                    'seq_path': seq_path,
                    'frame_ids': clip_frames,
                    'start_idx': start_idx,
                })
            
            logger.info(f"  {seq_name}: {len(valid_frames)} valid frames, {len(clips)} clips")
        
        return clips
    
    def _filter_valid_frames(self, seq_path: Path, frame_ids: List[str]) -> List[str]:
        """Filter frames based on annotation and pressure requirements."""
        valid_frames = []
        
        for frame_id in frame_ids:
            anno_file = seq_path / f"{frame_id}.anno.pkl"
            
            try:
                with open(anno_file, 'rb') as f:
                    anno = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {anno_file}: {e}")
                continue
            
            # Check annotation
            if self.require_annotation:
                if not anno.get('has_annotation', False):
                    continue
            
            # Must have joint_position (some annotated frames lack it)
            if anno.get('joint_position', None) is None:
                continue
            
            # Check pressure (use UV pressure map)
            if self.require_pressure:
                uv_pressure = anno.get('pressure_map', None)
                if uv_pressure is None or uv_pressure.max() <= 0:
                    continue
            
            valid_frames.append(frame_id)
        
        return valid_frames
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load one clip.
        
        Returns:
            dict with keys:
                - video: (T, 3, H, W) - RGB frames
                - poses: (T, 48, 3) - Hand poses (bilateral format)
                - tactile: (T, 2, 16, 16) - Pressure maps (bilateral format)
                - hand_side: str - 'left' or 'right'
                - frame_ids: List[str] - Frame IDs
        """
        clip_info = self.clips[idx]
        seq_path = clip_info['seq_path']
        frame_ids = clip_info['frame_ids']
        
        # Load frames
        video_frames = []
        poses = []
        tactile_maps = []
        hand_side = None
        
        for frame_id in frame_ids:
            # Load RGB
            rgb_path = seq_path / f"{frame_id}.cam-d_rgb.jpeg"
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.image_size, self.image_size))
            rgb = rgb.astype(np.float32) / 255.0
            video_frames.append(rgb)
            
            # Load annotation
            anno_path = seq_path / f"{frame_id}.anno.pkl"
            with open(anno_path, 'rb') as f:
                anno = pickle.load(f)
            
            # Extract hand pose: use original 21-joint 3D directly
            joint_position = anno.get('joint_position', None)
            hand_side_frame = anno.get('hand_side', 'left')
            if hand_side is None:
                hand_side = hand_side_frame
            
            if joint_position is None:
                # Should not happen after filtering, but be safe
                joint_position = np.zeros((21, 3), dtype=np.float32)
                logger.warning(f"Frame {frame_id} missing joint_position, using zeros")
            
            pose_bilateral = self._convert_pose_to_bilateral(joint_position, hand_side_frame)
            poses.append(pose_bilateral)
            
            # Load pressure from UV map (hand-space, not sensor-space)
            uv_pressure = anno.get('pressure_map', np.zeros((224, 224, 1), dtype=np.float32))
            if uv_pressure.ndim == 3:
                uv_pressure = uv_pressure[:, :, 0]  # (224, 224)
            
            # Convert to bilateral format (2, 16, 16)
            pressure_bilateral = self._convert_pressure_to_bilateral(uv_pressure, hand_side_frame)
            tactile_maps.append(pressure_bilateral)
        
        # Stack to tensors
        video = np.stack(video_frames, axis=0)  # (T, H, W, 3)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, 3, H, W)
        
        poses = np.stack(poses, axis=0)  # (T, 42, 3)
        tactile = np.stack(tactile_maps, axis=0)  # (T, 2, 16, 16)
        
        # Convert to torch tensors
        video = torch.from_numpy(video).float()
        poses = torch.from_numpy(poses).float()
        tactile = torch.from_numpy(tactile).float()
        
        return {
            'video': video,
            'poses': poses,
            'tactile': tactile,
            'hand_side': hand_side,
            'frame_ids': frame_ids,
            'sequence': clip_info['sequence'],
        }
    
    # Sentinel value for missing hand joints (model learns to ignore these)
    MISSING_JOINT_VALUE = -999.0
    
    def _convert_pose_to_bilateral(self, joint_position: np.ndarray, hand_side: str) -> np.ndarray:
        """
        Convert EgoPressure 21-joint pose to bilateral 42-joint format.
        
        Uses original 21-joint 3D coordinates directly (no padding to 24).
        Missing hand is filled with sentinel value MISSING_JOINT_VALUE.
        
        Args:
            joint_position: (21, 3) - Single hand joints from EgoPressure
            hand_side: 'left' or 'right'
        
        Returns:
            pose_42: (42, 3) - Bilateral format [left_21, right_21]
        """
        pose_42 = np.full((42, 3), self.MISSING_JOINT_VALUE, dtype=np.float32)
        
        if hand_side == 'left':
            pose_42[:21] = joint_position      # left hand: joints 0-20
            # right hand (21-41): stays MISSING_JOINT_VALUE
        else:  # 'right'
            pose_42[21:42] = joint_position    # right hand: joints 21-41
            # left hand (0-20): stays MISSING_JOINT_VALUE
        
        return pose_42
    
    def _convert_pressure_to_bilateral(self, uv_pressure_224: np.ndarray, hand_side: str) -> np.ndarray:
        """
        Convert EgoPressure UV pressure (224x224) to our 2x16x16 bilateral format.
        
        UV pressure is already in hand-space (mapped to MANO surface),
        so downsampling preserves hand-region correspondence.
        
        Note: pressure_map is per-frame normalized by EgoPressure
        (divided by per-frame max, stored in pressure_map_range).
        Values are roughly [0, 1] with slight overshoot up to ~1.3
        from interpolation. We clip negatives but keep >1 values.
        
        Args:
            uv_pressure_224: (224, 224) - UV space pressure (per-frame normalized)
            hand_side: 'left' or 'right'
        
        Returns:
            pressure_bilateral: (2, 16, 16) - Bilateral format [left, right] in UV space
        """
        # Clip negatives (interpolation artifacts), keep >1 values
        uv_clean = np.clip(uv_pressure_224, 0.0, None).astype(np.float32)
        
        # Downsample to 16x16 (preserves UV space structure)
        pressure_16x16 = cv2.resize(
            uv_clean,
            (self.pressure_size, self.pressure_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Create bilateral format
        pressure_bilateral = np.zeros((2, self.pressure_size, self.pressure_size), dtype=np.float32)
        
        if hand_side == 'left':
            pressure_bilateral[0] = pressure_16x16
        else:  # 'right'
            pressure_bilateral[1] = pressure_16x16
        
        return pressure_bilateral


def build_egopressure_dataloaders(
    data_root: str,
    train_sequences: List[str],
    val_sequences: List[str],
    batch_size: int = 4,
    clip_length: int = 16,
    num_workers: int = 4,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation dataloaders for EgoPressure dataset.
    
    Args:
        data_root: Root directory of EgoPressure dataset
        train_sequences: List of training sequence names
        val_sequences: List of validation sequence names
        batch_size: Batch size
        clip_length: Number of frames per clip
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for EgoPressureDataset
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = EgoPressureDataset(
        data_root=data_root,
        sequences=train_sequences,
        clip_length=clip_length,
        augmentation=True,
        **kwargs
    )
    
    val_dataset = EgoPressureDataset(
        data_root=data_root,
        sequences=val_sequences,
        clip_length=clip_length,
        augmentation=False,
        **kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    logger.info(f"Built EgoPressure dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} clips, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} clips, {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    data_root = "datasets/egopressure"
    sequences = ["extracted_sample"]  # Test with extracted sample
    
    dataset = EgoPressureDataset(
        data_root=data_root,
        sequences=sequences,
        clip_length=8,
        require_annotation=True,
        require_pressure=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Load one sample
        sample = dataset[0]
        
        print(f"\nSample 0:")
        print(f"  video: {sample['video'].shape}")
        print(f"  poses: {sample['poses'].shape}")
        print(f"  tactile: {sample['tactile'].shape}")
        print(f"  hand_side: {sample['hand_side']}")
        print(f"  frame_ids: {sample['frame_ids'][:3]}...")
        
        # Check data ranges
        print(f"\nData ranges:")
        print(f"  video: [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
        print(f"  poses: [{sample['poses'].min():.3f}, {sample['poses'].max():.3f}]")
        print(f"  tactile: [{sample['tactile'].min():.3f}, {sample['tactile'].max():.3f}]")
        
        # Check non-zero elements
        print(f"\nNon-zero elements:")
        print(f"  poses: {(sample['poses'] != 0).sum()} / {sample['poses'].numel()}")
        print(f"  tactile: {(sample['tactile'] != 0).sum()} / {sample['tactile'].numel()}")
