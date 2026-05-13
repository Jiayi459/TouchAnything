"""
TouchAnything HDF5 Dataset for PyTorch

Example usage:
    from src.data.hdf5_dataset import TouchAnythingHDF5Dataset
    
    dataset = TouchAnythingHDF5Dataset(
        hdf5_files=['path/to/traj1.hdf5', 'path/to/traj2.hdf5'],
        modalities=['images', 'pressure', 'hands'],
        cameras=['chest', 'left', 'right'],
        transform=None
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4
    )
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Callable


class TouchAnythingHDF5Dataset(Dataset):
    """
    HDF5 loader for the TouchAnything dataset.

    Supports multiple trajectories, multimodal loading, and flexible data selection/transforms.
    """
    
    def __init__(
        self,
        hdf5_files: List[str],
        modalities: List[str] = ['images', 'pressure', 'hands', 'poses'],
        cameras: List[str] = ['chest', 'left', 'right'],
        image_types: List[str] = ['color', 'depth'],
        transform: Optional[Callable] = None,
        baseline_correction: bool = True,
        cache_in_memory: bool = False
    ):
        """
        Args:
            hdf5_files: List of HDF5 file paths
            modalities: Modalities to load ['images', 'pressure', 'hands', 'poses']
            cameras: Cameras to load ['chest', 'left', 'right']
            image_types: Image types ['color', 'depth']
            transform: Data transform function
            baseline_correction: Whether to apply baseline correction to pressure data
            cache_in_memory: Whether to cache all data in memory (useful for small datasets)
        """
        self.hdf5_files = [Path(f) for f in hdf5_files]
        self.modalities = modalities
        self.cameras = cameras
        self.image_types = image_types
        self.transform = transform
        self.baseline_correction = baseline_correction
        self.cache_in_memory = cache_in_memory
        
        # Validate file existence.
        for f in self.hdf5_files:
            if not f.exists():
                raise FileNotFoundError(f"HDF5 file does not exist: {f}")
        
        # Build index entries as (file_idx, frame_idx).
        self.index = []
        self.file_handles = []
        self.metadata = []
        
        for file_idx, hdf5_path in enumerate(self.hdf5_files):
            with h5py.File(hdf5_path, 'r') as f:
                num_frames = f['metadata'].attrs['num_frames']
                meta = {
                    'trajectory_id': f['metadata'].attrs['trajectory_id'],
                    'task_name': f['metadata'].attrs['task_name'],
                    'num_frames': num_frames,
                    'fps': f['metadata'].attrs['fps']
                }
                self.metadata.append(meta)
                
                for frame_idx in range(num_frames):
                    self.index.append((file_idx, frame_idx))
        
        # Preload all data if memory caching is enabled.
        self.cache = None
        if self.cache_in_memory:
            print(f"Preloading {len(self.hdf5_files)} HDF5 files into memory...")
            self._load_to_memory()
    
    def _load_to_memory(self):
        """Load all data into memory."""
        self.cache = []
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                data = {}
                
                # Load images.
                if 'images' in self.modalities:
                    data['images'] = {}
                    for cam in self.cameras:
                        for img_type in self.image_types:
                            key = f'{cam}_{img_type}'
                            data['images'][key] = f[f'images/{key}'][...]
                
                # Load pressure data.
                if 'pressure' in self.modalities:
                    data['pressure'] = {
                        'left_sensor_raw': f['pressure/left_sensor_raw'][...],
                        'right_sensor_raw': f['pressure/right_sensor_raw'][...],
                        'left_quaternion': f['pressure/left_quaternion'][...],
                        'right_quaternion': f['pressure/right_quaternion'][...],
                    }
                    if self.baseline_correction:
                        data['pressure']['baseline_left'] = f['pressure/baseline_left'][...]
                        data['pressure']['baseline_right'] = f['pressure/baseline_right'][...]
                
                # Load hand data.
                if 'hands' in self.modalities:
                    data['hands'] = {
                        'left_joint_xyz': f['hands/left_joint_xyz'][...],
                        'right_joint_xyz': f['hands/right_joint_xyz'][...],
                    }
                    # joint_orientation is optional (removed in newer HDF5 files to save space)
                    if 'hands/left_joint_orientation' in f:
                        data['hands']['left_joint_orientation'] = f['hands/left_joint_orientation'][...]
                        data['hands']['right_joint_orientation'] = f['hands/right_joint_orientation'][...]
                
                # Load pose data.
                if 'poses' in self.modalities:
                    data['poses'] = {
                        'chest_pose': f['poses/chest_pose'][...],
                        'left_pose': f['poses/left_pose'][...],
                        'right_pose': f['poses/right_pose'][...],
                    }
                
                self.cache.append(data)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """
        Return a single-frame sample.
        
        Returns:
            dict: {
                'images': {
                    'chest_color': [H, W, 3] uint8,
                    'chest_depth': [H, W] uint16,
                    ...
                },
                'pressure': {
                    'left_sensor_raw': [256] uint8,
                    'right_sensor_raw': [256] uint8,
                    ...
                },
                'hands': {
                    'left_joint_xyz': [21, 3] float32,
                    'right_joint_xyz': [21, 3] float32,
                    ...
                },
                'poses': {
                    'chest_pose': [7] float32,
                    ...
                },
                'metadata': {
                    'trajectory_id': str,
                    'task_name': str,
                    'frame_idx': int
                }
            }
        """
        file_idx, frame_idx = self.index[idx]
        
        sample = {
            'metadata': {
                **self.metadata[file_idx],
                'frame_idx': frame_idx
            }
        }
        
        # Read from cache or directly from file.
        if self.cache_in_memory:
            data = self._get_from_cache(file_idx, frame_idx)
        else:
            data = self._get_from_file(file_idx, frame_idx)
        
        sample.update(data)
        
        # Apply transform.
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _get_from_cache(self, file_idx, frame_idx):
        """Read from the in-memory cache."""
        cached_data = self.cache[file_idx]
        sample = {}
        
        if 'images' in self.modalities:
            sample['images'] = {
                key: cached_data['images'][key][frame_idx]
                for key in cached_data['images']
            }
        
        if 'pressure' in self.modalities:
            sample['pressure'] = {
                'left_sensor_raw': cached_data['pressure']['left_sensor_raw'][frame_idx],
                'right_sensor_raw': cached_data['pressure']['right_sensor_raw'][frame_idx],
                'left_quaternion': cached_data['pressure']['left_quaternion'][frame_idx],
                'right_quaternion': cached_data['pressure']['right_quaternion'][frame_idx],
            }
            # Baseline correction.
            if self.baseline_correction:
                baseline_l = cached_data['pressure']['baseline_left']
                baseline_r = cached_data['pressure']['baseline_right']
                sample['pressure']['left_sensor_corrected'] = np.clip(
                    sample['pressure']['left_sensor_raw'].astype(np.float32) - baseline_l, 0, 255
                ).astype(np.uint8)
                sample['pressure']['right_sensor_corrected'] = np.clip(
                    sample['pressure']['right_sensor_raw'].astype(np.float32) - baseline_r, 0, 255
                ).astype(np.uint8)
        
        if 'hands' in self.modalities:
            sample['hands'] = {
                key: cached_data['hands'][key][frame_idx]
                for key in cached_data['hands']
            }
        
        if 'poses' in self.modalities:
            sample['poses'] = {
                key: cached_data['poses'][key][frame_idx]
                for key in cached_data['poses']
            }
        
        return sample
    
    def _get_from_file(self, file_idx, frame_idx):
        """Read from the HDF5 file on demand."""
        hdf5_path = self.hdf5_files[file_idx]
        sample = {}
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'images' in self.modalities:
                sample['images'] = {}
                for cam in self.cameras:
                    for img_type in self.image_types:
                        key = f'{cam}_{img_type}'
                        sample['images'][key] = f[f'images/{key}'][frame_idx]
            
            if 'pressure' in self.modalities:
                sample['pressure'] = {
                    'left_sensor_raw': f['pressure/left_sensor_raw'][frame_idx],
                    'right_sensor_raw': f['pressure/right_sensor_raw'][frame_idx],
                    'left_quaternion': f['pressure/left_quaternion'][frame_idx],
                    'right_quaternion': f['pressure/right_quaternion'][frame_idx],
                }
                # Baseline correction.
                if self.baseline_correction:
                    baseline_l = f['pressure/baseline_left'][...]
                    baseline_r = f['pressure/baseline_right'][...]
                    sample['pressure']['left_sensor_corrected'] = np.clip(
                        sample['pressure']['left_sensor_raw'].astype(np.float32) - baseline_l, 0, 255
                    ).astype(np.uint8)
                    sample['pressure']['right_sensor_corrected'] = np.clip(
                        sample['pressure']['right_sensor_raw'].astype(np.float32) - baseline_r, 0, 255
                    ).astype(np.uint8)
            
            if 'hands' in self.modalities:
                sample['hands'] = {
                    'left_joint_xyz': f['hands/left_joint_xyz'][frame_idx],
                    'right_joint_xyz': f['hands/right_joint_xyz'][frame_idx],
                }
                # joint_orientation is optional (removed in newer HDF5 files to save space)
                if 'hands/left_joint_orientation' in f:
                    sample['hands']['left_joint_orientation'] = f['hands/left_joint_orientation'][frame_idx]
                    sample['hands']['right_joint_orientation'] = f['hands/right_joint_orientation'][frame_idx]
            
            if 'poses' in self.modalities:
                sample['poses'] = {
                    'chest_pose': f['poses/chest_pose'][frame_idx],
                    'left_pose': f['poses/left_pose'][frame_idx],
                    'right_pose': f['poses/right_pose'][frame_idx],
                }
        
        return sample
    
    def get_trajectory_metadata(self, file_idx):
        """Get metadata for the specified trajectory."""
        return self.metadata[file_idx]
    
    def get_all_metadata(self):
        """Get metadata for all trajectories."""
        return self.metadata


# ============================================================================
# Example data transforms
# ============================================================================

class ToTensor:
    """Convert NumPy arrays to PyTorch tensors."""
    
    def __call__(self, sample):
        result = {'metadata': sample['metadata']}
        
        if 'images' in sample:
            result['images'] = {}
            for key, img in sample['images'].items():
                # [H, W, C] → [C, H, W] for color images
                if 'color' in key:
                    img = np.transpose(img, (2, 0, 1))
                result['images'][key] = torch.from_numpy(img)
        
        if 'pressure' in sample:
            result['pressure'] = {
                key: torch.from_numpy(val) if isinstance(val, np.ndarray) else val
                for key, val in sample['pressure'].items()
            }
        
        if 'hands' in sample:
            result['hands'] = {
                key: torch.from_numpy(val)
                for key, val in sample['hands'].items()
            }
        
        if 'poses' in sample:
            result['poses'] = {
                key: torch.from_numpy(val)
                for key, val in sample['poses'].items()
            }
        
        return result


class NormalizeImages:
    """Normalize images to the [0, 1] range."""
    
    def __call__(self, sample):
        if 'images' in sample:
            for key in sample['images']:
                if 'color' in key:
                    sample['images'][key] = sample['images'][key].float() / 255.0
                elif 'depth' in key:
                    # Depth normalization (assume max depth is 10 m = 10000 mm)
                    sample['images'][key] = sample['images'][key].float() / 10000.0
        
        return sample


# ============================================================================
# Usage example
# ============================================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Find all HDF5 files.
    hdf5_dir = Path('datasets/TouchAnything_hdf5_clean')
    hdf5_files = list(hdf5_dir.rglob('*.hdf5'))
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Create the dataset.
    transform = transforms.Compose([
        ToTensor(),
        NormalizeImages()
    ])
    
    dataset = TouchAnythingHDF5Dataset(
        hdf5_files=hdf5_files[:5],  # Use the first 5 files for testing
        modalities=['images', 'pressure', 'hands'],
        cameras=['chest', 'right'],  # Load a subset of cameras
        image_types=['color'],  # Load color images only
        transform=transform,
        baseline_correction=True,
        cache_in_memory=False
    )
    
    print(f"Dataset size: {len(dataset)} frames")
    
    # Create the DataLoader.
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Test loading.
    for batch in dataloader:
        print("\nBatch data:")
        print(f"  chest_color: {batch['images']['chest_color'].shape}")
        print(f"  right_sensor_raw: {batch['pressure']['right_sensor_raw'].shape}")
        print(f"  right_joint_xyz: {batch['hands']['right_joint_xyz'].shape}")
        print(f"  Task: {batch['metadata']['task_name']}")
        break
