"""
EgoDex Dataset - Fixed Version
Based on official EgoDex implementation
"""
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from decord import VideoReader, cpu
from typing import Dict, List, Tuple, Optional

from src.touchanything.utils.pressure_map import generate_pseudo_pressure_maps_batch


# Projection / cropping helpers (module-level for pickling with multiprocess DataLoader)

def _project_to_2d(points_3d: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project camera-frame 3D points to 2D pixel coordinates.
    
    Args:
        points_3d: (N, 3) in camera frame
        intrinsic: (3, 3)
    Returns:
        pts_2d: (N, 2) pixel coords
        valid: (N,) bool mask (z > 0)
    """
    proj = intrinsic @ points_3d.T  # (3, N)
    z = proj[2, :]
    valid = z > 0.01
    pts_2d = np.zeros((points_3d.shape[0], 2))
    pts_2d[valid, 0] = proj[0, valid] / z[valid]
    pts_2d[valid, 1] = proj[1, valid] / z[valid]
    return pts_2d, valid


def _get_hand_bbox(
    pts_2d: np.ndarray, valid: np.ndarray,
    img_w: int, img_h: int, padding_ratio: float = 0.4,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute a padded square bounding box around valid 2D hand joints.
    Returns (x1, y1, x2, y2) or None.
    """
    if not np.any(valid):
        return None
    pts = pts_2d[valid]
    in_img = (pts[:, 0] >= 0) & (pts[:, 0] < img_w) & \
             (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
    if not np.any(in_img):
        return None
    pts = pts[in_img]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    size = max(x_max - x_min, y_max - y_min)
    pad = size * padding_ratio
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = (size + 2 * pad) / 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return (x1, y1, x2, y2)


class EgoDexDatasetV2(Dataset):
    """
    EgoDex Dataset for hand pose prediction (Fixed Version).
    
    Key improvements:
    1. Uses correct joint definitions from official code
    2. Converts to camera coordinate frame
    3. Supports flexible joint selection
    
    Args:
        data_root: Root directory of EgoDex data
        clip_length: Number of frames per clip
        frame_interval: Interval between frames
        transform: Image transformations
        split: 'train', 'val', or 'test'
        use_camera_frame: Whether to convert to camera coordinate frame (recommended)
        joint_set: 'hands_only' (48 joints) or 'hands_arms' (56 joints)
    """
    
    # Joint definitions from official EgoDex code
    LEFT_HAND_JOINTS = [
        # Thumb (4 joints)
        'leftThumbKnuckle', 'leftThumbIntermediateBase', 
        'leftThumbIntermediateTip', 'leftThumbTip',
        # Index finger (5 joints)
        'leftIndexFingerMetacarpal', 'leftIndexFingerKnuckle',
        'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 
        'leftIndexFingerTip',
        # Middle finger (5 joints)
        'leftMiddleFingerMetacarpal', 'leftMiddleFingerKnuckle',
        'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip',
        'leftMiddleFingerTip',
        # Ring finger (5 joints)
        'leftRingFingerMetacarpal', 'leftRingFingerKnuckle',
        'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip',
        'leftRingFingerTip',
        # Little finger (5 joints)
        'leftLittleFingerMetacarpal', 'leftLittleFingerKnuckle',
        'leftLittleFingerIntermediateBase', 'leftLittleFingerIntermediateTip',
        'leftLittleFingerTip',
    ]  # Total: 24 joints
    
    LEFT_ARM_JOINTS = [
        'leftShoulder', 'leftArm', 'leftForearm', 'leftHand'
    ]  # Total: 4 joints
    
    def __init__(
        self,
        data_root: str,
        clip_length: int = 16,
        frame_interval: int = 1,
        transform=None,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        use_camera_frame: bool = True,
        joint_set: str = 'hands_only',  # 'hands_only' or 'hands_arms'
        image_size: Tuple[int, int] = (224, 224),  # target size for decord decode
        task: str = 'pose_prediction',  # 'pose_prediction' or 'tactile_prediction'
        multi_view: bool = False,  # if True, duplicate ego frames as wrist views
    ):
        self.data_root = Path(data_root)
        self.clip_length = clip_length
        self.frame_interval = frame_interval
        self.task = task
        # Use transform's resize_size if available (accounts for random_crop upscale)
        if transform is not None and hasattr(transform, 'resize_size'):
            self.image_size = transform.resize_size
        else:
            self.image_size = image_size
        self.transform = transform
        self.split = split
        self.use_camera_frame = use_camera_frame
        self.joint_set = joint_set
        self.multi_view = multi_view
        
        # Define joints based on joint_set
        if joint_set == 'hands_only':
            left_joints = self.LEFT_HAND_JOINTS  # 24 joints
            self.num_joints_per_hand = 24
        elif joint_set == 'hands_arms':
            left_joints = self.LEFT_ARM_JOINTS + self.LEFT_HAND_JOINTS  # 28 joints
            self.num_joints_per_hand = 28
        else:
            raise ValueError(f"Unknown joint_set: {joint_set}")
        
        right_joints = [j.replace('left', 'right') for j in left_joints]
        self.all_joints = left_joints + right_joints
        self.num_joints = len(self.all_joints)
        
        print(f"[EgoDexV2] Using {joint_set}: {self.num_joints} joints total "
              f"({self.num_joints_per_hand} per hand)")
        
        # Collect all task folders
        self.task_folders = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        # Collect all video-hdf5 pairs and cache frame counts
        self.samples = []
        print(f"[EgoDexV2] Caching video metadata...")
        for task_folder in self.task_folders:
            mp4_files = sorted(task_folder.glob('*.mp4'))
            for mp4_file in mp4_files:
                hdf5_file = mp4_file.with_suffix('.hdf5')
                if hdf5_file.exists():
                    # Cache total frames to avoid reopening video in __getitem__
                    try:
                        vr = VideoReader(str(mp4_file), ctx=cpu(0))
                        total_frames = len(vr)
                    except Exception:
                        cap = cv2.VideoCapture(str(mp4_file))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                    
                    self.samples.append({
                        'video_path': str(mp4_file),
                        'hdf5_path': str(hdf5_file),
                        'task': task_folder.name,
                        'total_frames': total_frames,
                    })
        
        # Split dataset
        total_samples = len(self.samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        if split == 'train':
            self.samples = self.samples[:train_size]
        elif split == 'val':
            self.samples = self.samples[train_size:train_size + val_size]
        elif split == 'test':
            self.samples = self.samples[train_size + val_size:]
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples from {len(self.task_folders)} tasks")
    
    def __len__(self):
        return len(self.samples)
    
    def _compute_frame_indices(self, start_frame: int, num_frames: int, total_frames: int) -> List[int]:
        """Compute frame indices for a clip."""
        indices = []
        for i in range(num_frames):
            idx = start_frame + i * self.frame_interval
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        return indices

    def _load_video_frames(self, video_path: str, start_frame: int, num_frames: int,
                           target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load video frames using decord.
        
        Args:
            target_size: (h, w) to resize during decode. None = original resolution.
        """
        h, w = target_size if target_size else (None, None)
        try:
            if target_size:
                vr = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
            else:
                vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._compute_frame_indices(start_frame, num_frames, total_frames)
            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, 3) RGB
            return frames
        except Exception:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            next_needed = start_frame
            for i in range(num_frames):
                frame_idx = min(start_frame + i * self.frame_interval, total_frames - 1)
                while next_needed < frame_idx:
                    cap.grab()
                    next_needed += 1
                ret, frame = cap.read()
                next_needed += 1
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if target_size:
                        frame = cv2.resize(frame, (w, h))
                    frames.append(frame)
                else:
                    prev = frames[-1].copy() if frames else np.zeros(
                        (h or 1080, w or 1920, 3), dtype=np.uint8)
                    frames.append(prev)
            cap.release()
            return np.stack(frames, axis=0)
    
    def _convert_to_camera_frame(self, tfs: np.ndarray, cam_ext: np.ndarray) -> np.ndarray:
        """
        Convert transforms from world frame to camera frame.
        
        Args:
            tfs: (N, 4, 4) transforms in world frame
            cam_ext: (4, 4) camera extrinsics in world frame
        
        Returns:
            tfs_cam: (N, 4, 4) transforms in camera frame
        """
        # Camera-to-world transform is cam_ext
        # World-to-camera transform is inv(cam_ext)
        cam_to_world_inv = np.linalg.inv(cam_ext)
        
        # Transform each joint: T_cam = inv(cam_ext) @ T_world
        tfs_cam = cam_to_world_inv[None] @ tfs  # (1, 4, 4) @ (N, 4, 4) = (N, 4, 4)
        
        return tfs_cam
    
    def _load_poses(self, hdf5_path: str, start_frame: int, num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load hand poses from HDF5 file.
        Optimized: use slice-based reading (much faster than fancy indexing).
        
        Returns:
            poses: (T, J, 3) tensor of 3D positions
            confidences: (T, J) tensor of confidence scores
        """
        with h5py.File(hdf5_path, 'r') as f:
            total_frames = f['transforms/leftHand'].shape[0]
            
            # Compute slice range (read a contiguous block, then subsample)
            end_frame = min(start_frame + num_frames * self.frame_interval, total_frames)
            actual_start = min(start_frame, total_frames - 1)
            
            has_confidences = 'confidences' in f.keys()
            
            all_transforms = []  # will be (J, T, 4, 4)
            all_confidences = []  # will be (J, T)
            
            for joint in self.all_joints:
                # Read contiguous slice (fast!) then subsample by frame_interval
                joint_data = f[f'transforms/{joint}'][actual_start:end_frame]  # (L, 4, 4)
                
                # Subsample and pad to num_frames
                sampled = joint_data[::self.frame_interval]  # subsample
                if len(sampled) < num_frames:
                    # Pad by repeating last frame
                    pad = np.tile(sampled[-1:], (num_frames - len(sampled), 1, 1))
                    sampled = np.concatenate([sampled, pad], axis=0)
                sampled = sampled[:num_frames]
                all_transforms.append(sampled)
                
                # Confidence
                if has_confidences and joint in f['confidences']:
                    conf_data = f[f'confidences/{joint}'][actual_start:end_frame]
                    conf_sampled = conf_data[::self.frame_interval]
                    if len(conf_sampled) < num_frames:
                        conf_sampled = np.concatenate([
                            conf_sampled, np.ones(num_frames - len(conf_sampled))
                        ])
                    all_confidences.append(conf_sampled[:num_frames])
                else:
                    all_confidences.append(np.ones(num_frames))
            
            # Stack: (J, T, 4, 4) -> (T, J, 4, 4)
            all_transforms = np.stack(all_transforms, axis=0).transpose(1, 0, 2, 3)
            all_confidences = np.stack(all_confidences, axis=0).transpose(1, 0)
            
            # Convert to camera frame if requested
            if self.use_camera_frame:
                cam_data = f['/transforms/camera'][actual_start:end_frame]
                cam_sampled = cam_data[::self.frame_interval]
                if len(cam_sampled) < num_frames:
                    pad = np.tile(cam_sampled[-1:], (num_frames - len(cam_sampled), 1, 1))
                    cam_sampled = np.concatenate([cam_sampled, pad], axis=0)
                cam_sampled = cam_sampled[:num_frames]
                
                for t in range(num_frames):
                    all_transforms[t] = self._convert_to_camera_frame(
                        all_transforms[t], cam_sampled[t]
                    )
            
            # Extract 3D positions from SE(3) matrices
            poses = all_transforms[:, :, :3, 3]  # (T, J, 3)
        
        return torch.from_numpy(poses.copy()).float(), torch.from_numpy(all_confidences.copy()).float()
    
    def _crop_hand_views(
        self,
        frames_fullres: np.ndarray,
        poses_cam: np.ndarray,
        hdf5_path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop left/right hand regions from full-resolution frames.
        
        Args:
            frames_fullres: (T, H_orig, W_orig, 3) full-res RGB frames
            poses_cam: (T, J, 3) joint positions in camera frame
            hdf5_path: path to HDF5 for loading intrinsics
        
        Returns:
            left_crops:  (T, h, w, 3) cropped+resized left hand views
            right_crops: (T, h, w, 3) cropped+resized right hand views
        """
        T, H_orig, W_orig, _ = frames_fullres.shape
        h_target, w_target = self.image_size
        n_left = self.num_joints_per_hand
        
        # Load camera intrinsics (constant per video)
        with h5py.File(hdf5_path, 'r') as f:
            intrinsic = f['camera/intrinsic'][:]  # (3, 3)
        
        left_crops = np.zeros((T, h_target, w_target, 3), dtype=np.uint8)
        right_crops = np.zeros((T, h_target, w_target, 3), dtype=np.uint8)
        
        for t in range(T):
            frame = frames_fullres[t]
            
            # Split left / right hand joints
            left_3d = poses_cam[t, :n_left, :]    # (24, 3)
            right_3d = poses_cam[t, n_left:, :]   # (24, 3)
            
            # --- Left hand crop ---
            pts_2d, valid = _project_to_2d(left_3d, intrinsic)
            bbox = _get_hand_bbox(pts_2d, valid, W_orig, H_orig)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                left_crops[t] = cv2.resize(crop, (w_target, h_target))
            else:
                # Fallback: center crop
                left_crops[t] = cv2.resize(frame, (w_target, h_target))
            
            # --- Right hand crop ---
            pts_2d, valid = _project_to_2d(right_3d, intrinsic)
            bbox = _get_hand_bbox(pts_2d, valid, W_orig, H_orig)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                right_crops[t] = cv2.resize(crop, (w_target, h_target))
            else:
                right_crops[t] = cv2.resize(frame, (w_target, h_target))
        
        return left_crops, right_crops

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Use cached total_frames
        total_frames = sample['total_frames']
        max_start = max(0, total_frames - self.clip_length * self.frame_interval)
        start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        # Load poses first (needed for multi-view cropping)
        poses, confidences = self._load_poses(
            sample['hdf5_path'],
            start_frame,
            self.clip_length
        )  # (T, J, 3), (T, J)
        
        if self.multi_view:
            # Load full-res frames for hand cropping
            frames_fullres = self._load_video_frames(
                sample['video_path'], start_frame, self.clip_length,
                target_size=None,
            )  # (T, H_orig, W_orig, 3)
            
            # Crop hand regions using camera-frame pose projections
            left_crops, right_crops = self._crop_hand_views(
                frames_fullres, poses.numpy(), sample['hdf5_path'],
            )  # (T, h, w, 3) each
            
            # Resize ego to target size
            h_t, w_t = self.image_size
            ego_resized = np.stack([
                cv2.resize(frames_fullres[t], (w_t, h_t))
                for t in range(self.clip_length)
            ])  # (T, h, w, 3)
            
            # Apply transforms to each view
            if self.transform:
                ego_frames = self.transform(ego_resized)
                left_frames = self.transform(left_crops)
                right_frames = self.transform(right_crops)
            else:
                ego_frames = torch.from_numpy(ego_resized).permute(0, 3, 1, 2).float() / 255.0
                left_frames = torch.from_numpy(left_crops).permute(0, 3, 1, 2).float() / 255.0
                right_frames = torch.from_numpy(right_crops).permute(0, 3, 1, 2).float() / 255.0
            
            frames = ego_frames  # backward compat: 'frames' key = ego view
            
            result = {
                'frames': frames,
                'poses': poses,
                'confidences': confidences,
                'task': sample['task'],
                'views': {
                    'ego': ego_frames,
                    'wrist_left': left_frames,
                    'wrist_right': right_frames,
                },
            }
        else:
            # Single-view: load at target resolution directly (fast)
            frames = self._load_video_frames(
                sample['video_path'], start_frame, self.clip_length,
                target_size=self.image_size,
            )
            if self.transform:
                frames = self.transform(frames)
            else:
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            
            result = {
                'frames': frames,
                'poses': poses,
                'confidences': confidences,
                'task': sample['task'],
            }
        
        # Generate pseudo pressure maps for tactile prediction task
        if self.task == 'tactile_prediction':
            pressure_maps = generate_pseudo_pressure_maps_batch(
                poses.numpy()
            )  # (T, 2, 16, 16)
            result['pressure_maps'] = torch.from_numpy(pressure_maps).float()
        
        return result


def collate_fn(batch):
    """Collate function for DataLoader."""
    frames = torch.stack([item['frames'] for item in batch])
    poses = torch.stack([item['poses'] for item in batch])
    confidences = torch.stack([item['confidences'] for item in batch])
    tasks = [item['task'] for item in batch]
    
    result = {
        'frames': frames,
        'poses': poses,
        'confidences': confidences,
        'tasks': tasks,
    }
    
    # Include multi-view data if available
    if 'views' in batch[0]:
        view_names = batch[0]['views'].keys()
        result['views'] = {
            name: torch.stack([item['views'][name] for item in batch])
            for name in view_names
        }
    
    # Include pressure maps and sensor mask if available (tactile prediction task)
    if 'pressure_maps' in batch[0]:
        result['pressure_maps'] = torch.stack([item['pressure_maps'] for item in batch])
    if 'sensor_mask' in batch[0]:
        result['sensor_mask'] = torch.stack([item['sensor_mask'] for item in batch])

    return result
