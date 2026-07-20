from .egodex_dataset_v2 import EgoDexDatasetV2
from .touchanything_dataset import TouchAnythingDataset
from .transforms import get_transforms

# Alias for backward compatibility
EgoDexDataset = EgoDexDatasetV2


def build_dataset(config: dict, split: str, transform=None):
    """
    Build the dataset instance according to config['data']['dataset'].

    Supported values:
        'egodex_v2'       → EgoDexDatasetV2
        'touchanything'   → TouchAnythingDataset
    """
    dataset_type = config['data'].get('dataset', 'egodex_v2')
    image_size = tuple(config['data']['image_size'])
    task = config['model']['task']
    multi_view = config['model'].get('multi_view', {}).get('enabled', False)

    if dataset_type == 'egodex_v2':
        return EgoDexDatasetV2(
            data_root=config['data']['data_root'],
            clip_length=config['data']['clip_length'],
            frame_interval=config['data']['frame_interval'],
            transform=transform,
            split=split,
            train_ratio=config['data']['train_split'],
            val_ratio=config['data']['val_split'],
            use_camera_frame=config['data'].get('use_camera_frame', True),
            joint_set=config['data'].get('joint_set', 'hands_only'),
            image_size=image_size,
            task=task,
            multi_view=multi_view,
        )
    elif dataset_type == 'touchanything':
        return TouchAnythingDataset(
            data_root=config['data']['data_root'],
            mapping_left_path=config['data']['mapping_left'],
            mapping_right_path=config['data']['mapping_right'],
            clip_length=config['data']['clip_length'],
            frame_interval=config['data']['frame_interval'],
            transform=transform,
            split=split,
            train_ratio=config['data']['train_split'],
            val_ratio=config['data']['val_split'],
            image_size=image_size,
            task=task,
            multi_view=multi_view,
            tactile_size=config['data'].get('tactile_size', 16),
            split_file=config['data'].get('split_file', None),
            pose_source=config['data'].get('pose_source', 'rokoko'),
            fallback_pose_source=config['data'].get('fallback_pose_source', None),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


__all__ = [
    'EgoDexDataset', 'EgoDexDatasetV2', 'TouchAnythingDataset',
    'get_transforms', 'build_dataset',
]
