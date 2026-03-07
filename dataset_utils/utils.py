"""
Data loading and preprocessing utilities for Viper.

This module provides:
- EpisodicDataset: PyTorch Dataset for loading demonstration data
- Data loading functions with normalization statistics
- Environment utilities for object pose sampling
"""
import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    """Dataset class for loading episodic demonstration data."""
    
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, horizon_size, is_train=None):
        """
        Initialize the dataset.
        
        Args:
            episode_ids: List of episode indices to load
            dataset_dir: Directory containing HDF5 episode files
            camera_names: List of camera names to load
            norm_stats: Normalization statistics (mean/std for qpos and action)
            horizon_size: Prediction horizon (number of future timesteps)
            is_train: Whether this is training data
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.is_train = is_train
        self.horizon_size = horizon_size
        self.history_size = self.horizon_size - 1
        self.future_size = self.horizon_size
        self.__getitem__(0)
    
    def __len__(self):
        return len(self.episode_ids)
    
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Returns:
            image_data: Historical images (5 frames, downsampled)
            image_data_future: Future images (5 frames, downsampled)
            qpos_data: Historical joint positions
            qpos_data_future: Future joint positions
            action_data: Action sequence (horizon_size steps)
        """
        sample_full_episode = False     # hard code
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:

            max_start_ts = 450 - self.horizon_size              # 450 means the episode length, and horizon_size is 50, so max_start_ts is 400. 
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(np.arange(0, max_start_ts))
            
            # usually, horizon_size == 50, history_size == 49 (cause don't including current time step)
            if start_ts >= self.history_size:
                # get the current + history time step data & future data: qpos
                qpos = root['/observations/qpos'][start_ts - self.history_size:start_ts + 1]
                qpos_future = root['/observations/qpos'][start_ts + 1 : start_ts + 1 + self.horizon_size]

                # get the current + history time step data & future data: image
                image = [root['/observations/images/top'][start_ts - self.horizon_size + i * 10] for i in range(1, 6)]
                image_future = [root['/observations/images/top'][start_ts + i * 10] for i in range(1, 6)]
                
                action = root['/action'][start_ts : start_ts + self.horizon_size]
            else:
                # need to pad the data
                # get the current + history time step data & future data: qpos
                shape = ((self.history_size - start_ts),) + root['/observations/qpos'].shape[1:]
                zero_metrics = np.zeros(shape)
                qpos = np.concatenate([zero_metrics, root['/observations/qpos'][:start_ts + 1]], axis=0)
                qpos_future = root['/observations/qpos'][start_ts + 1 : start_ts + 1 + self.horizon_size]

                # get the current + history time step data & history data: image
                shape = ((self.history_size - start_ts),) + root['/observations/images/top'].shape[1:]
                zero_metrics = np.zeros(shape)
                image = np.concatenate([zero_metrics, root['/observations/images/top'][:start_ts + 1]], axis=0)
                image = [image[self.history_size - self.horizon_size + i * 10] for i in range(1, 6)]
                image_future = [root['/observations/images/top'][start_ts + i * 10] for i in range(1, 6)]

                # get the future data: actions
                action = root['/action'][start_ts : start_ts + self.horizon_size]

            image = np.stack(image, axis=0)
            image_future = np.stack(image_future, axis=0)

            image_data = torch.from_numpy(image).float()
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(action).float()
            image_data_future = torch.from_numpy(image_future).float()
            qpos_data_future = torch.from_numpy(qpos_future).float()

            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data_future = torch.einsum('k h w c -> k c h w', image_data_future)

            image_data = image_data / 255.0
            image_data_future = image_data_future / 255.0
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            qpos_data_future = (qpos_data_future - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, image_data_future, qpos_data, qpos_data_future, action_data


def get_norm_stats(dataset_dir, num_episodes):
    """
    Compute normalization statistics from the dataset.
    
    Args:
        dataset_dir: Directory containing HDF5 episode files
        num_episodes: Number of episodes to process
    
    Returns:
        dict: Normalization statistics with action_mean, action_std, qpos_mean, qpos_std
    """
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, horizon_size):
    """
    Load training and validation data.
    
    Args:
        dataset_dir: Directory containing HDF5 episode files
        num_episodes: Total number of episodes
        camera_names: List of camera names
        batch_size_train: Batch size for training
        batch_size_val: Batch size for validation
        horizon_size: Prediction horizon
    
    Returns:
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation
        norm_stats: Normalization statistics
        is_sim: Whether the data is from simulation
    """
    print(f'\nData from: {dataset_dir}\n')
    
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, horizon_size, is_train=True)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, horizon_size, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### Environment utilities ###

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def sample_stack_pose():
    # red_box
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    red_box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    red_box_quat = np.array([1, 0, 0, 0])
    red_box_pose = np.concatenate([red_box_position, red_box_quat])

    # blue_box
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    blue_box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    blue_box_quat = np.array([1, 0, 0, 0])
    blue_box_pose = np.concatenate([blue_box_position, blue_box_quat])

    return red_box_pose, blue_box_pose


def sample_storage_pose():
    # cube
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    cube_pose = np.concatenate([cube_position, cube_quat])

    # box
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    box_quat = np.array([1, 0, 0, 0])
    box_pose = np.concatenate([box_position, box_quat])

    return cube_pose, box_pose


def compute_dict_mean(epoch_dicts):
    """Compute mean of dictionary values across multiple epochs."""
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    """Detach all tensor values in a dictionary."""
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
