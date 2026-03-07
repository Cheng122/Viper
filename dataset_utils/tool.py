"""
Utility tools for dataset preprocessing.

This module provides functions for padding episodes to fixed length.
"""
import os
import h5py
import numpy as np


def pad_episodes_to_fixed_length(dataset_dir, output_dir, target_length=500):
    """
    Pad all episodes to a fixed length by repeating the last frame.
    
    Args:
        dataset_dir: Input dataset directory path
        output_dir: Output dataset directory path
        target_length: Target episode length
    """
    os.makedirs(output_dir, exist_ok=True)
    
    episode_files = [f for f in os.listdir(dataset_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
    num_episodes = len(episode_files)
    
    for episode_idx in range(num_episodes):
        input_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        output_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
        
        with h5py.File(input_path, 'r') as src, h5py.File(output_path, 'w') as dst:
            qpos = src['/observations/qpos'][()]
            action = src['/action'][()]
            image_RGB = src['observations/images/top'][()]
            
            # 计算需要填充的长度
            current_length = qpos.shape[0]
            padding_length = target_length - current_length
            
            if padding_length > 0:
                last_qpos = np.tile(qpos[-1:], (padding_length, 1))
                last_action = np.tile(action[-1:], (padding_length, 1))
                last_image = np.tile(image_RGB[-1:], (padding_length, 1, 1, 1))
                
                padded_qpos = np.concatenate([qpos, last_qpos], axis=0)
                padded_action = np.concatenate([action, last_action], axis=0)
                padded_image = np.concatenate([image_RGB, last_image], axis=0)
            else:
                padded_qpos = qpos[:target_length]
                padded_action = action[:target_length]
                padded_image = image_RGB[:target_length]
            
            dst.create_dataset('/observations/qpos', data=padded_qpos)
            dst.create_dataset('/action', data=padded_action)
            dst.create_dataset('observations/images/top', data=padded_image)
            
            dst.attrs['original_length'] = current_length
            dst.attrs['padded_length'] = target_length

    print(f"Successfully processed {num_episodes} episodes, saved to {output_dir}")


# example
if __name__ == "__main__":
    dataset_dir = "./data/sim_insertion_scripted"
    output_dir = "./data_450/sim_insertion_scripted"
    pad_episodes_to_fixed_length(dataset_dir, output_dir, target_length=450)
