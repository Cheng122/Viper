"""
Main training and evaluation script for Viper imitation learning.

This script provides:
- Training loop with validation
- Evaluation in simulation environment
- Checkpoint saving and loading
- Logging utilities

Usage:
    Training:
        python imitate_episodes.py --task_name sim_transfer_cube_scripted
    
    Evaluation:
        python imitate_episodes.py --task_name sim_transfer_cube_scripted --eval
"""
import torch 
import numpy as np
import os
import sys
import time
import pickle
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

utils_dir = Path(__file__).resolve().parent / "dataset_utils"
sys.path.insert(0, str(utils_dir))
utils_dir = Path(__file__).resolve().parent / "models"
sys.path.insert(0, str(utils_dir))
utils_dir = Path(__file__).resolve().parent / "other_utils"
sys.path.insert(0, str(utils_dir))

from constants import DT
from copy import deepcopy
from collections import deque
from einops import rearrange
from visualize_episodes import save_videos
from constants import PUPPET_GRIPPER_JOINT_OPEN
from policy import ViperPolicy

from utils import load_data
from utils import compute_dict_mean, set_seed, detach_dict
from utils import sample_box_pose, sample_insertion_pose, sample_storage_pose, sample_stack_pose

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    # command line parameters
    set_seed(args['seed'])
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    torch.cuda.set_device(args['gpu'])

    # get task parameters
    from constants import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14                                                      # = args['action_dim']

    lr_backbone = 1e-5
    backbone = 'resnet18'

    dec_layers = 7
    nheads = 8
    policy_config = {
        'lr': args['lr'],
        'num_queries': args['horizon_size'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': camera_names,
        'feature_loss': args['feature_loss_weight'] > 0,
        'feature_loss_weight': args['feature_loss_weight'],
    }

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # Subsequent data reads will maintain the same chronological order: 0 -> earliest occurrence; -1 -> latest occurrence.
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, 
        num_episodes, 
        camera_names, 
        batch_size_train, 
        batch_size_val,
        policy_config['num_queries']
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    """
    Evaluate the trained policy in simulation.
    
    Args:
        config: Configuration dictionary
        ckpt_name: Name of checkpoint to load
        save_episode: Whether to save episode video
    
    Returns:
        Tuple of (success_rate, avg_return)
    """
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ViperPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    from sim_env import make_sim_env
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    query_frequency = 50                    # will be changed in the for loop below

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50                       # number of episodes to evaluate
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        elif 'sim_stack_cube' in task_name:
            BOX_POSE[0] = np.concatenate(sample_stack_pose()) # used in sim reset
        elif 'sim_storage_cube' in task_name:
            BOX_POSE[0] = np.concatenate(sample_storage_pose()) # used in sim reset

        ts = env.reset()

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        image_queue = deque(maxlen=50)
        qpos_queue = deque(maxlen=50)
        
        with torch.inference_mode():
            # start_time = time.time()
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                curr_image = get_image(ts, camera_names)
                qpos_numpy = np.array(ts.observation['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().unsqueeze(0)  # shape = [1, 14]
                if t == 0:
                    image_queue.append(curr_image.cpu().numpy())
                    qpos_queue.append(qpos)

                    for i in range(49):
                        zero_image = np.zeros((1, 1, 3, 480, 640), dtype=np.uint8)
                        image_queue.appendleft(zero_image)
                        zero_qpos = np.zeros((1, 14), dtype=np.uint8)
                        qpos_queue.appendleft(zero_qpos)
                else:
                    image_queue.append(curr_image)
                    qpos_queue.append(qpos)

                indices = [9, 19, 29, 39, 49]
                img = [image_queue[i] for i in indices]
                image_array = np.stack([img.cpu().numpy() if isinstance(img, torch.Tensor) else img for img in img], axis=1)
                qpos_array = np.stack([qpos.cpu().numpy() if isinstance(qpos, torch.Tensor) else qpos for qpos in qpos_queue], axis=1)
                image_tensor = torch.tensor(image_array).cuda()
                qpos_tensor = torch.tensor(qpos_array).cuda()
                image_tensor = rearrange(image_tensor, 'a b c d e f -> a b (c d) e f')

                image_future_tensor = torch.zeros_like(image_tensor)
                qpos_future_tensor = torch.zeros_like(qpos_tensor)
                
                if t % query_frequency == 0:
                    start_time = time.time()
                    all_actions = policy(qpos_tensor, qpos_future_tensor, image_tensor, image_future_tensor)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"timestep_{t}: 50 horizon tokens take {elapsed_time:.4f} seconds")
                if t == 50:
                    query_frequency = 1

                raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, image_data_future, qpos_data, qpos_data_future, action_data = data
    image_data, image_data_future, qpos_data, qpos_data_future, action_data = image_data.cuda(), image_data_future.cuda(), qpos_data.cuda(), qpos_data_future.cuda(), action_data.cuda()
    return policy(qpos_data, qpos_data_future, image_data, image_data_future, action_data) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    """
    Train the behavior cloning policy.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Configuration dictionary
    
    Returns:
        Tuple of (best_epoch, min_val_loss, best_state_dict)
    """
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = ViperPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    log_file = os.path.join(ckpt_dir, f'training_log_seed_{seed}.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,l1,feature_loss,l1_qpos,Validation Loss\n")
    
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):                   # data： image_data, image_data_future, qpos_data, qpos_data_future, action_data, is_pad, is_pad_status
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{summary_string}\n")

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        with open(log_file, 'a') as f:
        # 将 epoch 和损失细节写入日志
            f.write(f"{epoch},{summary_string}\n")

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    return best_ckpt_info


if __name__ == '__main__':
    task = 'sim_storage_cube_scripted'
    policy = 'Viper'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--onscreen_render', action='store_true', default=False)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default=f'./ckpt/{task}/{policy}')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class', default=f'{policy}')
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default=f'{task}')
    parser.add_argument('--horizon_size', action='store', type=int, help='horizon_size', default=50)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size',default=16)
    parser.add_argument('--seed', action='store', type=int, help='seed',default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs',default=2500)
    parser.add_argument('--lr', action='store', type=float, help='lr',default=1e-5)
    parser.add_argument('--action_dim', action='store', type=int, default=14)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200)
    parser.add_argument('--temporal_agg', action='store_true', default=False)
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=1, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=1, required=False)
    parser.add_argument('--self_attention', action="store", type=int, default=1)
    parser.add_argument('--feature_loss_weight', action='store', type=float, default=0.05)
    main(vars(parser.parse_args()))
