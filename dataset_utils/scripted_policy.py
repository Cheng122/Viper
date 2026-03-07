"""
Scripted policies for generating demonstration data.

This module provides waypoint-based policies for each task:
- PickAndTransferPolicy: Pick up cube with right arm, transfer to left arm
- InsertionPolicy: Insert peg into socket
- InsertionRevPolicy: Alternative insertion policy
- PickAndStackPolicy: Stack cubes
- StoragePolicy: Place cube into storage box
"""
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    """Base class for scripted policies using waypoint interpolation."""
    
    def __init__(self, inject_noise=False):
        """
        Initialize the policy.
        
        Args:
            inject_noise: Whether to add noise to actions (for data augmentation)
        """
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        """
        Generate trajectories for both arms based on initial observation.
        
        Args:
            ts_first: Initial timestep observation
        
        Returns:
            None (sets self.left_trajectory and self.right_trajectory)
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        """
        Interpolate between two waypoints.
        
        Args:
            curr_waypoint: Current waypoint dict with 't', 'xyz', 'quat', 'gripper'
            next_waypoint: Next waypoint dict
            t: Current timestep
        
        Returns:
            Tuple of (xyz, quat, gripper) interpolated values
        """
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        
        return xyz, quat, gripper

    def __call__(self, ts):
        """
        Get action for current timestep.
        
        Args:
            ts: Current timestep observation
        
        Returns:
            Action array [left_xyz, left_quat, left_gripper, right_xyz, right_quat, right_gripper]
        """
        if self.step_count == 0:
            self.generate_trajectory(ts)

        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):
    """Policy for transfer cube task: pick with right arm, transfer to left arm."""
    
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):
    """Policy for insertion task: insert peg into socket."""
    
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
        ]


class InsertionRevPolicy(BasePolicy):
    """Alternative policy for insertion task with different timing."""
    
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.012

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 70, "xyz": socket_xyz + np.array([0, 0, 0.10]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 140, "xyz": socket_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 180, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 210, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 230, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 280, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 320, "xyz": meet_xyz + np.array([-0.08, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 360, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 100, "xyz": peg_xyz + np.array([0.01, 0, 0.10]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7},
            {"t": 140, "xyz": peg_xyz + np.array([0.01, 0, 0.02]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7},
            {"t": 180, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7},
            {"t": 210, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7},
            {"t": 220, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 230, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 280, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 320, "xyz": meet_xyz + np.array([0.08, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 360, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
        ]


class PickAndStackPolicy(BasePolicy):
    """Policy for stack cube task: pick two cubes and stack them."""
    
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        object_info = np.array(ts_first.observation['env_state'])[:7]
        object_xyz_1 = object_info[:3]
        object_quat_1 = object_info[3:]
        
        object_info_2 = np.array(ts_first.observation['env_state'])[7:]
        object_xyz_2 = object_info_2[:3]
        object_quat_2 = object_info_2[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        central_position = np.array([0, 0.5, 0.05])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 90, "xyz": object_xyz_2 + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 150, "xyz": object_xyz_2 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 160, "xyz": object_xyz_2 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 210, "xyz": central_position + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 260, "xyz": central_position, "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 270, "xyz": central_position, "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 300, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 400, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 90, "xyz": object_xyz_1 + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 130, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 170, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 200, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 270, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 320, "xyz": central_position + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 360, "xyz": central_position + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 370, "xyz": central_position + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 400, "xyz": central_position + np.array([0.15, 0, 0.15]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
        ]


class StoragePolicy(BasePolicy):
    """Policy for storage cube task: place cube into storage box."""
    
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        object_info = np.array(ts_first.observation['env_state'])[:7]
        object_xyz_1 = object_info[:3]
        object_quat_1 = object_info[3:]
        
        object_info_2 = np.array(ts_first.observation['env_state'])[7:]
        object_xyz_2 = object_info_2[:3]
        object_quat_2 = object_info_2[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        central_position = np.array([0, 0.5, 0.1])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 90, "xyz": object_xyz_2 + np.array([0, -0.06, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 0.7},
            {"t": 150, "xyz": object_xyz_2 + np.array([0, -0.06, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.7},
            {"t": 170, "xyz": object_xyz_2 + np.array([0, -0.06, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 230, "xyz": central_position + np.array([0, -0.06, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 260, "xyz": central_position + np.array([0, -0.06, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 270, "xyz": central_position + np.array([0, -0.06, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 300, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            {"t": 400, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 90, "xyz": object_xyz_1 + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 130, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 170, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 200, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 270, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 320, "xyz": central_position + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 360, "xyz": central_position + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 370, "xyz": central_position + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            {"t": 400, "xyz": central_position + np.array([0.15, 0, 0.15]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
        ]


def test_policy(task_name):
    """Test a scripted policy by rolling it out in the environment."""
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)
