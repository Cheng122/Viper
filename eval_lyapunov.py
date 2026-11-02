#!/usr/bin/env python3
"""
Roll out ImiNMPC in sim aligned with a recorded episode; compute manuscript-aligned J_N
(Eq. (9)–(11) in main.md): proprio + sparse visual backbone features + control.

Requires /meta/env_state in the HDF5.

Optional: ``--save_video`` writes MuJoCo RGB rollout mp4 (``.npz`` logs contain no frames).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dataset_utils"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "other_utils"))

from constants import DT  # noqa: E402
from policy import ViperPolicy  # noqa: E402
from sim_env import BOX_POSE, make_sim_env  # noqa: E402
from utils import set_seed  # noqa: E402

from lyapunov import (  # noqa: E402
    PaperLyapunovConfig,
    compute_JN_expert_baseline,
    compute_JN_paper,
    extract_reference_slices,
    slice_reference_future_image_indices,
)
from sim_perturb import apply_disturbance  # noqa: E402


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def load_reference_episode(dataset_dir: str, episode_id: int, camera_names: list):
    path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
    cam = camera_names[0]
    with h5py.File(path, "r") as root:
        qpos_ref = np.array(root["/observations/qpos"])
        action_ref = np.array(root["/action"])
        images_ref = np.array(root[f"/observations/images/{cam}"])
        env_state = None
        if "meta" in root and "env_state" in root["meta"]:
            env_state = np.array(root["meta/env_state"])
    return qpos_ref, action_ref, images_ref, env_state, path


def backbone_future_features_flat(model, imgs_5_tc_hw: torch.Tensor) -> torch.Tensor:
    """
    Match training `image_feature_future` layout in models/main.py (else branch stacks 5 frames).
    imgs_5_tc_hw: (5, 3, H, W) normalized float cuda.
    Returns (L, C) with L = 5 * H_map * W_map.
    """
    feats = []
    for i in range(5):
        x = imgs_5_tc_hw[i : i + 1]
        out_list, _pos = model.backbones[0](x)
        feats.append(out_list[0])
    feat = torch.stack(feats, dim=1)
    feat = feat.flatten(3)
    feat = rearrange(feat, "b n c hw -> b (hw n) c")
    return feat.squeeze(0)


def run_lyapunov_rollout(
    *,
    task_name: str,
    ckpt_dir: str,
    ckpt_name: str,
    dataset_dir: str,
    episode_id: int,
    policy_config: dict,
    episode_len: int,
    camera_names: list,
    paper_cfg: PaperLyapunovConfig,
    lyapunov_log_stride: int = 10,
    disturb_t: int = -1,
    disturb_type: str = "none",
    delta_xyz=(0.05, 0.0, 0.0),
    delta_left_arm=None,
    delta_right_arm=None,
    onscreen_render: bool = False,
    fallback_seed: int = 0,
    save_video: bool = False,
    video_path: str | None = None,
):
    qpos_ref, action_ref, images_ref, env_state, h5_path = load_reference_episode(
        dataset_dir, episode_id, camera_names
    )
    horizon = int(policy_config["num_queries"])
    if horizon != 50:
        raise NotImplementedError("Paper sparse visual indices currently assume N=50.")
    if env_state is None:
        raise RuntimeError(
            f"{h5_path} missing meta/env_state. Re-record or annotate episodes."
        )

    last_t_vn = len(qpos_ref) - horizon - 1
    if last_t_vn < 0:
        raise RuntimeError(f"Reference episode shorter than horizon {horizon}")

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ViperPolicy(policy_config)
    try:
        sd = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    except TypeError:
        sd = torch.load(ckpt_path, map_location="cuda")
    policy.load_state_dict(sd)
    policy.cuda()
    policy.eval()

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    qmean = np.asarray(stats["qpos_mean"], dtype=np.float64).reshape(1, -1)
    qstd = np.asarray(stats["qpos_std"], dtype=np.float64).reshape(1, -1)
    pad_qpos_normalized = (np.zeros((1, qmean.shape[1]), dtype=np.float32) - qmean.astype(np.float32)) / qstd.astype(np.float32)
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process_action = lambda a: a * stats["action_std"] + stats["action_mean"]
    post_process_qpos = lambda z: np.asarray(z, dtype=np.float64) * qstd + qmean

    max_timesteps = min(episode_len, len(qpos_ref) - 1, len(action_ref))
    if max_timesteps < 2:
        raise RuntimeError(f"Episode too short: len(qpos)={len(qpos_ref)}")

    BOX_POSE[0] = env_state.copy()
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    set_seed(fallback_seed)
    ts = env.reset()

    query_frequency = 50
    ly_times = []
    vn_imi_list = []
    vn_exp_list = []
    vn_pro_list = []
    vn_img_list = []
    vn_u_list = []
    rewards_list = []

    image_queue = deque(maxlen=50)
    qpos_queue = deque(maxlen=50)
    all_actions = None

    image_list: list | None = [] if save_video else None

    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    onscreen_cam = "angle"
    if onscreen_render:
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        plt.ion()

    with torch.inference_mode():
        for t in range(max_timesteps):
            if onscreen_render:
                import matplotlib.pyplot as plt

                image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                plt_img.set_data(image)
                plt.pause(DT)

            physics = env._physics
            if disturb_t >= 0 and t == disturb_t:
                apply_disturbance(
                    physics,
                    task_name,
                    disturb_type,
                    delta_xyz=np.asarray(delta_xyz, dtype=np.float64),
                    delta_left_arm=delta_left_arm,
                    delta_right_arm=delta_right_arm,
                )

            qpos_numpy = np.array(env.task.get_qpos(physics), dtype=np.float64)
            images_dict = {
                cam: physics.render(height=480, width=640, camera_id=cam) for cam in camera_names
            }
            if save_video and image_list is not None:
                image_list.append({k: np.asarray(v, dtype=np.uint8).copy() for k, v in images_dict.items()})

            class _ShimTs:
                pass

            shim_ts = _ShimTs()
            shim_ts.observation = {"qpos": qpos_numpy, "images": images_dict}
            curr_image = get_image(shim_ts, camera_names)
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().unsqueeze(0)

            if t == 0:
                image_queue.append(curr_image.cpu().numpy())
                qpos_queue.append(qpos)
                for _ in range(49):
                    zero_image = np.zeros((1, 1, 3, 480, 640), dtype=np.uint8)
                    image_queue.appendleft(zero_image)
                    qpos_queue.appendleft(pad_qpos_normalized.copy())
            else:
                image_queue.append(curr_image)
                qpos_queue.append(qpos)

            indices = [9, 19, 29, 39, 49]
            img = [image_queue[i] for i in indices]
            image_array = np.stack([im.cpu().numpy() if isinstance(im, torch.Tensor) else im for im in img], axis=1)
            qpos_array = np.stack([qp.cpu().numpy() if isinstance(qp, torch.Tensor) else qp for qp in qpos_queue], axis=1)
            image_tensor = torch.tensor(image_array).cuda()
            qpos_tensor = torch.tensor(qpos_array).cuda()
            image_tensor = rearrange(image_tensor, "a b c d e f -> a b (c d) e f")

            image_future_tensor = torch.zeros_like(image_tensor)
            qpos_future_tensor = torch.zeros_like(qpos_tensor)

            if t % query_frequency == 0:
                all_actions = policy(qpos_tensor, qpos_future_tensor, image_tensor, image_future_tensor)

            if (
                lyapunov_log_stride > 0
                and t % lyapunov_log_stride == 0
                and t <= last_t_vn
            ):
                idx_img = slice_reference_future_image_indices(t, horizon)
                if idx_img[-1] < len(images_ref):
                    ref_states, ref_actions = extract_reference_slices(qpos_ref, action_ref, t, horizon)
                    pro_ref = ref_states[:horizon]
                    u_ref = ref_actions

                    frames = images_ref[idx_img]
                    ft = torch.from_numpy(frames.astype(np.float32) / 255.0).cuda()
                    ft = rearrange(ft, "t h w c -> t c h w")
                    ft = normalize_imagenet(ft)
                    img_ref_flat = backbone_future_features_flat(policy.model, ft).cpu().numpy()

                    pred = policy(
                        qpos_tensor,
                        qpos_future_tensor,
                        image_tensor,
                        image_future_tensor,
                        return_full_prediction=True,
                    )
                    hs_img_hat = pred["hs_img"].squeeze(0).cpu().numpy()
                    pro_hat = post_process_qpos(pred["a_proprio"].squeeze(0).cpu().numpy())
                    u_hat = post_process_action(pred["a_hat"].squeeze(0).cpu().numpy())

                    jn, parts = compute_JN_paper(
                        pro_hat,
                        u_hat,
                        pro_ref,
                        u_ref,
                        hs_img_hat,
                        img_ref_flat,
                        paper_cfg,
                        return_parts=True,
                    )
                    ly_times.append(t)
                    vn_imi_list.append(jn)
                    vn_exp_list.append(compute_JN_expert_baseline())
                    vn_pro_list.append(parts["sum_pro"])
                    vn_img_list.append(parts["sum_img"])
                    vn_u_list.append(parts["sum_u"])

            if t == 50:
                query_frequency = 1

            raw_action = all_actions[:, t % query_frequency]
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process_action(raw_action)
            ts = env.step(action)
            rewards_list.append(ts.reward)

    if onscreen_render:
        import matplotlib.pyplot as plt

        plt.close()

    if save_video and video_path and image_list:
        from visualize_episodes import save_videos

        save_videos(image_list, DT, video_path=video_path)
        print(f"Saved rollout video: {video_path}")

    success = float(np.max(rewards_list) == env_max_reward)
    meta = {
        "task_name": task_name,
        "episode_id": episode_id,
        "h5_path": h5_path,
        "ckpt_path": ckpt_path,
        "horizon": horizon,
        "disturb_t": disturb_t,
        "disturb_type": disturb_type,
        "success": success,
        "env_max_reward": float(env_max_reward),
        "lyapunov_log_stride": lyapunov_log_stride,
        "paper_omega_pro": paper_cfg.omega_pro,
        "paper_omega_img": paper_cfg.omega_img,
        "paper_lam_u": paper_cfg.lam_u,
    }
    return {
        "ly_times": np.array(ly_times, dtype=np.int32),
        "vn_imi": np.array(vn_imi_list, dtype=np.float64),
        "vn_expert": np.array(vn_exp_list, dtype=np.float64),
        "vn_imi_pro": np.array(vn_pro_list, dtype=np.float64),
        "vn_imi_img": np.array(vn_img_list, dtype=np.float64),
        "vn_imi_u": np.array(vn_u_list, dtype=np.float64),
        "rewards": np.array(rewards_list, dtype=np.float64),
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--horizon_size", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--feature_loss_weight", type=float, default=0.05)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--disturb_t", type=int, default=-1)
    parser.add_argument("--disturb_type", type=str, default="none", choices=("none", "object", "eef"))
    parser.add_argument("--delta_xyz", type=float, nargs=3, default=[0.05, 0.0, 0.0])
    parser.add_argument(
        "--delta_left_arm",
        type=float,
        nargs=6,
        metavar=("j1", "j2", "j3", "j4", "j5", "j6"),
        default=None,
        help="eef only: left arm 6 joint position deltas (rad), waist..wrist_rotate",
    )
    parser.add_argument(
        "--delta_right_arm",
        type=float,
        nargs=6,
        metavar=("j1", "j2", "j3", "j4", "j5", "j6"),
        default=None,
        help="eef only: right arm 6 joint position deltas (rad)",
    )
    parser.add_argument("--lyapunov_log_stride", type=int, default=10)
    parser.add_argument("--omega_pro", type=float, default=1.0)
    parser.add_argument("--omega_img", type=float, default=1.0)
    parser.add_argument("--lam_u", type=float, default=1.0)
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save MuJoCo rollout RGB (same cameras as policy) to mp4 via visualize_episodes.save_videos",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="_mp4 output path when --save_video (default: results_dir/lyapunov_<tag>_rollout.mp4)",
    )
    args = parser.parse_args()

    if args.disturb_type == "eef" and args.delta_left_arm is None and args.delta_right_arm is None:
        parser.error(
            "--disturb_type eef requires --delta_left_arm and/or --delta_right_arm (each 6 floats, radians)"
        )

    torch.cuda.set_device(args.gpu)

    from constants import TASK_CONFIGS as SIM_TASK_CONFIGS

    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = task_config["camera_names"]
    episode_len = task_config["episode_len"]

    lr_backbone = 1e-5
    policy_config = {
        "lr": 1e-5,
        "num_queries": args.horizon_size,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": lr_backbone,
        "backbone": "resnet18",
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "feature_loss": args.feature_loss_weight > 0,
        "feature_loss_weight": args.feature_loss_weight,
    }

    paper_cfg = PaperLyapunovConfig(
        omega_pro=args.omega_pro,
        omega_img=args.omega_img,
        lam_u=args.lam_u,
    )
    dl = np.asarray(args.delta_left_arm, dtype=np.float64) if args.delta_left_arm is not None else None
    dr = np.asarray(args.delta_right_arm, dtype=np.float64) if args.delta_right_arm is not None else None

    os.makedirs(args.results_dir, exist_ok=True)
    tag = f"ep{args.episode_id}"
    if args.disturb_t >= 0:
        tag += f"_dist{args.disturb_t}_{args.disturb_type}"
    video_out = None
    if args.save_video:
        video_out = args.video_path.strip() or os.path.join(args.results_dir, f"lyapunov_{tag}_rollout.mp4")

    out = run_lyapunov_rollout(
        task_name=args.task_name,
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        dataset_dir=args.dataset_dir,
        episode_id=args.episode_id,
        policy_config=policy_config,
        episode_len=episode_len,
        camera_names=camera_names,
        paper_cfg=paper_cfg,
        lyapunov_log_stride=args.lyapunov_log_stride,
        disturb_t=args.disturb_t,
        disturb_type=args.disturb_type,
        delta_xyz=tuple(args.delta_xyz),
        delta_left_arm=dl,
        delta_right_arm=dr,
        onscreen_render=args.onscreen_render,
        fallback_seed=args.seed,
        save_video=args.save_video,
        video_path=video_out,
    )

    npz_path = os.path.join(args.results_dir, f"lyapunov_{tag}.npz")
    np.savez_compressed(
        npz_path,
        ly_times=out["ly_times"],
        vn_imi=out["vn_imi"],
        vn_expert=out["vn_expert"],
        vn_imi_pro=out["vn_imi_pro"],
        vn_imi_img=out["vn_imi_img"],
        vn_imi_u=out["vn_imi_u"],
        rewards=out["rewards"],
    )
    json_path = os.path.join(args.results_dir, f"lyapunov_{tag}_meta.json")
    with open(json_path, "w") as f:
        json.dump(out["meta"], f, indent=2)
    print(f"Saved {npz_path} and {json_path}")


if __name__ == "__main__":
    main()
