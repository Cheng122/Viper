"""
Physics perturbations for Lyapunov / robustness experiments (simulation only).
"""
from __future__ import annotations

import numpy as np


def apply_object_translation(physics, task_name: str, delta_xyz: np.ndarray) -> None:
    """Translate free-joint object(s) position components by delta_xyz (3,)."""
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    qpos = physics.data.qpos.copy()
    if "transfer_cube" in task_name:
        qpos[-7:-4] += delta_xyz
    elif any(x in task_name for x in ("insertion", "stack_cube", "storage_cube")):
        qpos[-14:-11] += delta_xyz
        qpos[-7:-4] += delta_xyz
    else:
        raise NotImplementedError(f"object perturbation for {task_name}")
    physics.data.qpos[:] = qpos
    physics.forward()


def apply_eef_joint_bias(physics, delta_left_arm: np.ndarray, delta_right_arm: np.ndarray) -> None:
    """Add small biases to arm joint positions (first 6 left, then skip gripper, 6 right in full qpos[:16])."""
    dl = np.asarray(delta_left_arm, dtype=np.float64).reshape(6)
    dr = np.asarray(delta_right_arm, dtype=np.float64).reshape(6)
    qpos = physics.data.qpos.copy()
    qpos[:6] += dl
    qpos[8:14] += dr
    physics.data.qpos[:] = qpos
    physics.forward()


def apply_disturbance(
    physics,
    task_name: str,
    disturb_type: str,
    delta_xyz=(0.0, 0.0, 0.0),
    delta_left_arm=None,
    delta_right_arm=None,
) -> None:
    disturb_type = disturb_type.lower()
    if disturb_type == "none":
        return
    if disturb_type == "object":
        apply_object_translation(physics, task_name, np.asarray(delta_xyz, dtype=np.float64))
    elif disturb_type == "eef":
        dl = np.zeros(6) if delta_left_arm is None else np.asarray(delta_left_arm, dtype=np.float64)
        dr = np.zeros(6) if delta_right_arm is None else np.asarray(delta_right_arm, dtype=np.float64)
        apply_eef_joint_bias(physics, dl, dr)
    else:
        raise ValueError(disturb_type)
