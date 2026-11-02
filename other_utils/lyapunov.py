"""
Finite-horizon cost J_N for ImiNMPC vs recorded reference.
Legacy compute_VN (qpos rollout) kept for compatibility; paper figures use compute_JN_paper.
See lyapunov_spec.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_SPARSE_K: Tuple[int, ...] = (9, 19, 29, 39, 49)
NUM_FUTURE_VISUAL_FRAMES = 5


@dataclass
class PaperLyapunovConfig:
    omega_pro: float = 1.0
    omega_img: float = 1.0
    lam_u: float = 1.0
    sparse_k_0based: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_SPARSE_K)


@dataclass
class LyapunovConfig:
    alpha_state: float = 1.0
    beta_control: float = 1.0
    gamma_terminal: float = 1.0


def stage_cost(
    x_hat: np.ndarray,
    u_hat: np.ndarray,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    cfg: LyapunovConfig,
) -> float:
    return float(
        cfg.alpha_state * np.linalg.norm(x_hat - x_ref, ord=1)
        + cfg.beta_control * np.linalg.norm(u_hat - u_ref, ord=1)
    )


def terminal_cost(x_hat_N: np.ndarray, x_ref_N: np.ndarray, cfg: LyapunovConfig) -> float:
    return float(cfg.gamma_terminal * np.linalg.norm(x_hat_N - x_ref_N, ord=1))


def rollout_predicted_states(x0: np.ndarray, u_sequence: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    u_sequence = np.asarray(u_sequence, dtype=np.float64)
    if u_sequence.ndim != 2:
        raise ValueError("u_sequence must be (N, state_dim)")
    n = u_sequence.shape[0]
    d = x0.shape[0]
    x_hat = np.zeros((n + 1, d), dtype=np.float64)
    x_hat[0] = x0
    x_hat[1:] = u_sequence
    return x_hat


def compute_VN(
    x0_measured: np.ndarray,
    u_hat: np.ndarray,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    cfg: LyapunovConfig | None = None,
) -> float:
    cfg = cfg or LyapunovConfig()
    u_hat = np.asarray(u_hat, dtype=np.float64)
    ref_states = np.asarray(ref_states, dtype=np.float64)
    ref_actions = np.asarray(ref_actions, dtype=np.float64)
    n = u_hat.shape[0]
    if ref_states.shape != (n + 1, u_hat.shape[1]):
        raise ValueError(
            f"ref_states must be (N+1, d)={(n + 1, u_hat.shape[1])}, got {ref_states.shape}"
        )
    if ref_actions.shape != (n, u_hat.shape[1]):
        raise ValueError(f"ref_actions must be (N, d)={(n, u_hat.shape[1])}, got {ref_actions.shape}")

    x_hat = rollout_predicted_states(x0_measured, u_hat)
    total = 0.0
    for k in range(n):
        total += stage_cost(x_hat[k], u_hat[k], ref_states[k], ref_actions[k], cfg)
    total += terminal_cost(x_hat[n], ref_states[n], cfg)
    return float(total)


def compute_VN_expert_self_track(
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    cfg: LyapunovConfig | None = None,
) -> float:
    _ = ref_states, ref_actions, cfg
    return 0.0


def compute_JN_expert_baseline() -> float:
    return 0.0


def extract_reference_slices(
    qpos_ref: np.ndarray,
    action_ref: np.ndarray,
    t: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t = int(t)
    horizon = int(horizon)
    if t + horizon >= qpos_ref.shape[0]:
        raise IndexError(f"Need qpos rows through index {t+horizon}, have {qpos_ref.shape[0]}")
    if t + horizon > action_ref.shape[0]:
        raise IndexError(f"Need action rows through {t+horizon-1}, have {action_ref.shape[0]}")
    ref_states = qpos_ref[t : t + horizon + 1].astype(np.float64)
    ref_actions = action_ref[t : t + horizon].astype(np.float64)
    return ref_states, ref_actions


def _visual_block_index(k: int) -> int:
    return (k + 1) // 10 - 1


def compute_JN_paper(
    pro_hat: np.ndarray,
    u_hat: np.ndarray,
    pro_ref: np.ndarray,
    u_ref: np.ndarray,
    hs_img_hat: np.ndarray,
    img_ref_flat: np.ndarray,
    cfg: PaperLyapunovConfig | None = None,
    *,
    num_visual_frames: int = NUM_FUTURE_VISUAL_FRAMES,
    return_parts: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    cfg = cfg or PaperLyapunovConfig()
    pro_hat = np.asarray(pro_hat, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    pro_ref = np.asarray(pro_ref, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    hs_img_hat = np.asarray(hs_img_hat, dtype=np.float64)
    img_ref_flat = np.asarray(img_ref_flat, dtype=np.float64)

    n = pro_hat.shape[0]
    if u_hat.shape[0] != n or pro_ref.shape[0] != n or u_ref.shape[0] != n:
        raise ValueError("pro_hat, u_hat, pro_ref, u_ref must share horizon length N")
    if hs_img_hat.shape != img_ref_flat.shape:
        raise ValueError(f"hs_img_hat {hs_img_hat.shape} vs img_ref_flat {img_ref_flat.shape}")
    L = hs_img_hat.shape[0]
    if L % num_visual_frames != 0:
        raise ValueError(f"Visual length {L} not divisible by {num_visual_frames}")
    block_len = L // num_visual_frames

    sparse_set = set(cfg.sparse_k_0based)
    s_pro = 0.0
    s_img = 0.0
    s_u = 0.0

    for k in range(n):
        d_pro = cfg.omega_pro * np.mean(np.abs(pro_hat[k] - pro_ref[k]))
        d_u = cfg.lam_u * np.mean(np.abs(u_hat[k] - u_ref[k]))
        d_img = 0.0
        if k in sparse_set:
            b = _visual_block_index(k)
            if b < 0 or b >= num_visual_frames:
                raise ValueError(f"bad sparse k={k} -> block {b}")
            sl = slice(b * block_len, (b + 1) * block_len)
            d_img = cfg.omega_img * np.mean((hs_img_hat[sl] - img_ref_flat[sl]) ** 2)
        s_pro += d_pro
        s_img += d_img
        s_u += d_u

    jn = float(s_pro + s_img + s_u)
    if return_parts:
        return jn, {"sum_pro": s_pro, "sum_img": s_img, "sum_u": s_u}
    return jn


def slice_reference_future_image_indices(t: int, horizon: int = 50) -> List[int]:
    if horizon != 50:
        raise NotImplementedError("sparse visual indices only defined for N=50")
    return [t + 10 * i for i in range(1, 6)]
