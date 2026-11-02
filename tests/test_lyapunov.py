import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "other_utils"))

from lyapunov import (
    LyapunovConfig,
    PaperLyapunovConfig,
    NUM_FUTURE_VISUAL_FRAMES,
    compute_JN_expert_baseline,
    compute_JN_paper,
    compute_VN,
    compute_VN_expert_self_track,
    extract_reference_slices,
    rollout_predicted_states,
    slice_reference_future_image_indices,
    stage_cost,
)


def test_rollout_shape():
    x0 = np.zeros(14)
    u = np.ones((50, 14))
    x_hat = rollout_predicted_states(x0, u)
    assert x_hat.shape == (51, 14)
    assert np.allclose(x_hat[0], x0)
    assert np.allclose(x_hat[1:], u)


def test_expert_zero():
    rs = np.random.randn(51, 14)
    ra = np.random.randn(50, 14)
    assert compute_VN_expert_self_track(rs, ra) == 0.0


def test_VN_positive_when_mismatch():
    rng = np.random.default_rng(0)
    ref_states = rng.standard_normal((51, 14))
    ref_actions = rng.standard_normal((50, 14))
    x0 = ref_states[0].copy()
    u_hat = ref_actions + 1.0
    vn = compute_VN(x0, u_hat, ref_states, ref_actions, LyapunovConfig())
    assert vn > 1e-6


def test_VN_zero_when_identical_open_loop():
    """If reference trajectory satisfies open-loop model x_{k+1}=u_k, perfect match → 0."""
    h = 10
    d = 14
    x0 = np.zeros(d)
    u = np.random.default_rng(1).standard_normal((h, d))
    ref_actions = u.copy()
    ref_states = rollout_predicted_states(x0, u)
    vn = compute_VN(x0, u, ref_states, ref_actions, LyapunovConfig())
    assert vn < 1e-9


def test_extract_slices():
    q = np.arange(600).reshape(100, 6)
    a = np.arange(500).reshape(100, 5)
    rs, ra = extract_reference_slices(q, a, t=10, horizon=8)
    assert rs.shape == (9, 6)
    assert ra.shape == (8, 5)
    assert rs[0, 0] == q[10, 0]


def test_extract_slices_oob():
    q = np.zeros((20, 14))
    a = np.zeros((20, 14))
    with pytest.raises(IndexError):
        extract_reference_slices(q, a, t=0, horizon=25)


def _random_paper_inputs(n=50, d=14, spatial_per_frame=4, c=512):
    rng = np.random.default_rng(42)
    pro_hat = rng.standard_normal((n, d))
    u_hat = rng.standard_normal((n, d))
    pro_ref = pro_hat.copy()
    u_ref = u_hat.copy()
    L = spatial_per_frame * NUM_FUTURE_VISUAL_FRAMES
    hs = rng.standard_normal((L, c))
    img_ref = hs.copy()
    return pro_hat, u_hat, pro_ref, u_ref, hs, img_ref


def test_JN_paper_zero_identical():
    pro_hat, u_hat, pro_ref, u_ref, hs, img_ref = _random_paper_inputs()
    jn = compute_JN_paper(pro_hat, u_hat, pro_ref, u_ref, hs, img_ref, PaperLyapunovConfig())
    assert jn < 1e-9


def test_JN_paper_positive_pro_mismatch():
    pro_hat, u_hat, pro_ref, u_ref, hs, img_ref = _random_paper_inputs()
    pro_hat = pro_hat + 1.0
    jn = compute_JN_paper(pro_hat, u_hat, pro_ref, u_ref, hs, img_ref, PaperLyapunovConfig())
    assert jn > 1e-6


def test_JN_paper_positive_img_mismatch_sparse_only():
    pro_hat, u_hat, pro_ref, u_ref, hs, img_ref = _random_paper_inputs(spatial_per_frame=8)
    block_len = hs.shape[0] // NUM_FUTURE_VISUAL_FRAMES
    hs[:block_len, :] += 2.0
    jn = compute_JN_paper(pro_hat, u_hat, pro_ref, u_ref, hs, img_ref, PaperLyapunovConfig())
    assert jn > 1e-6


def test_JN_paper_shape_mismatch_raises():
    pro_hat, u_hat, pro_ref, u_ref, hs, img_ref = _random_paper_inputs()
    with pytest.raises(ValueError):
        compute_JN_paper(pro_hat, u_hat, pro_ref, u_ref, hs[:-1], img_ref)


def test_JN_expert_baseline_zero():
    assert compute_JN_expert_baseline() == 0.0


def test_slice_reference_future_image_indices():
    assert slice_reference_future_image_indices(3, 50) == [13, 23, 33, 43, 53]


def test_JN_paper_return_parts():
    pro_hat, u_hat, pro_ref, u_ref, hs, img_ref = _random_paper_inputs()
    jn, parts = compute_JN_paper(
        pro_hat, u_hat, pro_ref, u_ref, hs, img_ref, PaperLyapunovConfig(), return_parts=True
    )
    assert "sum_pro" in parts and "sum_img" in parts and "sum_u" in parts
    assert abs(jn - (parts["sum_pro"] + parts["sum_img"] + parts["sum_u"])) < 1e-9
