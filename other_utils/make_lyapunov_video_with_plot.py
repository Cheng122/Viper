#!/usr/bin/env python3
"""
Compose rollout video (from eval_lyapunov --save_video) with a time-synchronized
Lyapunov V_N plot (from the matching .npz).

Each output frame = [video frame at step n] | [curve truncated to all points with ly_times <= n],
plus a vertical cursor at n and optional disturbance line from meta JSON.

Example:
  python other_utils/make_lyapunov_video_with_plot.py \\
    --video results/lyapunov_insertion_dist_eef/my_rollout_dist90.mp4 \\
    --npz results/lyapunov_insertion_dist_eef/lyapunov_ep0_dist90_eef.npz \\
    --meta_json results/lyapunov_insertion_dist_eef/lyapunov_ep0_dist90_eef_meta.json \\
    --out results/lyapunov_insertion_dist_eef/rollout_dist90_sync.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _fig_to_rgb(fig: Figure, canvas: FigureCanvasAgg) -> np.ndarray:
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return buf[:, :, :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, required=True, help="Rollout mp4 from eval_lyapunov --save_video")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--meta_json", type=str, default="", help="Optional *_meta.json (disturb_t, stride text)")
    parser.add_argument("--out", type=str, required=True, help="Output mp4 path")
    parser.add_argument("--burn_in", type=int, default=0, help="Drop Lyapunov samples with ly_times < burn_in")
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=100,
        help="Matplotlib dpi for the right-hand plot panel (width scales with dpi).",
    )
    parser.add_argument(
        "--no_time_cursor",
        action="store_true",
        help="Do not draw vertical line at current simulation step n on the plot.",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="Output fps (0 = use video's reported fps)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Missing video: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.npz):
        print(f"Missing npz: {args.npz}", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.npz)
    ly_times = data["ly_times"].astype(np.int64)
    vn_imi = np.asarray(data["vn_imi"], dtype=np.float64)
    vn_exp = np.asarray(data["vn_expert"], dtype=np.float64)
    rewards = np.asarray(data["rewards"], dtype=np.float64)
    n_rollout = int(rewards.shape[0])

    if args.burn_in > 0:
        mask = ly_times >= args.burn_in
        ly_times, vn_imi, vn_exp = ly_times[mask], vn_imi[mask], vn_exp[mask]

    if ly_times.size == 0:
        print("No Lyapunov samples left after burn_in.", file=sys.stderr)
        sys.exit(1)

    disturb_t: int | None = None
    log_stride: int | None = None
    if args.meta_json and os.path.isfile(args.meta_json):
        with open(args.meta_json, "r") as f:
            meta = json.load(f)
        if meta.get("disturb_t", -1) >= 0:
            disturb_t = int(meta["disturb_t"])
        if meta.get("lyapunov_log_stride"):
            log_stride = int(meta["lyapunov_log_stride"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_out = float(args.fps) if args.fps > 0 else float(vid_fps)
    if not np.isfinite(fps_out) or fps_out <= 0:
        fps_out = 30.0

    n_frames = min(n_vid, n_rollout)
    if n_vid != n_rollout:
        print(
            f"Warning: video frames ({n_vid}) != npz rewards length ({n_rollout}); "
            f"using first {n_frames} frames.",
            file=sys.stderr,
        )

    y_all = np.concatenate([vn_imi, vn_exp])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.05 * max(y_max - y_min, 1e-6)
    y_lim = (y_min - pad, y_max + pad)
    n_max = max(int(ly_times[-1]), n_frames - 1)

    plot_w_in, plot_h_in = 7.2, 4.0
    fig = Figure(figsize=(plot_w_in, plot_h_in), dpi=args.plot_dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    (line_exp,) = ax.plot([], [], label=r"$V_N$ nominal reference (self-track)", color="#1f77b4", linewidth=1.8)
    (line_imi,) = ax.plot([], [], label=r"$V_N$ ImiNMPC", color="#d62728", linewidth=1.8)
    if disturb_t is not None:
        ax.axvline(disturb_t, color="0.35", linestyle="--", linewidth=1.2, label="disturbance")
    cursor_line = None
    if not args.no_time_cursor:
        (cursor_line,) = ax.plot([], [], color="#2ca02c", linewidth=1.5, alpha=0.85, label=r"current $n$")
    if log_stride and log_stride > 1:
        ax.set_xlabel(r"time step $n$ (Lyapunov every $\Delta n=%d$)" % log_stride)
    else:
        ax.set_xlabel(r"time step $n$")
    ax.set_ylabel(r"horizon cost $V_N(n)$")
    ax.set_xlim(0, n_max)
    ax.set_ylim(y_lim)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    plot_rgb = _fig_to_rgb(fig, canvas)
    ph, pw = plot_rgb.shape[:2]

    target_h = max(h0, ph)
    scale_v = target_h / float(h0)
    new_vw = int(round(w0 * scale_v))
    new_vh = target_h
    scale_p = target_h / float(ph)
    new_pw = int(round(pw * scale_p))
    new_ph = target_h

    out_w = new_vw + new_pw
    out_h = target_h
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        print(f"Could not open VideoWriter for {args.out}", file=sys.stderr)
        sys.exit(1)

    for frame_idx in range(n_frames):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            print(f"Warning: short read at frame {frame_idx}", file=sys.stderr)
            break

        frame_rs = cv2.resize(frame_bgr, (new_vw, new_vh), interpolation=cv2.INTER_AREA)

        k = int(np.searchsorted(ly_times, frame_idx, side="right"))
        if k > 0:
            line_exp.set_data(ly_times[:k], vn_exp[:k])
            line_imi.set_data(ly_times[:k], vn_imi[:k])
        else:
            line_exp.set_data([], [])
            line_imi.set_data([], [])

        if cursor_line is not None:
            cursor_line.set_data([frame_idx, frame_idx], [y_lim[0], y_lim[1]])

        ax.set_title(r"$V_N$ vs rollout (prefix for $n \leq %d$)" % frame_idx, fontsize=10)
        plot_rgb = _fig_to_rgb(fig, canvas)
        plot_rs = cv2.resize(plot_rgb, (new_pw, new_ph), interpolation=cv2.INTER_AREA)
        plot_bgr = cv2.cvtColor(plot_rs, cv2.COLOR_RGB2BGR)

        composite = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        composite[:, :new_vw] = frame_rs
        composite[:, new_vw:new_vw + new_pw] = plot_bgr
        writer.write(composite)

    cap.release()
    writer.release()
    print(f"Saved {os.path.abspath(args.out)} ({n_frames} frames @ {fps_out:.2f} fps)")


if __name__ == "__main__":
    main()
