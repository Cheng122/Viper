#!/usr/bin/env python3
"""
Overlay multiple Lyapunov curves from eval_lyapunov.py runs on one figure.

Each --series supplies:  LABEL | path/to/lyapunov_*.npz [| optional_meta.json]
If meta is omitted, uses <npz_basename>_meta.json next to the .npz (same rule as eval_lyapunov saves).

Example (adjust paths if your .npz filenames differ):
  python other_utils/plot_lyapunov_compare.py \\
    --out results/lyapunov_insertion/compare_four.png \\
    --series "no disturbance|results/lyapunov_insertion/lyapunov_ep0.npz" \\
    --series "object disturbance, $n_d=60$|results/lyapunov_insertion_dist_obs/lyapunov_ep0_dist60_object.npz" \\
    --series "EEF (large), $n_d=60$|results/lyapunov_insertion_huge_dist_eef/lyapunov_ep0_dist60_eef.npz" \\
    --series "EEF, $n_d=90$|results/lyapunov_insertion_dist_eef/lyapunov_ep0_dist90_eef.npz"
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def infer_meta_path(npz_path: str) -> str:
    d = os.path.dirname(os.path.abspath(npz_path))
    base = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(d, f"{base}_meta.json")


def parse_series_arg(s: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in s.split("|")]
    if len(parts) == 2:
        label, npz = parts
        meta = infer_meta_path(npz)
    elif len(parts) == 3:
        label, npz, meta = parts
    else:
        raise argparse.ArgumentTypeError(
            "Each --series must be 'LABEL|NPZ' or 'LABEL|NPZ|META_JSON'"
        )
    return label, npz, meta


def load_meta(meta_path: str) -> dict | None:
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        type=parse_series_arg,
        metavar="LABEL|NPZ|[META]",
        help="Repeat for each curve (see docstring).",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--burn_in", type=int, default=0)
    parser.add_argument("--figsize", type=float, nargs=2, default=[8.0, 4.5])
    parser.add_argument(
        "--no_expert",
        action="store_true",
        help="Do not plot the nominal expert baseline (all zeros).",
    )
    args = parser.parse_args()

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    expert_plotted = False
    log_stride = None

    for idx, (label, npz_path, meta_path) in enumerate(args.series):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        t = data["ly_times"].astype(int)
        vn_imi = np.asarray(data["vn_imi"], dtype=np.float64)
        vn_exp = np.asarray(data["vn_expert"], dtype=np.float64)

        if args.burn_in > 0:
            mask = t >= args.burn_in
            t, vn_imi, vn_exp = t[mask], vn_imi[mask], vn_exp[mask]

        meta = load_meta(meta_path)
        if meta is not None and meta.get("lyapunov_log_stride"):
            log_stride = int(meta["lyapunov_log_stride"])

        c = colors[idx % len(colors)]
        if not args.no_expert and not expert_plotted:
            ax.plot(
                t,
                vn_exp,
                label=r"$V_N$ nominal reference (self-track)",
                color="#1f77b4",
                linewidth=1.8,
                zorder=1,
            )
            expert_plotted = True

        leg_label = label
        ax.plot(t, vn_imi, label=rf"$V_N$ ImiNMPC ({leg_label})", color=c, linewidth=1.8, zorder=2)

        if meta is not None:
            dt = int(meta.get("disturb_t", -1))
            if dt >= 0:
                ax.axvline(
                    dt,
                    color=c,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    dashes=(6, 4),
                    zorder=0,
                )

    if log_stride and log_stride > 1:
        ax.set_xlabel(r"time step $n$ (markers every $\Delta n=%d$)" % log_stride)
    else:
        ax.set_xlabel(r"time step $n$")
    ax.set_ylabel(r"horizon cost $V_N(n)$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
