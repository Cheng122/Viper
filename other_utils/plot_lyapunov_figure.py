#!/usr/bin/env python3
"""Publication-style Lyapunov plot from eval_lyapunov.py outputs (.npz + optional meta JSON)."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--meta_json", type=str, default="", help="Optional lyapunov_*_meta.json for disturb_t vertical line")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--burn_in", type=int, default=0, help="Drop first K samples (e.g. zero-padded image history)")
    args = parser.parse_args()

    data = np.load(args.npz)
    t = data["ly_times"].astype(int)
    vn_imi = data["vn_imi"]
    vn_exp = data["vn_expert"]

    if args.burn_in > 0:
        mask = t >= args.burn_in
        t, vn_imi, vn_exp = t[mask], vn_imi[mask], vn_exp[mask]

    disturb_t = None
    log_stride = None
    if args.meta_json and os.path.isfile(args.meta_json):
        with open(args.meta_json, "r") as f:
            meta = json.load(f)
        if meta.get("disturb_t", -1) >= 0:
            disturb_t = int(meta["disturb_t"])
        if meta.get("lyapunov_log_stride"):
            log_stride = int(meta["lyapunov_log_stride"])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, vn_exp, label=r"$V_N$ nominal reference (self-track)", color="#1f77b4", linewidth=1.8)
    ax.plot(t, vn_imi, label=r"$V_N$ ImiNMPC", color="#d62728", linewidth=1.8)
    if disturb_t is not None:
        ax.axvline(disturb_t, color="0.35", linestyle="--", linewidth=1.2, label="disturbance")
    if log_stride and log_stride > 1:
        ax.set_xlabel(r"time step $n$ (markers every $\Delta n=%d$)" % log_stride)
    else:
        ax.set_xlabel(r"time step $n$")
    ax.set_ylabel(r"horizon cost $V_N(n)$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
