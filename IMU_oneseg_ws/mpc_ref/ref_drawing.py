#!/usr/bin/env python3
"""
Real-time tip reference viewer (RB1 X–Z).

Reads ``tip_ref.csv`` (columns ``x``, ``z``) in this folder and animates
the trajectory in a matplotlib window. No files are saved.

Fig.5 in ``00data_processing.py`` overwrites this same file for any shape
(svg / icra / circle / square / star).

  cd /root/DataCollection/IMU_oneseg_ws/mpc_ref
  python ref_drawing.py

Or point at another CSV:

  python ref_drawing.py /path/to/other.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "tip_ref.csv"

# Playback knobs
FPS = 60                    # animation frame rate (display)
POINTS_PER_FRAME = 4        # how many CSV rows to advance each frame
TRAIL_COLOR = "tab:red"
DOT_COLOR = "tab:blue"
BG_COLOR = "0.75"


def load_xz(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "x" not in cols or "z" not in cols:
        raise ValueError(f"{csv_path} must have columns x,z; got {list(df.columns)}")
    xz = df[[cols["x"], cols["z"]]].to_numpy(dtype=np.float64)
    if xz.ndim != 2 or xz.shape[1] != 2 or xz.shape[0] < 2:
        raise ValueError(f"need ≥2 (x,z) rows, got shape {xz.shape}")
    return xz


def main() -> None:
    csv_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.is_file():
        # allow bare name in this folder
        alt = HERE / csv_path.name
        if alt.is_file():
            csv_path = alt
        else:
            raise FileNotFoundError(
                f"missing {csv_path}\n"
                f"Run Fig.5 in 00data_processing.py with save_fig5_ref_csv=True "
                f"to write mpc_ref/tip_ref.csv, or pass a CSV path."
            )

    traj = load_xz(csv_path)
    n = int(traj.shape[0])
    print(f"[ref_drawing] {csv_path.name}: N={n}  "
          f"x=[{traj[:, 0].min():.4f},{traj[:, 0].max():.4f}]  "
          f"z=[{traj[:, 1].min():.4f},{traj[:, 1].max():.4f}]")

    fig, ax = plt.subplots(figsize=(8.0, 7.0), layout="constrained")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)  [RB1]")
    ax.set_ylabel("z (m)  [RB1 / base plane]")
    ax.set_title(f"Tip reference (real-time)  |  {csv_path.name}  |  N={n}")

    # Full path faint background
    ax.plot(traj[:, 0], traj[:, 1], color=BG_COLOR, lw=1.0, zorder=1, label="full path")
    ax.scatter([0.0], [0.0], c="k", s=50, marker="^", zorder=5, label="RB1 origin")
    ax.scatter([traj[0, 0]], [traj[0, 1]], c="limegreen", s=40, zorder=5, label="start")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], c="tab:orange", s=40, zorder=5, label="end")

    (trail,) = ax.plot([], [], color=TRAIL_COLOR, lw=2.0, zorder=3, label="drawn")
    (dot,) = ax.plot([], [], "o", color=DOT_COLOR, ms=7, zorder=4)

    pad = 0.02
    ax.set_xlim(float(traj[:, 0].min()) - pad, float(traj[:, 0].max()) + pad)
    ax.set_ylim(float(traj[:, 1].min()) - pad, float(traj[:, 1].max()) + pad)
    ax.legend(fontsize=8, loc="best")

    step = max(int(POINTS_PER_FRAME), 1)
    n_frames = int(np.ceil(n / step))

    def init():
        trail.set_data([], [])
        dot.set_data([], [])
        return trail, dot

    def update(frame: int):
        i = min((frame + 1) * step, n)
        trail.set_data(traj[:i, 0], traj[:i, 1])
        dot.set_data([traj[i - 1, 0]], [traj[i - 1, 1]])
        return trail, dot

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / float(FPS),
        blit=True,
        repeat=True,
    )
    # keep reference so GC does not stop the animation
    fig._ref_drawing_ani = ani  # type: ignore[attr-defined]
    plt.show()


if __name__ == "__main__":
    main()
