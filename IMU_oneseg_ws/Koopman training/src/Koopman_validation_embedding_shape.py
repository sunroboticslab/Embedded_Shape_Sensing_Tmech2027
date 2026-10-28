#!/usr/bin/env python
"""
Koopman data-size convergence on NN-reconstructed shape.

Fits the same linear Koopman as ``Koopman_training_embedding_shape.py`` at
300k / 500k / 700k / 1M / 1.5M **train** frames (val split unchanged), then:

  1. Same per-model figures as the training script (A eigs, base-plane shape rollout)
  2. 100 random val trajectories: mean error curve + ±std shadow
  3. Overlay comparison across train sizes

One-seg base plane = **x–z** in RB1 (``plane="xz"`` from NN config).

    python Koopman_validation_embedding_shape.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Koopman_training_embedding_shape as kt  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TRAIN_SIZES = (300_000, 500_000, 700_000, 1_000_000, 1_500_000)
_MODEL_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")


def _ntr_tag(n: int) -> str:
    n = int(n)
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def _cap_train_row(train_row: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    idx = np.flatnonzero(train_row)
    n_use = min(int(n), int(idx.size))
    out = np.zeros_like(train_row)
    if n_use > 0:
        out[idx[:n_use]] = True
    return out, n_use


def _valid_k0s(
    trial: np.ndarray,
    train_row: np.ndarray,
    *,
    delay: int,
    horizon: int,
) -> np.ndarray:
    dly = int(delay)
    n_pred = int(horizon)
    val_idx = np.flatnonzero(~train_row)
    val_idx = val_idx[(val_idx >= dly) & (val_idx + n_pred + 2 < trial.size)]
    keep: list[int] = []
    for k in val_idx:
        sl = trial[k - dly : k + n_pred + 1]
        if sl.size == (dly + n_pred + 1) and np.all(sl == trial[k]):
            keep.append(int(k))
    return np.asarray(keep, dtype=np.int64)


def _pick_k0s(candidates: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    cand = np.asarray(candidates, dtype=np.int64)
    n = int(cand.size)
    k = int(k)
    if n <= 0 or k <= 0:
        return []
    k = min(k, n)
    gap = max(1, n // max(k * 4, 1))
    order = rng.permutation(n)
    picked: list[int] = []
    for i in order:
        c = int(cand[int(i)])
        if all(abs(c - j) >= gap for j in picked):
            picked.append(c)
        if len(picked) >= k:
            break
    if len(picked) < k:
        picked = [int(cand[i]) for i in np.linspace(0, n - 1, k, dtype=int)]
    return sorted(picked[:k])


def _rollout_one(
    koop: dict[str, Any],
    xn: np.ndarray,
    un: np.ndarray,
    x_gt: np.ndarray,
    x_mu: np.ndarray,
    x_sd: np.ndarray,
    k0: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_n = kt.rollout(koop, xn, un, int(k0), int(horizon))
    true_n = xn[k0 + 1 : k0 + 1 + int(horizon)]
    h = min(len(pred_n), len(true_n), int(x_gt.shape[0]) - k0 - 1)
    pred_x = kt._denorm(pred_n[:h], x_mu, x_sd)
    nn_x = kt._denorm(true_n[:h], x_mu, x_sd)
    gt_x = np.asarray(x_gt[k0 + 1 : k0 + 1 + h], dtype=np.float64)
    return pred_x, nn_x, gt_x


def _rmse_series(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-step RMSE over all packed state channels [m] (xz coords for one-seg)."""
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def _rollout_errors(
    pred_x: np.ndarray,
    nn_x: np.ndarray,
    gt_x: np.ndarray,
    names: list[str],
    *,
    plane: str,
    tip_rb: int,
) -> dict[str, np.ndarray]:
    """State + base-plane geometric + tip (RB5 x–z) errors along a rollout."""
    out: dict[str, np.ndarray] = {
        "state_vs_gt": _rmse_series(pred_x, gt_x),
        "state_vs_nn": _rmse_series(pred_x, nn_x),
        "plane_vs_gt": kt._mean_plane_rmse_series(pred_x, gt_x, names, plane=plane),
        "plane_vs_nn": kt._mean_plane_rmse_series(pred_x, nn_x, names, plane=plane),
    }
    tip_gt = kt._tip_plane_rmse_series(pred_x, gt_x, names, plane=plane, tip_rb=tip_rb)
    tip_nn = kt._tip_plane_rmse_series(pred_x, nn_x, names, plane=plane, tip_rb=tip_rb)
    if tip_gt is not None:
        out["tip_vs_gt"] = tip_gt
    if tip_nn is not None:
        out["tip_vs_nn"] = tip_nn
    return out


def rollout_rmse_trajs(
    koop: dict[str, Any],
    xn: np.ndarray,
    un: np.ndarray,
    x_gt: np.ndarray,
    x_mu: np.ndarray,
    x_sd: np.ndarray,
    k0s: list[int],
    horizon: int,
    *,
    names: list[str],
    plane: str,
    tip_rb: int,
) -> dict[str, np.ndarray]:
    """Return (n_traj, H) error series [m] for state / plane / tip metrics."""
    buckets: dict[str, list[np.ndarray]] = {
        "state_vs_gt": [],
        "state_vs_nn": [],
        "plane_vs_gt": [],
        "plane_vs_nn": [],
    }
    tip_gt_list: list[np.ndarray] = []
    tip_nn_list: list[np.ndarray] = []
    for k0 in k0s:
        pred_x, nn_x, gt_x = _rollout_one(
            koop, xn, un, x_gt, x_mu, x_sd, int(k0), int(horizon)
        )
        errs = _rollout_errors(
            pred_x, nn_x, gt_x, names, plane=plane, tip_rb=tip_rb,
        )
        for key in buckets:
            buckets[key].append(errs[key])
        if "tip_vs_gt" in errs:
            tip_gt_list.append(errs["tip_vs_gt"])
        if "tip_vs_nn" in errs:
            tip_nn_list.append(errs["tip_vs_nn"])
    h = min(len(r) for r in buckets["state_vs_gt"])
    out = {k: np.stack([r[:h] for r in v], axis=0) for k, v in buckets.items()}
    if tip_gt_list:
        out["tip_vs_gt"] = np.stack([r[:h] for r in tip_gt_list], axis=0)
    if tip_nn_list:
        out["tip_vs_nn"] = np.stack([r[:h] for r in tip_nn_list], axis=0)
    return out


def plot_horizon_rmse(
    save: Path,
    *,
    rmse_vs_gt: np.ndarray,
    rmse_vs_nn: np.ndarray,
    sample_rate_hz: float,
    title: str,
    ylabel: str = "state RMSE [mm]",
) -> None:
    t = np.arange(len(rmse_vs_gt)) / float(sample_rate_hz)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    ax.plot(t, 1e3 * rmse_vs_gt, color="tab:red", lw=1.4, label="Koopman vs OptiTrack")
    ax.plot(t, 1e3 * rmse_vs_nn, color="tab:blue", lw=1.1, ls="--", label="Koopman vs NN ŷ")
    ax.set_xlabel("rollout time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    kt._save_fig(fig, save)
    plt.close(fig)


def plot_rmse_shadow(
    save: Path,
    *,
    series: dict[str, np.ndarray],
    sample_rate_hz: float,
    title: str,
    ylabel: str,
) -> None:
    """``series`` maps legend label → (n_traj, H) RMSE [m]."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    for i, (label, err) in enumerate(series.items()):
        color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        err = np.asarray(err, dtype=np.float64)
        mean = np.mean(err, axis=0)
        std = np.std(err, axis=0)
        t = np.arange(mean.size) / float(sample_rate_hz)
        ax.plot(t, 1e3 * mean, color=color, lw=1.6, label=label)
        ax.fill_between(
            t,
            1e3 * np.maximum(mean - std, 0.0),
            1e3 * (mean + std),
            color=color,
            alpha=0.22,
            linewidth=0,
        )
    ax.set_xlabel("rollout time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    kt._save_fig(fig, save)
    plt.close(fig)


def plot_shape_rollout_compare(
    save: Path,
    *,
    gt_x: np.ndarray,
    nn_x: np.ndarray,
    koop_curves: list[tuple[str, np.ndarray]],
    names: list[str],
    option: str,
    plane: str,
    x_mode: str,
    k0: int,
) -> dict[str, Any]:
    groups = kt._shape_plane_groups(names, plane=plane)
    if not groups:
        raise RuntimeError(f"no {plane} position channels in this NN output to plot")
    ax0 = groups[0].get("ax0", "x")
    ax1 = groups[0].get("ax1", "z")
    n_panels = len(groups)
    n_cols = min(5, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.35 * n_cols, 3.15 * n_rows), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()
    panel_stats: list[dict[str, Any]] = []

    for j, g in enumerate(groups):
        ax = axes[j]
        ix, iy = g["ix"], g["iy"]
        ax.plot(gt_x[:, ix], gt_x[:, iy], color="0.35", lw=1.8, label="OptiTrack GT")
        ax.plot(nn_x[:, ix], nn_x[:, iy], color="0.55", lw=1.2, ls=":", label="NN reconstructed")
        rec: dict[str, Any] = {**g}
        title_bits: list[str] = []
        for m_i, (label, kp_x) in enumerate(koop_curves):
            color = _MODEL_COLORS[m_i % len(_MODEL_COLORS)]
            ax.plot(kp_x[:, ix], kp_x[:, iy], color=color, lw=1.15, ls="--", label=label)
            rec[f"rmse_vs_gt_{label}"] = kt._plane_rmse(kp_x, gt_x, ix, iy)
            title_bits.append(f"{label}={1000.0 * rec[f'rmse_vs_gt_{label}']:.2f}")
        ax.scatter(float(gt_x[0, ix]), float(gt_x[0, iy]), c="green", s=14, zorder=5)
        panel_stats.append(rec)
        ax.set_title(f"{g['title']}\nvs GT [mm]: " + ", ".join(title_bits), fontsize=7)
        ax.set_xlabel(f"{g.get('ax0', ax0)} (m)")
        ax.set_ylabel(f"{g.get('ax1', ax1)} (m)")
        ax.grid(True, alpha=0.3)
        try:
            ax.set_aspect("equal", adjustable="datalim")
        except Exception:
            pass
        if j == 0:
            ax.legend(fontsize=6.5, loc="best")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"option {option}/{plane}  x_mode={x_mode}  k0={k0}  steps={len(gt_x)}\n"
        "Koopman by train size vs OptiTrack / NN",
        fontsize=11,
    )
    kt._save_fig(fig, save)
    plt.close(fig)
    return {"panels": panel_stats}


def plot_rmse_vs_ntrain(
    save: Path,
    n_train: np.ndarray,
    mean_gt: np.ndarray,
    std_gt: np.ndarray,
    mean_nn: np.ndarray,
    std_nn: np.ndarray,
    *,
    title: str = "Koopman open-loop RMSE vs training data  (mean over 100 val trajs)",
    ylabel: str = "rollout state RMSE [mm]",
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    ax.errorbar(
        n_train, 1e3 * mean_gt, yerr=1e3 * std_gt,
        fmt="-o", color="tab:red", capsize=4, lw=1.6, label=r"vs GT  mean$\pm$std",
    )
    ax.errorbar(
        n_train, 1e3 * mean_nn, yerr=1e3 * std_nn,
        fmt="-s", color="tab:blue", capsize=4, lw=1.4, label=r"vs NN  mean$\pm$std",
    )
    ax.set_xlabel("Koopman train frames")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    kt._save_fig(fig, save)
    plt.close(fig)


def _load_fit_from_npz(
    path: Path,
    x_src: np.ndarray,
    u_delta: np.ndarray,
) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=False)
    x_mu = np.asarray(blob["x_mean"], dtype=np.float64)
    x_sd = np.asarray(blob["x_std"], dtype=np.float64)
    u_mu = np.asarray(blob["u_mean"], dtype=np.float64)
    u_sd = np.asarray(blob["u_std"], dtype=np.float64)
    koop = {
        "A": np.asarray(blob["A"], dtype=np.float64),
        "B": np.asarray(blob["B"], dtype=np.float64),
        "C": np.asarray(blob["C"], dtype=np.float64),
        "delay": int(np.asarray(blob["delay"]).reshape(-1)[0]),
        "eig": np.linalg.eigvals(np.asarray(blob["A"], dtype=np.float64)),
        "x_mean": x_mu,
        "x_std": x_sd,
        "u_mean": u_mu,
        "u_std": u_sd,
    }
    return {
        "koop": koop,
        "xn": (x_src - x_mu) / x_sd,
        "un": (u_delta - u_mu) / u_sd,
        "x_mu": x_mu,
        "x_sd": x_sd,
        "one_rmse": float("nan"),
        "one_rmse_norm": float("nan"),
        "n_train_pairs": -1,
        "n_val_pairs": -1,
    }


def main() -> None:
    # =====================================================================
    # Same Koopman knobs as Koopman_training_embedding_shape.py
    # Only n_train (TRAIN_SIZES) changes between models.
    # =====================================================================
    train_sizes = list(TRAIN_SIZES)
    n_traj = 100
    skip_existing = True
    snapshot_seed = 0

    # Same locked NN as Koopman_training_embedding_shape.py (INPUT from config.json)
    nn_model_rel = (
        "LSTM/lstm_L2_H256_seq100_stride50_drop1e-1_lr1e-3_sr400_valtip_rmse_oute+xy_lossglocal"
    )
    # nn_model_rel = (
    #     "MLP/mlp_L2_W256_drop0_lr1e-3_actrelu_sr400_valtip_rmse_oute+xy_lossglocal"
    # )

    data_dir = kt.ICRA
    csv_name_contains = "data_20260821_191427"
    max_csvs = None
    max_frames_per_csv = None

    delay = 10
    seed = 42
    koopman_x_mode = "full"
    train_ratio = 0.95
    split_mode = "per_trial"
    skip_start_freeze = True
    delta_u_handle = 1
    u_norm = 1
    x_norm = 1
    mode_reg = 0
    ridge_alpha = 1e-2
    ident_tol = 1e-12
    warmup = "auto"

    num_predict = 4000

    # =====================================================================
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    snap_rng = np.random.default_rng(int(snapshot_seed))
    csvs = kt._list_csvs(Path(data_dir), name_contains=str(csv_name_contains), max_csvs=max_csvs)
    nn_rel, nn_path = kt.resolve_nn_run(nn_model_rel)
    print(f"\n########  Koopman data-size convergence  {nn_rel}  ########", flush=True)
    run_data_convergence(
        nn_path,
        nn_rel=nn_rel,
        csvs=csvs,
        max_frames_per_csv=max_frames_per_csv,
        train_sizes=train_sizes,
        n_traj=n_traj,
        skip_existing=skip_existing,
        delay=delay,
        seed=seed,
        koopman_x_mode=koopman_x_mode,
        train_ratio=train_ratio,
        split_mode=split_mode,
        skip_start_freeze=skip_start_freeze,
        delta_u_handle=delta_u_handle,
        u_norm=u_norm,
        x_norm=x_norm,
        mode_reg=mode_reg,
        ridge_alpha=ridge_alpha,
        ident_tol=ident_tol,
        warmup=warmup,
        num_predict=num_predict,
        snap_rng=snap_rng,
    )


def run_data_convergence(
    nn_dir: Path,
    *,
    nn_rel: str = "",
    csvs: list[Path],
    max_frames_per_csv: int | None,
    train_sizes: list[int],
    n_traj: int,
    skip_existing: bool,
    delay: int,
    seed: int,
    koopman_x_mode: str,
    train_ratio: float,
    split_mode: str,
    skip_start_freeze: bool,
    delta_u_handle: int,
    u_norm: int,
    x_norm: int,
    mode_reg: int,
    ridge_alpha: float,
    ident_tol: float,
    warmup: str | int,
    num_predict: int,
    snap_rng: np.random.Generator,
) -> None:
    nn_dir = kt._resolve_nn_dir(Path(nn_dir))
    if not nn_rel:
        nn_rel, _ = kt.resolve_nn_run(nn_dir)
    cfg = json.loads((nn_dir / "config.json").read_text(encoding="utf-8"))
    ckpt = kt._resolve_ckpt(nn_dir)
    arch = str(cfg.get("arch", "MLP")).strip().lower()
    option = str(cfg.get("shape_option", "e"))
    plane = str(cfg.get("plane", "xz"))
    tip_rb = int(
        (cfg.get("data_meta") or {}).get("tip_rb")
        or (cfg.get("label_meta") or {}).get("tip_rb")
        or 5
    )
    if str(warmup).strip().lower() == "auto":
        warm = int(cfg["seq_len"]) if arch in ("lstm", "tcn") and "seq_len" in cfg else 0
    else:
        warm = int(warmup)
    ident = kt._parse_mode_reg(mode_reg)
    n_imu_calib = int(
        (cfg.get("data_meta") or {}).get("n_imu_calib", kt.N_IMU_CALIB_DEFAULT)
    )
    n_skip_start = max(warm, n_imu_calib if skip_start_freeze else 0)
    native_hz = float(cfg.get("sample_rate_hz", 400.0))
    dly = int(delay)
    n_pred = int(num_predict)

    out_root = kt.KOOPMAN_MODEL_DIR / nn_rel / "koopman_data_convergence"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("Koopman data-size convergence  (NN reconstructed ŷ → A, B)")
    print(f"NN          : {nn_rel}")
    print(f"             folder={nn_dir}")
    print(f"             arch={arch}  option={option}/{plane}  tip=RB{tip_rb}  "
          f"trq={cfg.get('use_trq')} imu={cfg.get('imu')} cmd={cfg.get('cmd')}  "
          f"loss={cfg.get('loss_id')}")
    print(f"train sizes : {train_sizes}")
    print(f"val trajs   : {n_traj} random  horizon={n_pred} @ {native_hz:g} Hz")
    print(f"ident       : {kt._ident_name(ident)}  delay={dly}  seed={seed}")
    print(f"data        : {len(csvs)} csv(s)")
    print(f"out         : {out_root}")
    print("=" * 72, flush=True)

    data = kt.load_robot_trials(csvs, cfg, max_frames_per_csv=max_frames_per_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nn = kt.build_nn(cfg, device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model_nn.load_state_dict(state)
    model_nn.eval()
    print(f"[nn] loaded {ckpt}  device={device}", flush=True)

    y_hat = kt.nn_reconstruct_y(model_nn, cfg, data, device)
    x_nn, x_info = kt._slice_x(y_hat, cfg, koopman_x_mode)
    x_gt, _ = kt._slice_x(np.asarray(data["Y_gt"], dtype=np.float64), cfg, koopman_x_mode)
    names = list(x_info["feature_names"])
    print(f"[nn] ŷ → x  dim={x_nn.shape[1]}  mode={x_info['x_mode']}", flush=True)

    keep = kt._warmup_per_trial(data["trial_id"], n_skip_start)
    x_nn, x_gt, u_abs = x_nn[keep], x_gt[keep], data["U_abs"][keep]
    trial = data["trial_id"][keep]
    x_fit, x_gt_fit, u_delta, trial_fit, train_row = kt._delta_and_split(
        x_nn, x_gt, u_abs, trial,
        delta_u_handle=int(delta_u_handle),
        train_ratio=train_ratio, split_mode=split_mode,
    )
    val_row = ~train_row
    n_train_avail = int(train_row.sum())
    print(
        f"[split] available train={n_train_avail}  val={int(val_row.sum())}  "
        f"(after freeze skip={n_skip_start})",
        flush=True,
    )

    sizes: list[int] = []
    for n in train_sizes:
        if n > n_train_avail:
            print(f"[conv] requested n={n} exceeds available train={n_train_avail} — skip", flush=True)
            continue
        sizes.append(int(n))
    if any(n > n_train_avail for n in train_sizes) and n_train_avail not in sizes:
        print(f"[conv] add full-train point n={n_train_avail}", flush=True)
        sizes.append(int(n_train_avail))
        sizes.sort()
    if not sizes:
        raise SystemExit("no requested train size fits the available train split")

    cand = _valid_k0s(trial_fit, train_row, delay=dly, horizon=n_pred)
    if cand.size < 1:
        raise RuntimeError("no contiguous validation windows (try a smaller num_predict)")
    k0s = _pick_k0s(cand, n_traj, snap_rng)
    k0_show = int(k0s[0])
    print(
        f"[val] {len(k0s)} traj starts (wanted {n_traj}, candidates={cand.size})  "
        f"showcase k0={k0_show}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    shadow_state_gt: dict[str, np.ndarray] = {}
    shadow_state_nn: dict[str, np.ndarray] = {}
    shadow_plane_gt: dict[str, np.ndarray] = {}
    shadow_plane_nn: dict[str, np.ndarray] = {}
    shadow_tip_gt: dict[str, np.ndarray] = {}
    shadow_tip_nn: dict[str, np.ndarray] = {}
    koop_curves: list[tuple[str, np.ndarray]] = []
    nn_show = None
    gt_show = None

    for n in sizes:
        tag = _ntr_tag(n)
        label = f"Koopman {tag}"
        run_dir = out_root / f"ntr{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        npz_path = run_dir / "koopman_model.npz"
        train_cap, n_used = _cap_train_row(train_row, n)
        u_mu, u_sd = kt._u_norm_params(
            u_delta, train_cap, u_norm=int(u_norm), delta_u_handle=int(delta_u_handle),
        )

        if skip_existing and npz_path.is_file():
            print(f"[conv] skip ident n={n_used}  existing {npz_path}", flush=True)
            fit = _load_fit_from_npz(npz_path, x_fit, u_delta)
        else:
            print(f"\n[conv] IDENT n_train={n_used}  ntr{tag}", flush=True)
            fit = kt.identify_on_state(
                x_fit, u_delta, trial_fit, train_cap,
                delay=dly, x_norm=int(x_norm), u_mu=u_mu, u_sd=u_sd,
                mode_reg=ident, alpha=float(ridge_alpha), tol=float(ident_tol),
                tag=f"NN ŷ  ntr{tag}",
                val_row=val_row,
            )
            koop_save = fit["koop"]
            np.savez_compressed(
                npz_path,
                A=koop_save["A"], B=koop_save["B"], C=koop_save["C"],
                x_mean=fit["x_mu"], x_std=fit["x_sd"], u_mean=u_mu, u_std=u_sd,
                delay=np.array([dly], dtype=np.int64),
                seed=np.array([int(seed)], dtype=np.int64),
                n_train=np.array([n_used], dtype=np.int64),
                delta_u_handle=np.array([int(delta_u_handle)], dtype=np.int64),
                feature_names=np.array(names),
                cols=np.array(x_info["cols"], dtype=np.int64),
                source=np.array(["nn"]),
                sample_rate_hz=np.array([native_hz], dtype=np.float64),
            )
            print(f"[conv] saved {npz_path}", flush=True)

        koop = fit["koop"]
        x_mu, x_sd = fit["x_mu"], fit["x_sd"]
        xn, un = fit["xn"], fit["un"]

        pred_show, nn_show, gt_show = _rollout_one(
            koop, xn, un, x_gt_fit, x_mu, x_sd, k0_show, n_pred
        )
        show_errs = _rollout_errors(
            pred_show, nn_show, gt_show, names, plane=plane, tip_rb=tip_rb,
        )

        kt.plot_A_eigs(
            run_dir / "fig_A_eigs.png",
            koop["eig"], koop["A"],
            title=f"A eigenvalues  ntr{tag} (trained on NN ŷ)",
        )
        plt.close("all")
        plot_horizon_rmse(
            run_dir / "fig_compare_rmse_horizon_state.png",
            rmse_vs_gt=show_errs["state_vs_gt"],
            rmse_vs_nn=show_errs["state_vs_nn"],
            sample_rate_hz=native_hz,
            title=f"Open-loop Koopman state RMSE  ntr{tag}  (k0={k0_show})",
            ylabel="state RMSE [mm]",
        )
        plot_horizon_rmse(
            run_dir / "fig_compare_rmse_horizon_plane.png",
            rmse_vs_gt=show_errs["plane_vs_gt"],
            rmse_vs_nn=show_errs["plane_vs_nn"],
            sample_rate_hz=native_hz,
            title=f"Open-loop Koopman base-plane ({plane}) RMSE  ntr{tag}  (k0={k0_show})",
            ylabel=f"mean RB {plane} distance [mm]",
        )
        if "tip_vs_gt" in show_errs:
            plot_horizon_rmse(
                run_dir / "fig_compare_rmse_horizon_tip.png",
                rmse_vs_gt=show_errs["tip_vs_gt"],
                rmse_vs_nn=show_errs["tip_vs_nn"],
                sample_rate_hz=native_hz,
                title=f"Tip RB{tip_rb} {plane} error  ntr{tag}  (k0={k0_show})",
                ylabel=f"tip {plane} distance [mm]",
            )
        shape_stats = kt.plot_shape_rollout(
            run_dir / "fig_shape_rollout_plane.png",
            gt_x=gt_show,
            nn_x=nn_show,
            kp_x=pred_show,
            names=names,
            option=option,
            plane=plane,
            x_mode=x_info["x_mode"],
            k0=k0_show,
            train_tag=f"NN {tag}",
        )
        plt.close("all")

        print(f"[eval] rolling {len(k0s)} trajs for ntr{tag} …", flush=True)
        err_trajs = rollout_rmse_trajs(
            koop, xn, un, x_gt_fit, x_mu, x_sd, k0s, n_pred,
            names=names, plane=plane, tip_rb=tip_rb,
        )
        plot_rmse_shadow(
            run_dir / "fig_rollout_rmse_100traj_state.png",
            series={label: err_trajs["state_vs_gt"]},
            sample_rate_hz=native_hz,
            title=f"ntr{tag}: {len(k0s)} random val trajs  (state RMSE vs GT)",
            ylabel="state RMSE vs GT [mm]",
        )
        plot_rmse_shadow(
            run_dir / "fig_rollout_rmse_100traj_plane.png",
            series={label: err_trajs["plane_vs_gt"]},
            sample_rate_hz=native_hz,
            title=f"ntr{tag}: {len(k0s)} random val trajs  (mean RB {plane} vs GT)",
            ylabel=f"mean RB {plane} distance vs GT [mm]",
        )
        shadow_state_gt[label] = err_trajs["state_vs_gt"]
        shadow_state_nn[label] = err_trajs["state_vs_nn"]
        shadow_plane_gt[label] = err_trajs["plane_vs_gt"]
        shadow_plane_nn[label] = err_trajs["plane_vs_nn"]
        if "tip_vs_gt" in err_trajs:
            shadow_tip_gt[label] = err_trajs["tip_vs_gt"]
            shadow_tip_nn[label] = err_trajs["tip_vs_nn"]
        koop_curves.append((label, pred_show))

        def _mean_std_mm(arr: np.ndarray) -> tuple[float, float]:
            per_traj = np.mean(arr, axis=1)
            return float(1e3 * np.mean(per_traj)), float(1e3 * np.std(per_traj))

        st_gt_m, st_gt_s = _mean_std_mm(err_trajs["state_vs_gt"])
        st_nn_m, st_nn_s = _mean_std_mm(err_trajs["state_vs_nn"])
        pl_gt_m, pl_gt_s = _mean_std_mm(err_trajs["plane_vs_gt"])
        pl_nn_m, pl_nn_s = _mean_std_mm(err_trajs["plane_vs_nn"])

        rec = {
            "n_train_requested": int(n),
            "n_train_actual": int(n_used),
            "n_train_pairs": int(fit["n_train_pairs"]),
            "n_val_pairs": int(fit["n_val_pairs"]),
            "one_step_rmse": float(fit["one_rmse"]),
            "k0_show": int(k0_show),
            "n_traj": int(len(k0s)),
            "plane": plane,
            "tip_rb": int(tip_rb),
            "rollout_state_rmse_vs_gt_mean_mm": st_gt_m,
            "rollout_state_rmse_vs_gt_std_mm": st_gt_s,
            "rollout_state_rmse_vs_nn_mean_mm": st_nn_m,
            "rollout_state_rmse_vs_nn_std_mm": st_nn_s,
            "rollout_plane_rmse_vs_gt_mean_mm": pl_gt_m,
            "rollout_plane_rmse_vs_gt_std_mm": pl_gt_s,
            "rollout_plane_rmse_vs_nn_mean_mm": pl_nn_m,
            "rollout_plane_rmse_vs_nn_std_mm": pl_nn_s,
            # Legacy keys (state RMSE) for older parsers
            "rollout_rmse_vs_gt_mean_mm": st_gt_m,
            "rollout_rmse_vs_gt_std_mm": st_gt_s,
            "rollout_rmse_vs_nn_mean_mm": st_nn_m,
            "rollout_rmse_vs_nn_std_mm": st_nn_s,
            "shape_mean_rmse_vs_gt_m": shape_stats["mean_rmse_vs_gt_m"],
            "shape_mean_rmse_vs_nn_m": shape_stats["mean_rmse_vs_nn_m"],
            "npz": str(npz_path),
        }
        if "tip_vs_gt" in err_trajs:
            tip_gt_m, tip_gt_s = _mean_std_mm(err_trajs["tip_vs_gt"])
            tip_nn_m, tip_nn_s = _mean_std_mm(err_trajs["tip_vs_nn"])
            rec.update(
                {
                    "rollout_tip_rmse_vs_gt_mean_mm": tip_gt_m,
                    "rollout_tip_rmse_vs_gt_std_mm": tip_gt_s,
                    "rollout_tip_rmse_vs_nn_mean_mm": tip_nn_m,
                    "rollout_tip_rmse_vs_nn_std_mm": tip_nn_s,
                }
            )
        rows.append(rec)
        (run_dir / "results.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        tip_line = ""
        if "rollout_tip_rmse_vs_gt_mean_mm" in rec:
            tip_line = (
                f"  tip {plane} vs GT={rec['rollout_tip_rmse_vs_gt_mean_mm']:.3f}±"
                f"{rec['rollout_tip_rmse_vs_gt_std_mm']:.3f} mm"
            )
        print(
            f"[eval] ntr{tag}  state vs GT={st_gt_m:.3f}±{st_gt_s:.3f} mm  "
            f"plane {plane} vs GT={pl_gt_m:.3f}±{pl_gt_s:.3f} mm"
            f"{tip_line}",
            flush=True,
        )

    plot_rmse_shadow(
        out_root / "fig_rollout_rmse_100traj_state_vs_gt.png",
        series=shadow_state_gt,
        sample_rate_hz=native_hz,
        title=f"{n_traj} random val trajs  Koopman state vs OptiTrack  (mean ± std)",
        ylabel="state RMSE vs GT [mm]",
    )
    plot_rmse_shadow(
        out_root / "fig_rollout_rmse_100traj_state_vs_nn.png",
        series=shadow_state_nn,
        sample_rate_hz=native_hz,
        title=f"{n_traj} random val trajs  Koopman state vs NN ŷ  (mean ± std)",
        ylabel="state RMSE vs NN [mm]",
    )
    plot_rmse_shadow(
        out_root / "fig_rollout_rmse_100traj_plane_vs_gt.png",
        series=shadow_plane_gt,
        sample_rate_hz=native_hz,
        title=f"{n_traj} random val trajs  mean RB {plane} vs OptiTrack  (mean ± std)",
        ylabel=f"mean RB {plane} distance vs GT [mm]",
    )
    plot_rmse_shadow(
        out_root / "fig_rollout_rmse_100traj_plane_vs_nn.png",
        series=shadow_plane_nn,
        sample_rate_hz=native_hz,
        title=f"{n_traj} random val trajs  mean RB {plane} vs NN ŷ  (mean ± std)",
        ylabel=f"mean RB {plane} distance vs NN [mm]",
    )
    if shadow_tip_gt:
        plot_rmse_shadow(
            out_root / "fig_rollout_rmse_100traj_tip_vs_gt.png",
            series=shadow_tip_gt,
            sample_rate_hz=native_hz,
            title=f"{n_traj} random val trajs  tip RB{tip_rb} {plane} vs OptiTrack",
            ylabel=f"tip {plane} distance vs GT [mm]",
        )
    if nn_show is not None and gt_show is not None and koop_curves:
        plot_shape_rollout_compare(
            out_root / "fig_shape_rollout_plane.png",
            gt_x=gt_show,
            nn_x=nn_show,
            koop_curves=koop_curves,
            names=names,
            option=option,
            plane=plane,
            x_mode=x_info["x_mode"],
            k0=k0_show,
        )
    n_arr = np.array([r["n_train_actual"] for r in rows], dtype=np.float64)
    plot_rmse_vs_ntrain(
        out_root / "fig_mean_state_rmse_vs_ntrain.png",
        n_arr,
        np.array([r["rollout_state_rmse_vs_gt_mean_mm"] * 1e-3 for r in rows]),
        np.array([r["rollout_state_rmse_vs_gt_std_mm"] * 1e-3 for r in rows]),
        np.array([r["rollout_state_rmse_vs_nn_mean_mm"] * 1e-3 for r in rows]),
        np.array([r["rollout_state_rmse_vs_nn_std_mm"] * 1e-3 for r in rows]),
    )
    plot_rmse_vs_ntrain(
        out_root / "fig_mean_plane_rmse_vs_ntrain.png",
        n_arr,
        np.array([r["rollout_plane_rmse_vs_gt_mean_mm"] * 1e-3 for r in rows]),
        np.array([r["rollout_plane_rmse_vs_gt_std_mm"] * 1e-3 for r in rows]),
        np.array([r["rollout_plane_rmse_vs_nn_mean_mm"] * 1e-3 for r in rows]),
        np.array([r["rollout_plane_rmse_vs_nn_std_mm"] * 1e-3 for r in rows]),
        title=f"Koopman base-plane ({plane}) RMSE vs training data  (mean over {n_traj} val trajs)",
        ylabel=f"mean RB {plane} distance [mm]",
    )

    summary = {
        "nn_rel": nn_rel,
        "nn_run_id": nn_dir.name,
        "nn_ckpt": str(ckpt),
        "plane": plane,
        "tip_rb": int(tip_rb),
        "train_sizes_requested": list(train_sizes),
        "train_sizes_run": sizes,
        "n_train_available": n_train_avail,
        "n_val": int(val_row.sum()),
        "n_traj": int(len(k0s)),
        "k0s": k0s,
        "k0_show": int(k0_show),
        "num_predict": n_pred,
        "sample_rate_hz": native_hz,
        "delay": dly,
        "koopman_x_mode": x_info["x_mode"],
        "csv": data["csv_paths"],
        "models": rows,
        "out_dir": str(out_root),
    }
    out_json = out_root / "koopman_data_convergence.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[conv] wrote {out_json}", flush=True)
    print(f"[conv] figures -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
