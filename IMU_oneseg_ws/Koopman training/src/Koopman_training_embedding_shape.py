#!/usr/bin/env python
"""
Koopman on NN-reconstructed shape (one-seg MLP / LSTM / TCN).

Sensor → NN INPUT (must match the chosen NN ``config.json``) → frozen NN → ŷ
→ Koopman state x ⊂ ŷ → delay embedding + EDMD.

IMU / OT packing is the **same** pipeline as NN training
(``common.data_pipeline.process_csv_rich``)::

  IMU_raw → R_imu_in_rb (RB-align) → under IMU3_aligned  (IMU3 ≡ I)
  OT_raw  → under RB1   (shape labels)
  enc     → home-relative

Locked NN runs (edit ``nn_model_rel`` in ``main()``)::

  LSTM/…  INPUT enc_trq_imu12 (14-D)   OUTPUT e/xz (16-D)
  MLP/…   INPUT enc_imu12_u   (16-D)   OUTPUT e/xz (16-D)

Artifacts::

  Koopman training/Model/<nn_model_rel>/koop_d…/

    python Koopman_training_embedding_shape.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
KOOPMAN_DIR = ROOT.parent
ICRA = KOOPMAN_DIR.parent
NN_SRC = ICRA / "NN_training" / "Training" / "src"
NN_MODELS_DIR = ICRA / "NN_training" / "Training" / "models"
KOOPMAN_MODEL_DIR = KOOPMAN_DIR / "Model"
if str(NN_SRC) not in sys.path:
    sys.path.insert(0, str(NN_SRC))

from common import pathsetup  # noqa: E402

pathsetup.ensure()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import Lasso  # noqa: E402

from LSTM.LSTM_training import PoseRNN  # noqa: E402
from MLP.MLP_training import PoseMLP  # noqa: E402
from TCN.TCN_training import PoseTCN  # noqa: E402
from common.data_pipeline import (  # noqa: E402
    N_IMU_CALIB_DEFAULT,
    process_csv_rich,
)
from common.input_state import build_input_X, input_key_from_flags  # noqa: E402
from common.shape_state import pack_shape_state, project_packed_quats_svd  # noqa: E402
from common.train_loop import _predict_split  # noqa: E402


def _list_csvs(data_dir: Path, *, name_contains: str, max_csvs: int | None) -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    files = sorted(root.rglob("data_*.csv"))
    if not files:
        files = sorted(p for p in root.rglob("*.csv") if "log" not in p.name.lower())
    needle = str(name_contains or "").strip()
    if needle:
        files = [p for p in files if needle.lower() in str(p).lower()]
    if max_csvs is not None:
        files = files[: int(max_csvs)]
    if not files:
        raise FileNotFoundError(f"no CSV under {root} (filter={needle!r})")
    return files


def _normalize_nn_rel(nn_model_rel: str | Path) -> str:
    """``LSTM/run_id`` or absolute path → posix relative key under Training/models when possible."""
    raw = str(nn_model_rel).strip().replace("\\", "/")
    p = Path(raw).expanduser()
    if p.is_absolute():
        try:
            return p.resolve().relative_to(NN_MODELS_DIR.resolve()).as_posix()
        except ValueError:
            return p.resolve().as_posix()
    return Path(raw).as_posix().lstrip("./")


def _resolve_nn_dir(nn_model: str | Path) -> Path:
    """Resolve a trained NN folder (must contain ``config.json`` + ``model.pt``).

    Accepts:
      - relative under ``NN_training/Training/models`` (e.g. ``LSTM/lstm_L2_…``)
      - absolute path to that run folder
    """
    rel = _normalize_nn_rel(nn_model)
    candidates: list[Path] = []
    p = Path(rel).expanduser()
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(NN_MODELS_DIR / rel)
        candidates.append(NN_MODELS_DIR / Path(rel).name)
        candidates.append(Path(rel))
    for cand in candidates:
        cand = cand.resolve()
        if (cand / "config.json").is_file():
            return cand
    looked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"no config.json for NN model {nn_model!r}\n  looked:\n  {looked}")


def _resolve_ckpt(nn_dir: Path) -> Path:
    ckpt = Path(nn_dir) / "model.pt"
    if ckpt.is_file():
        return ckpt
    raise FileNotFoundError(
        f"missing model.pt next to config.json:\n  {ckpt}\n"
        "Train the NN first (or copy checkpoint into that run folder)."
    )


def resolve_nn_run(nn_model_rel: str | Path) -> tuple[str, Path]:
    """Return ``(relative_key, absolute_nn_dir)`` for save layout + loading."""
    nn_dir = _resolve_nn_dir(nn_model_rel)
    rel = _normalize_nn_rel(nn_model_rel)
    if Path(rel).is_absolute() or not rel:
        try:
            rel = nn_dir.resolve().relative_to(NN_MODELS_DIR.resolve()).as_posix()
        except ValueError:
            rel = f"{nn_dir.parent.name}/{nn_dir.name}"
    return rel, nn_dir


def build_nn(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    arch = str(cfg.get("arch", "MLP")).strip().lower()
    in_dim, out_dim = int(cfg["in_dim"]), int(cfg["out_dim"])
    if arch == "mlp":
        return PoseMLP(
            in_dim,
            out_dim,
            layers=int(cfg.get("layers", 2)),
            width=int(cfg.get("width", 256)),
            dropout=float(cfg.get("dropout", 0.0)),
            activation=str(cfg.get("activation", "relu")),
        ).to(device)
    if arch == "lstm":
        return PoseRNN(
            in_dim,
            out_dim,
            layers=int(cfg.get("layers", 2)),
            hidden=int(cfg.get("hidden", 256)),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device)
    if arch == "tcn":
        dil = cfg.get("dilations", (1, 2, 4, 8, 16))
        if isinstance(dil, str):
            dil = tuple(int(x) for x in dil.strip("()[]").split(",") if x.strip())
        return PoseTCN(
            in_dim,
            out_dim,
            channels=int(cfg.get("channels", 128)),
            dilations=tuple(int(d) for d in dil),
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "relu")),
        ).to(device)
    raise ValueError(f"unsupported arch {arch!r}")


def _slice_x(y_hat: np.ndarray, cfg: dict[str, Any], mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Select Koopman x from the full campaign ŷ (future embedding modes live here)."""
    mode = str(mode).strip().lower()
    names = list((cfg.get("label_meta") or {}).get("feature_names") or [])
    option = str(cfg.get("shape_option", "g")).strip().lower()
    dual = bool((cfg.get("label_meta") or {}).get("dual", option in ("e", "f", "g")))
    n_out = int(y_hat.shape[1])
    if not names:
        names = [f"y{i}" for i in range(n_out)]

    if mode in ("full", "all", ""):
        cols = list(range(n_out))
        return y_hat, {"x_mode": "full", "feature_names": names, "cols": cols, "out_dim": n_out}

    if mode == "global":
        if not dual:
            print("[warn] x_mode=global but pack is not dual e/f/g; using full ŷ")
            cols = list(range(n_out))
            return y_hat, {"x_mode": "full", "feature_names": names, "cols": cols, "out_dim": n_out}
        half = n_out // 2
        cols = list(range(half))
        return y_hat[:, cols], {
            "x_mode": "global",
            "feature_names": names[:half],
            "cols": cols,
            "out_dim": n_out,
        }

    if mode == "tip":
        tip_rb = int(
            (cfg.get("data_meta") or {}).get("tip_rb")
            or (cfg.get("label_meta") or {}).get("tip_rb")
            or 5
        )
        tip_tok = f"RB{tip_rb}"
        keys = [
            i for i, n in enumerate(names)
            if tip_tok in str(n) and ("_p" in str(n) or f"pF_{tip_tok}" in str(n))
            and "Chain" not in str(n) and "pB" not in str(n)
        ]
        if not keys:
            keys = [i for i, n in enumerate(names) if f"pF_{tip_tok}" in str(n)]
        if not keys:
            raise ValueError(f"cannot find {tip_tok} global position channels for x_mode=tip")
        return y_hat[:, keys], {
            "x_mode": "tip",
            "feature_names": [names[i] for i in keys],
            "cols": keys,
            "out_dim": n_out,
            "tip_rb": tip_rb,
        }
    raise ValueError(f"unknown koopman_x_mode {mode!r} (use full | global | tip)")


def _nn_io_flags(cfg: dict[str, Any]) -> dict[str, Any]:
    """INPUT/OUTPUT flags from NN ``config.json`` (defaults = locked one-seg)."""
    data_meta = cfg.get("data_meta") or {}
    use_trq = bool(cfg.get("use_trq", True))
    imu = str(cfg.get("imu", "imu12")).strip().lower()
    cmd = str(cfg.get("cmd", "none")).strip().lower()
    option = str(cfg.get("shape_option", "e")).strip().lower()
    plane = str(cfg.get("plane", "xz")).strip().lower()
    n_imu_calib = int(data_meta.get("n_imu_calib", N_IMU_CALIB_DEFAULT))
    imu_on_rb = {
        int(k): int(v)
        for k, v in data_meta.get("imu_on_rb", {1: 5, 2: 3, 3: 1}).items()
    }
    key = input_key_from_flags(use_trq=use_trq, imu=imu, cmd=cmd)
    return {
        "use_trq": use_trq,
        "imu": imu,
        "cmd": cmd,
        "key": key,
        "option": option,
        "plane": plane,
        "n_imu_calib": n_imu_calib,
        "imu_on_rb": imu_on_rb,
        "use_ideal_imu": bool(data_meta.get("use_ideal_imu", True)),
        "imu_frame": str(data_meta.get("imu_frame", "IMU3_aligned")),
    }


def load_robot_trials(
    csv_paths: list[Path],
    cfg: dict[str, Any],
    *,
    max_frames_per_csv: int | None,
    start_row: int = 0,
) -> dict[str, Any]:
    """CSV → rich (RB-align → IMU3_aligned; OT@RB1) → NN INPUT X + OT label Y.

    Packing flags come from the NN ``config.json`` so LSTM (14-D) and MLP (16-D)
    each get the exact feature layout they were trained with.
    """
    flags = _nn_io_flags(cfg)
    use_trq = flags["use_trq"]
    imu = flags["imu"]
    cmd = flags["cmd"]
    option = flags["option"]
    plane = flags["plane"]
    expected_key = flags["key"]
    print(
        f"[data] NN INPUT pack={expected_key}  "
        f"trq={use_trq} imu={imu} cmd={cmd}  in_dim={cfg.get('in_dim')}\n"
        f"[data] NN OUTPUT option={option}/{plane}  out_dim={cfg.get('out_dim')}\n"
        f"[data] IMU path: raw → RB-align (n_imu_calib={flags['n_imu_calib']}) "
        f"→ {flags['imu_frame']}  (imu12 drops IMU3≡I)  |  OT labels @ RB1",
        flush=True,
    )

    Xs, Ygts, Us, tids, poss = [], [], [], [], []
    tid = 0
    names: list[str] = []
    in_meta: dict[str, Any] = {}
    y_meta: dict[str, Any] = {}
    n_u = 0
    for csv_path in csv_paths:
        rich = process_csv_rich(
            csv_path,
            use_ideal_imu=flags["use_ideal_imu"],
            n_imu_calib=flags["n_imu_calib"],
            imu_on_rb=flags["imu_on_rb"],
            max_frames=max_frames_per_csv,
            start_row=int(start_row),
        )
        X, names, in_meta = build_input_X(rich, use_trq=use_trq, imu=imu, cmd=cmd)
        Y_gt, y_meta = pack_shape_state(rich["pos"], rich["quat"], option, plane)
        # Koopman control u: home-relative encoders (same as rich["real_enc"]).
        u_abs = np.asarray(rich["real_enc"], dtype=np.float64)
        if u_abs.ndim != 2 or u_abs.shape[1] < 1:
            raise ValueError(f"{csv_path.name}: real_enc shape {u_abs.shape}, expected (N, n_motors)")
        got_key = str(in_meta.get("key", ""))
        if got_key and got_key != expected_key:
            raise ValueError(
                f"{csv_path.name}: packed key {got_key!r} != NN key {expected_key!r}"
            )
        if int(X.shape[1]) != int(cfg["in_dim"]):
            raise ValueError(
                f"{csv_path.name}: packed INPUT width {X.shape[1]} != NN in_dim {cfg['in_dim']}. "
                f"Expected {expected_key} from use_trq/imu/cmd in config.json."
            )
        if int(Y_gt.shape[1]) != int(cfg["out_dim"]):
            raise ValueError(
                f"{csv_path.name}: packed OUTPUT width {Y_gt.shape[1]} != NN out_dim {cfg['out_dim']}. "
                "shape_option/plane in config.json must match pack_shape_state."
            )
        if str(in_meta.get("imu_frame", "")) not in ("", "IMU3_aligned"):
            print(
                f"[data] WARNING: imu_frame={in_meta.get('imu_frame')!r} "
                "(expected IMU3_aligned)",
                flush=True,
            )
        n = int(X.shape[0])
        n_u = int(u_abs.shape[1])
        Xs.append(X.astype(np.float32))
        Ygts.append(Y_gt.astype(np.float32))
        Us.append(u_abs)
        tids.append(np.full(n, tid, dtype=np.int32))
        poss.append(np.asarray(rich["pos"], dtype=np.float64))
        print(
            f"[data] {csv_path.name}: N={n}  Xin={X.shape[1]} ({got_key})  "
            f"Y={Y_gt.shape[1]}  U=enc({n_u})  trial={tid}",
            flush=True,
        )
        tid += 1
        del rich

    return {
        "X": np.concatenate(Xs, axis=0),
        "Y_gt": np.concatenate(Ygts, axis=0),
        "U_abs": np.concatenate(Us, axis=0),
        "trial_id": np.concatenate(tids, axis=0),
        "pos_gt": np.concatenate(poss, axis=0),
        "input_names": names,
        "input_meta": in_meta,
        "label_meta": y_meta,
        "n_trials": tid,
        "n_u": n_u,
        "u_kind": "enc_rel",
        "csv_paths": [str(p) for p in csv_paths],
        "nn_io": flags,
    }


@torch.no_grad()
def nn_reconstruct_y(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    data: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Normalize X with NN train stats (quat ch: mean=0, std=1) → frozen NN → ŷ."""
    arch = str(cfg.get("arch", "MLP")).strip().lower()
    x_mean = np.asarray(cfg["x_mean"], dtype=np.float32).copy()
    x_std = np.asarray(cfg["x_std"], dtype=np.float32).clip(min=1e-6).copy()
    y_mean = np.asarray(cfg["y_mean"], dtype=np.float32)
    y_std = np.asarray(cfg["y_std"], dtype=np.float32).clip(min=1e-6)
    if x_mean.shape[0] != data["X"].shape[1]:
        raise ValueError(f"x_mean dim {x_mean.shape[0]} != packed X {data['X'].shape[1]}")
    # Match train_loop: never z-score IMU quat channels (identity transform).
    qmask = cfg.get("x_quat_channel_mask")
    if qmask is not None:
        qmask = np.asarray(qmask, dtype=bool).reshape(-1)
        if qmask.shape == x_mean.shape:
            x_mean[qmask] = 0.0
            x_std[qmask] = 1.0
    Xn = ((data["X"] - x_mean) / x_std).astype(np.float32)
    n_q = int(qmask.sum()) if qmask is not None and qmask.shape == x_mean.shape else 0
    print(
        f"[nn] infer {arch} on N={len(Xn)}  Xin={Xn.shape[1]}  "
        f"(skip z-score on {n_q} quat ch) …",
        flush=True,
    )
    yhat = _predict_split(model, Xn, data["trial_id"], device, arch)
    y = yhat * y_std + y_mean
    y = project_packed_quats_svd(y, str(cfg.get("shape_option", "g")), str(cfg.get("plane", "xyz")))
    if int(y.shape[1]) != int(cfg["out_dim"]):
        raise ValueError(f"NN out {y.shape[1]} != config out_dim {cfg['out_dim']}")
    return np.asarray(y, dtype=np.float64)


def gen_delay_traj(X: np.ndarray, U: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray]:
    """z[t] = [x[t], x[t-1], …, x[t-d]]; u aligned at time t. Rows start at t=d."""
    d = int(delay)
    n, nx = X.shape
    n_row = n - d
    if n_row <= 0:
        raise ValueError(f"series length {n} too short for delay={d}")
    Z = np.empty((n_row, nx * (d + 1)), dtype=np.float64)
    for i in range(d + 1):
        Z[:, i * nx : (i + 1) * nx] = X[d - i : n - i]
    return Z, np.asarray(U[d:], dtype=np.float64)


def _invalid_delay_mask(trial_id: np.ndarray, delay: int) -> np.ndarray:
    t = np.asarray(trial_id)
    n_row = t.size - int(delay)
    bad = np.zeros(n_row, dtype=bool)
    for i in range(int(delay) + 1):
        bad |= t[int(delay) - i : t.size - i] != t[int(delay) :]
    return bad


def _warmup_per_trial(trial_id: np.ndarray, warmup: int) -> np.ndarray:
    keep = np.ones(trial_id.shape[0], dtype=bool)
    w = int(max(warmup, 0))
    if w <= 0:
        return keep
    for tid in np.unique(trial_id):
        idx = np.flatnonzero(trial_id == tid)
        keep[idx[: min(w, idx.size)]] = False
    return keep


def _x_norm_params(X: np.ndarray, train_m: np.ndarray, handle: int) -> tuple[np.ndarray, np.ndarray]:
    """0 = off, 1 = X/std, 2 = (X-mean)/std  (same handles as the 5-seg script)."""
    d = X.shape[1]
    if int(handle) == 0:
        return np.zeros(d), np.ones(d)
    sd = X[train_m].std(axis=0)
    sd[sd < 1e-12] = 1.0
    if int(handle) == 1:
        return np.zeros(d), sd
    if int(handle) == 2:
        return X[train_m].mean(axis=0), sd
    raise ValueError(f"x_norm must be 0, 1, or 2, got {handle}")


def time_split_mask(trial_id: np.ndarray, train_ratio: float, *, mode: str) -> np.ndarray:
    """True on training rows of *already freeze-trimmed* series.

    ``per_trial`` keeps each CSV's chronology; ``global`` is a prefix of concat.
    """
    r = float(np.clip(train_ratio, 0.05, 0.99))
    if str(mode).strip().lower() == "global":
        n_tr = max(1, int(round(r * trial_id.size)))
        m = np.zeros(trial_id.shape[0], dtype=bool)
        m[:n_tr] = True
        return m
    m = np.zeros(trial_id.shape[0], dtype=bool)
    for tid in np.unique(trial_id):
        idx = np.flatnonzero(trial_id == tid)
        n_tr = max(1, int(round(r * idx.size)))
        m[idx[:n_tr]] = True
    return m


def _delta_and_split(
    x: np.ndarray,
    x_gt: np.ndarray,
    u_abs: np.ndarray,
    trial: np.ndarray,
    *,
    delta_u_handle: int,
    train_ratio: float,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Δκ (or identity) + 95/5 split. ``x`` and ``x_gt`` must share the time axis."""
    if int(delta_u_handle) == 1:
        x_fit = x[:-1]
        x_gt_fit = x_gt[:-1]
        u_delta = u_abs[1:] - u_abs[:-1]
        trial_fit = trial[:-1]
        train_row = time_split_mask(trial_fit, train_ratio, mode=split_mode)
        cross = trial[1:] != trial[:-1]
        u_delta[cross] = 0.0
        train_row = train_row & ~cross
        return x_fit, x_gt_fit, u_delta, trial_fit, train_row
    train_row = time_split_mask(trial, train_ratio, mode=split_mode)
    return x, x_gt, u_abs.copy(), trial, train_row


def _u_norm_params(
    u_delta: np.ndarray,
    train_row: np.ndarray,
    *,
    u_norm: int,
    delta_u_handle: int,
) -> tuple[np.ndarray, np.ndarray]:
    nu = int(u_delta.shape[1])
    if int(u_norm) == 1:
        u_sd = u_delta[train_row].std(axis=0)
        u_sd[u_sd < 1e-12] = 1.0
        return np.zeros(nu, dtype=np.float64), u_sd
    u_mu = np.zeros(nu, dtype=np.float64)
    u_sd = np.ones(nu, dtype=np.float64)
    if int(delta_u_handle) == 0:
        u_mu = u_delta[train_row].mean(axis=0)
        u_sd = u_delta[train_row].std(axis=0)
        u_sd[u_sd < 1e-12] = 1.0
    return u_mu, u_sd


def _ident_name(mode_reg: int) -> str:
    return {0: "pinv", 1: "lasso", 2: "ridge"}.get(int(mode_reg), f"reg{mode_reg}")


def _parse_mode_reg(mode_reg: int | str) -> int:
    if isinstance(mode_reg, str):
        key = mode_reg.strip().lower()
        table = {
            "0": 0, "pinv": 0, "lstsq": 0, "pseudo": 0, "pseudoinverse": 0,
            "1": 1, "lasso": 1,
            "2": 2, "ridge": 2,
        }
        if key not in table:
            raise ValueError(f"mode_reg {mode_reg!r} not in {sorted(table)}")
        return table[key]
    m = int(mode_reg)
    if m not in (0, 1, 2):
        raise ValueError(f"mode_reg must be 0 (pinv), 1 (lasso), or 2 (ridge), got {mode_reg}")
    return m


def fit_xb(
    Z: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray,
    *,
    mode_reg: int,
    alpha: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify A_x, B_x for Y ≈ A_x Z + B_x U  (Y is next x).

    mode_reg (same as 5-seg): 0 = pseudo-inverse / lstsq, 1 = lasso, 2 = ridge.
    """
    feat = np.hstack([Z, U])
    nz = int(Z.shape[1])
    mode = int(mode_reg)
    if mode == 0:
        f = np.linalg.lstsq(feat, Y, rcond=None)[0].T
    elif mode == 1:
        print(
            f"[train] lasso on N={feat.shape[0]} feat={feat.shape[1]} "
            f"(slow on long recordings) …",
            flush=True,
        )
        clf = Lasso(alpha=float(alpha), fit_intercept=False, tol=float(tol), max_iter=4000)
        clf.fit(feat, Y)
        f = np.atleast_2d(np.asarray(clf.coef_, dtype=np.float64))
    elif mode == 2:
        if float(alpha) <= 0.0:
            f = np.linalg.lstsq(feat, Y, rcond=None)[0].T
        else:
            g = feat.T @ feat
            g.flat[:: feat.shape[1] + 1] += float(alpha)
            f = np.linalg.solve(g, feat.T @ Y).T
    else:
        raise ValueError(f"mode_reg must be 0 (pinv), 1 (lasso), or 2 (ridge), got {mode_reg}")
    f = np.asarray(f, dtype=np.float64)
    return f[:, :nz], f[:, nz:]


def lift_ab_with_shift(a_x: np.ndarray, b_x: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square A so z+ = A z + B u with z = [x, x_prev, …] (matches 5-seg unconstrained EDMD)."""
    nx, nz = int(a_x.shape[0]), int(a_x.shape[1])
    nu = int(b_x.shape[1])
    if nz != nx * (int(delay) + 1):
        raise ValueError(f"A_x shape {a_x.shape} inconsistent with delay={delay}")
    a = np.zeros((nz, nz), dtype=np.float64)
    b = np.zeros((nz, nu), dtype=np.float64)
    a[:nx, :] = a_x
    a[nx:, : nz - nx] = np.eye(nz - nx)
    b[:nx, :] = b_x
    c = np.zeros((nx, nz), dtype=np.float64)
    c[:, :nx] = np.eye(nx)
    return a, b, c


def train_koopman(
    Z_pair: np.ndarray,
    U_pair: np.ndarray,
    Y_next: np.ndarray,
    *,
    delay: int,
    mode_reg: int,
    alpha: float,
    tol: float,
) -> dict[str, Any]:
    name = _ident_name(mode_reg)
    print(f"[train] identifying A,B with {name}  alpha={alpha:g}", flush=True)
    a_x, b_x = fit_xb(Z_pair, U_pair, Y_next, mode_reg=mode_reg, alpha=alpha, tol=tol)
    a, b, c = lift_ab_with_shift(a_x, b_x, delay)
    pred = Z_pair @ a_x.T + U_pair @ b_x.T
    err = np.linalg.norm(pred - Y_next, axis=1)
    print(
        f"[train] pair samples={Z_pair.shape[0]}  A{a.shape} B{b.shape}  "
        f"RMSE={float(np.sqrt(np.mean(err ** 2))):.6f}  MAE={float(np.mean(err)):.6f}",
        flush=True,
    )
    ev = np.linalg.eigvals(a)
    n_unst = int(np.sum(np.abs(ev) > 1.0 + 1e-6))
    if n_unst:
        print(
            f"[warn] {n_unst} eigenvalues outside unit circle "
            f"(max|λ|={float(np.max(np.abs(ev))):.4f}) — try mode_reg=2 (ridge) and/or larger alpha",
            flush=True,
        )
    return {"A": a, "B": b, "C": c, "delay": int(delay), "eig": ev, "mode_reg": int(mode_reg)}


def identify_on_state(
    x_src: np.ndarray,
    u_delta: np.ndarray,
    trial_fit: np.ndarray,
    train_row: np.ndarray,
    *,
    delay: int,
    x_norm: int,
    u_mu: np.ndarray,
    u_sd: np.ndarray,
    mode_reg: int,
    alpha: float,
    tol: float,
    tag: str,
    val_row: np.ndarray | None = None,
) -> dict[str, Any]:
    """EDMD on one aligned state series (NN ŷ or OptiTrack packed Y).

    ``val_row`` defaults to ``~train_row``. Pass it explicitly when train is a
    prefix of the original train split so leftover train frames are not scored.
    """
    print(f"[train] Koopman on {tag}  x_dim={x_src.shape[1]}  N={x_src.shape[0]}", flush=True)
    x_mu, x_sd = _x_norm_params(x_src, train_row, int(x_norm))
    xn = (x_src - x_mu) / x_sd
    un = (u_delta - u_mu) / u_sd
    dly = int(delay)
    z_all, u_all = gen_delay_traj(xn, un, dly)
    bad_dly = _invalid_delay_mask(trial_fit, dly)
    t_latest = np.arange(z_all.shape[0]) + dly
    row_ok = train_row[t_latest] & ~bad_dly
    if val_row is None:
        val_ok = (~train_row[t_latest]) & ~bad_dly
    else:
        val_ok = np.asarray(val_row, dtype=bool)[t_latest] & ~bad_dly
    pair_tr = row_ok[:-1] & row_ok[1:]
    pair_va = val_ok[:-1] & val_ok[1:]
    if int(pair_tr.sum()) < 50:
        raise RuntimeError(f"too few {tag} training delay pairs ({int(pair_tr.sum())})")
    koop = train_koopman(
        z_all[:-1][pair_tr],
        u_all[:-1][pair_tr],
        z_all[1:, : x_src.shape[1]][pair_tr],
        delay=dly,
        mode_reg=int(mode_reg),
        alpha=float(alpha),
        tol=float(tol),
    )
    one_rmse = float("nan")
    one_rmse_n = float("nan")
    if pair_va.any():
        y1n = (
            z_all[:-1][pair_va] @ koop["A"][: x_src.shape[1]].T
            + u_all[:-1][pair_va] @ koop["B"][: x_src.shape[1]].T
        )
        ytn = z_all[1:, : x_src.shape[1]][pair_va]
        one_rmse_n = float(np.sqrt(np.mean((y1n - ytn) ** 2)))
        y1 = _denorm(y1n, x_mu, x_sd)
        yt = _denorm(ytn, x_mu, x_sd)
        one_rmse = float(np.sqrt(np.mean((y1 - yt) ** 2)))
    print(
        f"[val] {tag} 1-step RMSE(x)={one_rmse:.6g}  (norm={one_rmse_n:.6g})  "
        f"pairs={int(pair_va.sum())}",
        flush=True,
    )
    koop.update({
        "x_mean": x_mu, "x_std": x_sd, "u_mean": u_mu, "u_std": u_sd,
        "tag": str(tag),
    })
    return {
        "koop": koop, "xn": xn, "un": un, "x_mu": x_mu, "x_sd": x_sd,
        "one_rmse": one_rmse, "one_rmse_norm": one_rmse_n,
        "n_train_pairs": int(pair_tr.sum()), "n_val_pairs": int(pair_va.sum()),
    }


def plot_horizon_rmse_compare(
    save: Path,
    *,
    rmse_nn_vs_gt: np.ndarray,
    rmse_ot_vs_gt: np.ndarray,
    rmse_nn_vs_nn: np.ndarray | None = None,
    sample_rate_hz_nn: float = 400.0,
    sample_rate_hz_ot: float = 100.0,
) -> None:
    t_nn = np.arange(len(rmse_nn_vs_gt)) / float(sample_rate_hz_nn)
    t_ot = np.arange(len(rmse_ot_vs_gt)) / float(sample_rate_hz_ot)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    ax.plot(
        t_nn, 1e3 * rmse_nn_vs_gt, color="tab:red", lw=1.4,
        label=f"Koopman (NN-trained, {sample_rate_hz_nn:g} Hz) vs OptiTrack",
    )
    ax.plot(
        t_ot, 1e3 * rmse_ot_vs_gt, color="tab:green", lw=1.4,
        label=f"Koopman (OptiTrack-trained, {sample_rate_hz_ot:g} Hz) vs OptiTrack",
    )
    if rmse_nn_vs_nn is not None:
        ax.plot(
            t_nn, 1e3 * rmse_nn_vs_nn, color="tab:blue", lw=1.1, ls="--",
            label=f"Koopman (NN-trained, {sample_rate_hz_nn:g} Hz) vs NN ŷ",
        )
    ax.set_xlabel("rollout time [s]")
    ax.set_ylabel("state RMSE [mm]")
    ax.set_title(
        f"Open-loop Koopman rollout  (same wall-clock window, "
        f"NN @ {sample_rate_hz_nn:g} Hz / OT @ {sample_rate_hz_ot:g} Hz)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save_fig(fig, save)


def plot_A_eigs_compare(save: Path, eig_nn: np.ndarray, eig_ot: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained")
    th = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(th), np.sin(th), color="0.75", lw=0.8)
    ax.scatter(eig_nn.real, eig_nn.imag, s=10, c="tab:red", zorder=3, label="trained on NN ŷ")
    ax.scatter(eig_ot.real, eig_ot.imag, s=10, c="tab:green", zorder=3, label="trained on OptiTrack")
    ax.set_aspect("equal")
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.set_title("Koopman A eigenvalues")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save_fig(fig, save)


def rollout(model: dict[str, Any], Xn: np.ndarray, Un: np.ndarray, k0: int, horizon: int) -> np.ndarray:
    d = int(model["delay"])
    a, b, c = model["A"], model["B"], model["C"]
    z = np.concatenate([Xn[k0 - i] for i in range(d + 1)])
    out = []
    for h in range(horizon):
        z = a @ z + b @ Un[k0 + h]
        out.append(c @ z)
    return np.stack(out, axis=0)


def _denorm(xn: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return xn * sd + mu


_RE_PF = re.compile(r"^pF_RB(\d+)_([xyz])$")
_RE_CHAIN = re.compile(r"^pChain_RB(\d+)_in_RB(\d+)_([xyz])$")
_RE_PB = re.compile(r"^pB_RB(\d+)_in_RB(\d+)_([xyz])$")
_RE_FLAT = re.compile(r"^RB(\d+)_p([xyz])$")


def _plane_axis_pair(plane: str) -> tuple[str, str]:
    p = str(plane).strip().lower().replace("+", "")
    if p in ("xy", "yx"):
        return "x", "y"
    if p in ("xz", "zx"):
        return "x", "z"
    if p in ("yz", "zy"):
        return "y", "z"
    # Full xyz state or unknown tag → default to one-seg base plane (x–z).
    return "x", "z"


def _shape_plane_groups(names: list[str], plane: str = "xz") -> list[dict[str, Any]]:
    """Group packed base-plane channels: one panel per (rigid body, frame, kind)."""
    ax0, ax1 = _plane_axis_pair(plane)
    buckets: dict[tuple, dict[str, Any]] = {}
    for i, raw in enumerate(names):
        s = str(raw)
        kind = body = frame = axis = None
        m = _RE_PF.match(s)
        if m:
            kind, body, frame, axis = "global", int(m.group(1)), 1, m.group(2)
        if kind is None:
            m = _RE_CHAIN.match(s)
            if m:
                kind, body, frame, axis = "local", int(m.group(1)), int(m.group(2)), m.group(3)
        if kind is None:
            m = _RE_PB.match(s)
            if m:
                kind, body, frame, axis = "local", int(m.group(1)), int(m.group(2)), m.group(3)
        if kind is None:
            m = _RE_FLAT.match(s)
            if m:
                kind, body, frame, axis = "global", int(m.group(1)), 1, m.group(2)
        if kind is None or axis not in ("x", "y", "z"):
            continue
        key = (0 if kind == "global" else 1, body, frame, kind)
        rec = buckets.setdefault(key, {"kind": kind, "body": body, "frame": frame, "axes": {}})
        rec["axes"][axis] = i
    groups = []
    for key in sorted(buckets):
        rec = buckets[key]
        axmap = rec["axes"]
        if ax0 not in axmap or ax1 not in axmap:
            have = [a for a in ("x", "y", "z") if a in axmap]
            if len(have) < 2:
                continue
            a0, a1 = have[0], have[1]
        else:
            a0, a1 = ax0, ax1
        kind = rec["kind"]
        if kind == "global":
            title = f"RB{rec['body']} global (pF)"
        else:
            title = f"RB{rec['body']} local in RB{rec['frame']} (pB)"
        groups.append(
            {
                "kind": kind,
                "body": rec["body"],
                "frame": rec["frame"],
                "ix": axmap[a0],
                "iy": axmap[a1],
                "ax0": a0,
                "ax1": a1,
                "title": title,
            }
        )
    return groups


# Backward-compatible alias (plots/errors use base-plane axes, not always x–y).
_shape_xy_groups = _shape_plane_groups


def _plane_rmse(a: np.ndarray, b: np.ndarray, ix: int, iy: int) -> float:
    """Mean Euclidean error in the selected base-plane axes (e.g. x–z for one-seg)."""
    dx = a[:, ix] - b[:, ix]
    dy = a[:, iy] - b[:, iy]
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))


_xy_rmse = _plane_rmse


def _plane_rmse_per_step(a: np.ndarray, b: np.ndarray, ix: int, iy: int) -> np.ndarray:
    dx = a[:, ix] - b[:, ix]
    dy = a[:, iy] - b[:, iy]
    return np.sqrt(dx * dx + dy * dy)


def _mean_plane_rmse_series(
    a: np.ndarray,
    b: np.ndarray,
    names: list[str],
    *,
    plane: str = "xz",
) -> np.ndarray:
    """Per-timestep mean base-plane distance [m] over all RB panels (x–z for one-seg)."""
    groups = _shape_plane_groups(names, plane=plane)
    if not groups:
        return np.sqrt(np.mean((a - b) ** 2, axis=1))
    per_body = np.stack(
        [_plane_rmse_per_step(a, b, g["ix"], g["iy"]) for g in groups],
        axis=1,
    )
    return np.mean(per_body, axis=1)


def _tip_plane_rmse_series(
    a: np.ndarray,
    b: np.ndarray,
    names: list[str],
    *,
    plane: str = "xz",
    tip_rb: int = 5,
) -> np.ndarray | None:
    """Tip RB base-plane error [m] per timestep (RB5 x–z for one-seg)."""
    tip_tok = f"RB{int(tip_rb)}"
    groups = [
        g for g in _shape_plane_groups(names, plane=plane)
        if g["kind"] == "global" and int(g["body"]) == int(tip_rb)
    ]
    if not groups:
        return None
    g = groups[0]
    return _plane_rmse_per_step(a, b, g["ix"], g["iy"])


def _save_fig(fig, save: Path) -> None:
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    target = save
    if os.name == "nt":
        raw = str(save.resolve())
        if not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\" + raw
        target = raw
    fig.savefig(target, dpi=130, bbox_inches="tight")


def plot_shape_rollout(
    save: Path,
    *,
    gt_x: np.ndarray,
    nn_x: np.ndarray,
    kp_x: np.ndarray,
    names: list[str],
    option: str,
    plane: str,
    x_mode: str,
    k0: int,
    kp_ot_x: np.ndarray | None = None,
    gt_ot_x: np.ndarray | None = None,
    train_tag: str = "NN",
) -> dict[str, Any]:
    """One plane panel per packed rigid body / frame.

    Default three curves: OptiTrack, NN reconstructed, Koopman trained on NN.
    Optional ``kp_ot_x``: Koopman trained on OptiTrack packed shape (may be
    a coarser 100 Hz grid). Pass matching ``gt_ot_x`` for OT RMSE.
    """
    groups = _shape_plane_groups(names, plane=plane)
    if not groups:
        raise RuntimeError(f"no {plane} position channels in this NN output to plot")
    n_panels = len(groups)
    n_cols = min(5, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.35 * n_cols, 3.15 * n_rows), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()
    panel_stats: list[dict[str, Any]] = []
    ax0 = groups[0].get("ax0", "x")
    ax1 = groups[0].get("ax1", "y")

    for j, g in enumerate(groups):
        ax = axes[j]
        ix, iy = g["ix"], g["iy"]
        ax.plot(gt_x[:, ix], gt_x[:, iy], color="0.35", lw=1.6, label="OptiTrack GT")
        ax.plot(nn_x[:, ix], nn_x[:, iy], color="tab:blue", lw=1.3, label="NN reconstructed")
        ax.plot(
            kp_x[:, ix], kp_x[:, iy], color="tab:red", lw=1.2, ls="--",
            label=f"Koopman ({train_tag})",
        )
        if kp_ot_x is not None:
            ax.plot(
                kp_ot_x[:, ix], kp_ot_x[:, iy], color="tab:green", lw=1.2, ls=":",
                label="Koopman (OptiTrack)",
            )
        ax.scatter(float(gt_x[0, ix]), float(gt_x[0, iy]), c="green", s=14, zorder=5)
        rmse_nn = _plane_rmse(kp_x, nn_x, ix, iy)
        rmse_gt = _plane_rmse(kp_x, gt_x, ix, iy)
        rec = {**g, "rmse_vs_nn_m": rmse_nn, "rmse_vs_gt_m": rmse_gt}
        title = (
            f"{g['title']}\n"
            f"K({train_tag}) vs NN={1000.0 * rmse_nn:.3g}  vs GT={1000.0 * rmse_gt:.3g} mm"
        )
        if kp_ot_x is not None:
            gt_for_ot = gt_x if gt_ot_x is None else gt_ot_x
            rec["rmse_ot_vs_gt_m"] = _plane_rmse(kp_ot_x, gt_for_ot, ix, iy)
            title += f"\nK(OT) vs GT={1000.0 * rec['rmse_ot_vs_gt_m']:.3g} mm"
        panel_stats.append(rec)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel(f"{g.get('ax0', ax0)} (m)")
        ax.set_ylabel(f"{g.get('ax1', ax1)} (m)")
        ax.grid(True, alpha=0.3)
        try:
            ax.set_aspect("equal", adjustable="datalim")
        except Exception:
            pass
        if j == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    mean_nn = float(np.mean([p["rmse_vs_nn_m"] for p in panel_stats]))
    mean_gt = float(np.mean([p["rmse_vs_gt_m"] for p in panel_stats]))
    n_g = sum(1 for p in panel_stats if p["kind"] == "global")
    n_l = sum(1 for p in panel_stats if p["kind"] == "local")
    extra = ""
    mean_ot = None
    ot_steps = ""
    if kp_ot_x is not None:
        mean_ot = float(np.mean([p["rmse_ot_vs_gt_m"] for p in panel_stats]))
        extra = f"    K(OT) vs GT={1000.0 * mean_ot:.3g} mm"
        ot_steps = f", OT steps={len(kp_ot_x)}"
    fig.suptitle(
        f"option {option}/{plane}  x_mode={x_mode}  "
        f"{n_g} global + {n_l} local  (k0={k0}, NN steps={len(kp_x)}{ot_steps})\n"
        f"K({train_tag}) vs NN={1000.0 * mean_nn:.3g} mm    vs GT={1000.0 * mean_gt:.3g} mm"
        + extra,
        fontsize=11,
    )
    _save_fig(fig, save)
    out = {"panels": panel_stats, "mean_rmse_vs_nn_m": mean_nn, "mean_rmse_vs_gt_m": mean_gt}
    if mean_ot is not None:
        out["mean_rmse_ot_vs_gt_m"] = mean_ot
    return out


def plot_A_eigs(save: Path, eig: np.ndarray, a: np.ndarray, *, title: str = "Koopman A eigenvalues") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
    th = np.linspace(0, 2 * np.pi, 256)
    axes[0].plot(np.cos(th), np.sin(th), color="0.75", lw=0.8)
    axes[0].scatter(eig.real, eig.imag, s=10, c="tab:red", zorder=3)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Re λ")
    axes[0].set_ylabel("Im λ")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    amax = float(np.max(np.abs(a)))
    if not np.isfinite(amax) or amax <= 0.0:
        amax = 1.0
    im = axes[1].imshow(a, cmap="bwr", aspect="auto", vmin=-amax, vmax=amax)
    axes[1].set_title("A heatmap")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    _save_fig(fig, save)


def main() -> None:
    # =====================================================================
    # Tunables  (edit these; changing the NN auto-changes INPUT / OUTPUT / x)
    # =====================================================================
    # Relative path under NN_training/Training/models/  (or absolute run folder).
    # Switching NN auto-selects INPUT pack from that run's config.json:
    #   LSTM → enc_trq_imu12 (14-D)     MLP → enc_imu12_u (16-D)
    #   both → OUTPUT e/xz (16-D) as Koopman x (mode=full)
    nn_model_rel = (
        "LSTM/lstm_L2_H256_seq100_stride50_drop1e-1_lr1e-3_sr400_valtip_rmse_oute+xy_lossglocal"
    )
    # nn_model_rel = (
    #     "MLP/mlp_L2_W256_drop0_lr1e-3_actrelu_sr400_valtip_rmse_oute+xy_lossglocal"
    #      LSTM/lstm_L2_H256_seq100_stride50_drop1e-1_lr1e-3_sr400_valtip_rmse_oute+xy_lossglocal
    # )

    data_dir = ICRA                         # workspace root (data_*.csv)
    csv_name_contains = "data_20260825_192414"  # "" = every data_*.csv under data_dir
    max_csvs = None                         # e.g. 1 for a smoke run
    max_frames_per_csv = None               # e.g. 40000 for a smoke run

    delay = 10                              # taps at the NN Koopman rate (native unless subsampled)
    seed = 42
    koopman_x_mode = "full"                 # full | global | tip
    train_ratio = 0.95                      # 95/5 on freeze-trimmed rows only
    split_mode = "per_trial"                # per_trial | global
    skip_start_freeze = True                # drop first n_imu_calib frames (~5 s)
    delta_u_handle = 1                      # 1 = Δenc; 0 = absolute enc
    u_norm = 1                              # 1 = divide u by train std
    x_norm = 2                              # 0=off, 1=X/std, 2=(X-mean)/std
    mode_reg = 2                            # 0=lstsq, 1=lasso, 2=ridge
    ridge_alpha = 1e-1
    ident_tol = 1e-12
    warmup = "auto"                         # auto → seq_len for LSTM/TCN else 0

    # Rates: NN Koopman stays at native CSV Hz (400); OptiTrack Koopman stays at 100 Hz.
    # Set koopman_train_hz (e.g. 100.0) only for optional subsample / ablation tests.
    koopman_train_hz = None                 # None → native (400 Hz); do not change for ramping
    num_predict = 4000                      # wall-clock horizon at native (CSV) Hz
    optitrack_hz = 100.0                    # OptiTrack always 100 Hz (compare path)
    show_figs = False                       # True opens interactive windows
    compare_optitrack_koopman = True        # also fit/eval Koopman on packed OT Y

    # =====================================================================
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    csvs = _list_csvs(Path(data_dir), name_contains=str(csv_name_contains), max_csvs=max_csvs)
    nn_rel, nn_path = resolve_nn_run(nn_model_rel)
    print(f"\n########  Koopman on NN  {nn_rel}  ########", flush=True)
    run_koopman_train(
        nn_path,
        nn_rel=nn_rel,
        csvs=csvs,
        max_frames_per_csv=max_frames_per_csv,
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
        koopman_train_hz=koopman_train_hz,
        num_predict=num_predict,
        optitrack_hz=optitrack_hz,
        show_figs=show_figs,
        compare_optitrack_koopman=compare_optitrack_koopman,
        rng=rng,
    )


def run_koopman_train(
    nn_dir: Path,
    *,
    nn_rel: str = "",
    csvs: list[Path],
    max_frames_per_csv: int | None,
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
    koopman_train_hz: float | None,
    num_predict: int,
    optitrack_hz: float,
    show_figs: bool,
    compare_optitrack_koopman: bool,
    rng: np.random.Generator,
) -> None:
    nn_dir = _resolve_nn_dir(Path(nn_dir))
    if not nn_rel:
        nn_rel, _ = resolve_nn_run(nn_dir)
    cfg = json.loads((nn_dir / "config.json").read_text(encoding="utf-8"))
    ckpt = _resolve_ckpt(nn_dir)
    arch = str(cfg.get("arch", "MLP")).strip().lower()
    option = str(cfg.get("shape_option", "e"))
    plane = str(cfg.get("plane", "xz"))
    if str(warmup).strip().lower() == "auto":
        warm = int(cfg["seq_len"]) if arch in ("lstm", "tcn") and "seq_len" in cfg else 0
    else:
        warm = int(warmup)
    ident = _parse_mode_reg(mode_reg)
    n_imu_calib = int(
        (cfg.get("data_meta") or {}).get("n_imu_calib", N_IMU_CALIB_DEFAULT)
    )
    n_skip_start = max(warm, n_imu_calib if skip_start_freeze else 0)
    io = _nn_io_flags(cfg)
    native_hz = float(cfg.get("sample_rate_hz", 400.0))
    if koopman_train_hz is None or float(koopman_train_hz) <= 0.0 or float(koopman_train_hz) >= native_hz:
        stride_nn = 1
        nn_hz = native_hz
    else:
        stride_nn = max(1, int(round(native_hz / float(koopman_train_hz))))
        nn_hz = native_hz / float(stride_nn)
    n_pred_nn = max(1, int(num_predict) // stride_nn)
    stride_ot = max(1, int(round(native_hz / float(optitrack_hz))))
    ot_hz = native_hz / float(stride_ot)
    n_pred_ot = max(1, int(num_predict) // stride_ot)

    print("=" * 72, flush=True)
    print(f"NN          : {nn_rel}")
    print(f"             folder={nn_dir}")
    print(
        f"             arch={arch}  INPUT={io['key']} ({cfg.get('in_dim')}-D)  "
        f"OUTPUT={option}/{plane} ({cfg.get('out_dim')}-D)  loss={cfg.get('loss_id')}"
    )
    print(
        f"             trq={io['use_trq']} imu={io['imu']} cmd={io['cmd']}  "
        f"imu_frame={io['imu_frame']}  → Koopman x_mode={koopman_x_mode}"
    )
    print(
        f"rate        : native={native_hz:g} Hz  Koopman(NN)={nn_hz:g} Hz  "
        f"OptiTrack={ot_hz:g} Hz  stride_nn={stride_nn} stride_ot={stride_ot}  "
        f"horizon NN={n_pred_nn} / OT={n_pred_ot} ({num_predict / native_hz:.3g} s wall-clock)"
    )
    print(f"u           : enc_rel (3)  delta_u={delta_u_handle}  delay={delay}  seed={seed}")
    print(
        f"ident       : {_ident_name(ident)} (mode_reg={ident})  alpha={ridge_alpha:g}"
    )
    print(
        f"split       : {split_mode}  train_ratio={train_ratio}  "
        f"(freeze-tail dropped at load; start skip={n_skip_start} "
        f"warmup={warm} imu_calib={n_imu_calib})"
    )
    print(f"save        : {KOOPMAN_MODEL_DIR / nn_rel}")
    print(f"data        : {len(csvs)} csv(s)")
    for p in csvs:
        print(f"             {p}", flush=True)
    print("=" * 72, flush=True)

    data = load_robot_trials(csvs, cfg, max_frames_per_csv=max_frames_per_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nn = build_nn(cfg, device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model_nn.load_state_dict(state)
    model_nn.eval()
    print(f"[nn] loaded {ckpt}  device={device}", flush=True)

    y_hat = nn_reconstruct_y(model_nn, cfg, data, device)
    x_nn, x_info = _slice_x(y_hat, cfg, koopman_x_mode)
    x_gt, _ = _slice_x(np.asarray(data["Y_gt"], dtype=np.float64), cfg, koopman_x_mode)
    names = list(x_info["feature_names"])
    print(f"[nn] ŷ → x  dim={x_nn.shape[1]}  mode={x_info['x_mode']}", flush=True)

    keep = _warmup_per_trial(data["trial_id"], n_skip_start)
    print(
        f"[split] dropped freeze/warmup prefix {n_skip_start} frames/trial; "
        f"95/5 on remaining N={int(keep.sum())} (no freeze-tail rows)",
        flush=True,
    )
    x_nn, x_gt, u_abs = x_nn[keep], x_gt[keep], data["U_abs"][keep]
    trial = data["trial_id"][keep]

    if stride_nn > 1:
        print(
            f"[nn] subsample stride={stride_nn}  {native_hz:g}→{nn_hz:g} Hz  "
            f"N={x_nn.shape[0] // stride_nn}  (Δenc at {nn_hz:g} Hz, not every-{stride_nn}th 400 Hz Δ)",
            flush=True,
        )
        x_nn_k = x_nn[::stride_nn]
        x_gt_k = x_gt[::stride_nn]
        u_abs_k = u_abs[::stride_nn]
        trial_k = trial[::stride_nn]
    else:
        x_nn_k, x_gt_k, u_abs_k, trial_k = x_nn, x_gt, u_abs, trial

    x_fit, x_gt_fit, u_delta, trial_fit, train_row = _delta_and_split(
        x_nn_k, x_gt_k, u_abs_k, trial_k,
        delta_u_handle=int(delta_u_handle),
        train_ratio=train_ratio, split_mode=split_mode,
    )
    u_mu, u_sd = _u_norm_params(
        u_delta, train_row, u_norm=int(u_norm), delta_u_handle=int(delta_u_handle),
    )

    dly = int(delay)
    fit_nn = identify_on_state(
        x_fit, u_delta, trial_fit, train_row,
        delay=dly, x_norm=int(x_norm), u_mu=u_mu, u_sd=u_sd,
        mode_reg=ident, alpha=float(ridge_alpha), tol=float(ident_tol),
        tag="NN reconstructed ŷ",
    )
    koop = fit_nn["koop"]
    koop.update(
        {
            "delta_u_handle": int(delta_u_handle),
            "u_norm": int(u_norm),
            "x_norm": int(x_norm),
            "nn_run_id": nn_dir.name,
            "nn_rel": nn_rel,
            "x_mode": x_info["x_mode"],
            "feature_names": names,
            "shape_option": option,
            "plane": plane,
            "source": "nn",
            "sample_rate_hz": float(nn_hz),
            "native_sample_rate_hz": float(native_hz),
            "stride": int(stride_nn),
            "u_kind": "enc_rel",
        }
    )
    x_mu, x_sd = fit_nn["x_mu"], fit_nn["x_sd"]
    xn, un = fit_nn["xn"], fit_nn["un"]
    one_rmse, one_rmse_n = fit_nn["one_rmse"], fit_nn["one_rmse_norm"]

    fit_ot = None
    x_gt_fit_ot = None
    trial_ot = None
    u_mu_ot, u_sd_ot = u_mu, u_sd
    if compare_optitrack_koopman:
        # Subsample THEN Δκ so 100 Hz controls are κ[t+10ms]−κ[t], not every-4th 400 Hz Δκ.
        _, x_gt_fit_ot, u_delta_ot, trial_ot, train_row_ot = _delta_and_split(
            x_gt[::stride_ot], x_gt[::stride_ot], u_abs[::stride_ot], trial[::stride_ot],
            delta_u_handle=int(delta_u_handle),
            train_ratio=train_ratio, split_mode=split_mode,
        )
        u_mu_ot, u_sd_ot = _u_norm_params(
            u_delta_ot, train_row_ot, u_norm=int(u_norm), delta_u_handle=int(delta_u_handle),
        )
        print(
            f"[ot] subsample stride={stride_ot}  {native_hz:g}→{ot_hz:g} Hz  "
            f"N={x_gt_fit_ot.shape[0]}  delay={dly} taps  horizon={n_pred_ot}",
            flush=True,
        )
        fit_ot = identify_on_state(
            x_gt_fit_ot, u_delta_ot, trial_ot, train_row_ot,
            delay=dly, x_norm=int(x_norm), u_mu=u_mu_ot, u_sd=u_sd_ot,
            mode_reg=ident, alpha=float(ridge_alpha), tol=float(ident_tol),
            tag=f"OptiTrack packed Y @ {ot_hz:g} Hz",
        )
        fit_ot["koop"].update(
            {
                "delta_u_handle": int(delta_u_handle),
                "u_norm": int(u_norm),
                "x_norm": int(x_norm),
                "nn_run_id": nn_dir.name,
                "x_mode": x_info["x_mode"],
                "feature_names": names,
                "shape_option": option,
                "plane": plane,
                "source": "optitrack",
                "sample_rate_hz": float(ot_hz),
                "stride": int(stride_ot),
            }
        )

    val_idx = np.flatnonzero(~train_row)
    val_idx = val_idx[(val_idx >= dly) & (val_idx + n_pred_nn + 2 < x_fit.shape[0])]
    same_trial = [
        int(k) for k in val_idx
        if np.all(trial_fit[k - dly : k + n_pred_nn + 1] == trial_fit[k])
    ]
    if fit_ot is not None:
        ot_ok = []
        for k in same_trial:
            if stride_ot == stride_nn:
                k_ot = k
            elif k % stride_ot != 0:
                continue
            else:
                k_ot = k // stride_ot
            if k_ot < dly or k_ot + n_pred_ot + 2 >= x_gt_fit_ot.shape[0]:
                continue
            sl = trial_ot[k_ot - dly : k_ot + n_pred_ot + 1]
            if sl.size == (dly + n_pred_ot + 1) and np.all(sl == trial_ot[k_ot]):
                ot_ok.append(k)
        if ot_ok:
            same_trial = ot_ok
        else:
            print(
                "[warn] no NN val k0 also valid on the OptiTrack 100 Hz grid; "
                "OT rollout will snap k0//stride",
                flush=True,
            )
    if not same_trial:
        raise RuntimeError("no contiguous validation window (try a longer CSV or smaller num_predict)")
    k0 = int(same_trial[int(rng.integers(0, len(same_trial)))])
    n_pred = int(n_pred_nn)
    print(
        f"[val] NN k0={k0}  horizon={n_pred} @ {nn_hz:g} Hz  "
        f"({n_pred / nn_hz:.3g} s)  candidates={len(same_trial)}",
        flush=True,
    )

    pred_n = rollout(koop, xn, un, k0, n_pred)
    true_n = xn[k0 + 1 : k0 + 1 + n_pred]
    h = min(len(pred_n), len(true_n))
    pred_n, true_n = pred_n[:h], true_n[:h]
    nn_x = _denorm(true_n, x_mu, x_sd)
    pred_x = _denorm(pred_n, x_mu, x_sd)
    gt_x = np.asarray(x_gt_fit[k0 + 1 : k0 + 1 + h], dtype=np.float64)
    rmse_h = np.sqrt(np.mean((pred_x - nn_x) ** 2, axis=1))
    rmse_nn_vs_gt = np.sqrt(np.mean((pred_x - gt_x) ** 2, axis=1))
    tip_rb = int(
        (cfg.get("data_meta") or {}).get("tip_rb")
        or (cfg.get("label_meta") or {}).get("tip_rb")
        or 5
    )
    rmse_plane_vs_nn = _mean_plane_rmse_series(pred_x, nn_x, names, plane=plane)
    rmse_plane_vs_gt = _mean_plane_rmse_series(pred_x, gt_x, names, plane=plane)
    rmse_tip_vs_nn = _tip_plane_rmse_series(pred_x, nn_x, names, plane=plane, tip_rb=tip_rb)
    rmse_tip_vs_gt = _tip_plane_rmse_series(pred_x, gt_x, names, plane=plane, tip_rb=tip_rb)

    pred_ot_x = None
    rmse_ot_vs_gt = None
    gt_ot_x = None
    k0_ot = None
    h_ot = None
    if fit_ot is not None:
        if stride_ot == stride_nn:
            k0_ot = int(k0)
        else:
            k0_ot = int(k0 // stride_ot)
        if k0_ot < dly:
            k0_ot = int(dly)
        h_ot = min(int(n_pred_ot), int(x_gt_fit_ot.shape[0]) - k0_ot - 1)
        if h_ot < 2:
            raise RuntimeError("OptiTrack 100 Hz validation window is too short")
        pred_ot_n = rollout(fit_ot["koop"], fit_ot["xn"], fit_ot["un"], k0_ot, h_ot)
        h_ot = min(h_ot, len(pred_ot_n))
        pred_ot_x = _denorm(pred_ot_n[:h_ot], fit_ot["x_mu"], fit_ot["x_sd"])
        gt_ot_x = np.asarray(x_gt_fit_ot[k0_ot + 1 : k0_ot + 1 + h_ot], dtype=np.float64)
        rmse_ot_vs_gt = np.sqrt(np.mean((pred_ot_x - gt_ot_x) ** 2, axis=1))
        print(
            f"[val] OT k0={k0_ot}  horizon={h_ot} @ {ot_hz:g} Hz  "
            f"({h_ot / ot_hz:.3g} s)",
            flush=True,
        )
        print(
            f"[val] rollout vs OptiTrack  state K(NN)={1000.0 * float(np.mean(rmse_nn_vs_gt)):.3g} mm  "
            f"plane {plane} K(NN)={1000.0 * float(np.mean(rmse_plane_vs_gt)):.3g} mm  "
            f"K(OT)={1000.0 * float(np.mean(rmse_ot_vs_gt)):.3g} mm",
            flush=True,
        )
    if rmse_tip_vs_gt is not None:
        print(
            f"[val] tip RB{tip_rb} {plane}  K(NN) vs GT={1000.0 * float(np.mean(rmse_tip_vs_gt)):.3g} mm  "
            f"vs NN={1000.0 * float(np.mean(rmse_tip_vs_nn)):.3g} mm",
            flush=True,
        )

    # Mirror the NN folder under Koopman training/Model/<nn_rel>/koop_…
    hz_tag = "" if stride_nn <= 1 else f"_hz{int(round(nn_hz))}"
    out_dir = KOOPMAN_MODEL_DIR / nn_rel / (
        f"koop_d{dly}_seed{seed}_{koopman_x_mode}_{_ident_name(ident)}{hz_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_A_eigs(out_dir / "fig_A_eigs.png", koop["eig"], koop["A"], title="A eigenvalues (trained on NN ŷ)")
    if fit_ot is not None:
        plot_A_eigs(
            out_dir / "fig_A_eigs_optitrack.png",
            fit_ot["koop"]["eig"], fit_ot["koop"]["A"],
            title=f"A eigenvalues (trained on OptiTrack @ {ot_hz:g} Hz)",
        )
        plot_A_eigs_compare(out_dir / "fig_A_eigs_compare.png", koop["eig"], fit_ot["koop"]["eig"])
        plot_horizon_rmse_compare(
            out_dir / "fig_compare_rmse_horizon.png",
            rmse_nn_vs_gt=rmse_nn_vs_gt,
            rmse_ot_vs_gt=rmse_ot_vs_gt,
            rmse_nn_vs_nn=rmse_h,
            sample_rate_hz_nn=nn_hz,
            sample_rate_hz_ot=ot_hz,
        )
    shape_stats = plot_shape_rollout(
        out_dir / "fig_shape_rollout_plane.png",
        gt_x=gt_x,
        nn_x=nn_x,
        kp_x=pred_x,
        names=names,
        option=option,
        plane=plane,
        x_mode=x_info["x_mode"],
        k0=k0,
        kp_ot_x=pred_ot_x,
        gt_ot_x=gt_ot_x,
        train_tag="NN",
    )
    print(
        f"[val] shape {plane}  K(NN) vs NN={1000.0 * shape_stats['mean_rmse_vs_nn_m']:.3g} mm  "
        f"vs GT={1000.0 * shape_stats['mean_rmse_vs_gt_m']:.3g} mm",
        flush=True,
    )
    if "mean_rmse_ot_vs_gt_m" in shape_stats:
        print(
            f"[val] shape {plane}  K(OT) vs GT={1000.0 * shape_stats['mean_rmse_ot_vs_gt_m']:.3g} mm",
            flush=True,
        )

    np.savez_compressed(
        out_dir / "koopman_model.npz",
        A=koop["A"], B=koop["B"], C=koop["C"],
        x_mean=x_mu, x_std=x_sd, u_mean=u_mu, u_std=u_sd,
        delay=np.array([dly], dtype=np.int64),
        seed=np.array([int(seed)], dtype=np.int64),
        delta_u_handle=np.array([int(delta_u_handle)], dtype=np.int64),
        feature_names=np.array(names),
        cols=np.array(x_info["cols"], dtype=np.int64),
        source=np.array(["nn"]),
        sample_rate_hz=np.array([nn_hz], dtype=np.float64),
        stride=np.array([stride_nn], dtype=np.int64),
    )
    if fit_ot is not None:
        np.savez_compressed(
            out_dir / "koopman_model_optitrack.npz",
            A=fit_ot["koop"]["A"], B=fit_ot["koop"]["B"], C=fit_ot["koop"]["C"],
            x_mean=fit_ot["x_mu"], x_std=fit_ot["x_sd"], u_mean=u_mu_ot, u_std=u_sd_ot,
            delay=np.array([dly], dtype=np.int64),
            seed=np.array([int(seed)], dtype=np.int64),
            delta_u_handle=np.array([int(delta_u_handle)], dtype=np.int64),
            feature_names=np.array(names),
            cols=np.array(x_info["cols"], dtype=np.int64),
            source=np.array(["optitrack"]),
            sample_rate_hz=np.array([ot_hz], dtype=np.float64),
            stride=np.array([stride_ot], dtype=np.int64),
        )
    meta = {
        "nn_rel": nn_rel,
        "nn_run_id": nn_dir.name,
        "nn_ckpt": str(ckpt),
        "nn_config": {
            k: cfg[k] for k in (
                "arch", "shape_option", "plane", "use_trq", "imu", "cmd",
                "in_dim", "out_dim", "loss_id", "seq_len", "layers", "hidden", "width",
                "sample_rate_hz",
            ) if k in cfg
        },
        "nn_input_key": io["key"],
        "nn_imu_frame": io["imu_frame"],
        "koopman_x_mode": x_info["x_mode"],
        "feature_names": names,
        "delay": dly,
        "seed": int(seed),
        "k0": k0,
        "num_predict": h,
        "sample_rate_hz": nn_hz,
        "native_sample_rate_hz": native_hz,
        "koopman_stride": int(stride_nn),
        "koopman_train_hz": float(nn_hz),
        "source": "nn",
        "u_kind": "enc_rel",
        "n_u": int(data.get("n_u", 3)),
        "one_step_rmse": one_rmse,
        "one_step_rmse_norm": one_rmse_n,
        "rollout_rmse_last": float(rmse_h[-1]),
        "rollout_rmse_vs_gt_mean": float(np.mean(rmse_nn_vs_gt)),
        "rollout_plane_rmse_vs_gt_mean": float(np.mean(rmse_plane_vs_gt)),
        "rollout_plane_rmse_vs_nn_mean": float(np.mean(rmse_plane_vs_nn)),
        "shape_mean_rmse_vs_nn_m": shape_stats["mean_rmse_vs_nn_m"],
        "shape_mean_rmse_vs_gt_m": shape_stats["mean_rmse_vs_gt_m"],
        "shape_panels": [
            {
                "title": p["title"],
                "rmse_vs_nn_m": p["rmse_vs_nn_m"],
                "rmse_vs_gt_m": p["rmse_vs_gt_m"],
                **({"rmse_ot_vs_gt_m": p["rmse_ot_vs_gt_m"]} if "rmse_ot_vs_gt_m" in p else {}),
            }
            for p in shape_stats["panels"]
        ],
        "n_train_pairs": fit_nn["n_train_pairs"],
        "n_val_pairs": fit_nn["n_val_pairs"],
        "csv": data["csv_paths"],
        "delta_u_handle": int(delta_u_handle),
        "x_norm": int(x_norm),
        "u_norm": int(u_norm),
        "ridge_alpha": float(ridge_alpha),
        "mode_reg": int(ident),
        "ident": _ident_name(ident),
        "skip_start_freeze": bool(skip_start_freeze),
        "n_skip_start": int(n_skip_start),
        "warmup": warm,
        "compare_optitrack_koopman": bool(compare_optitrack_koopman),
        "optitrack_hz": float(optitrack_hz),
        "optitrack_stride": int(stride_ot),
        "out_dir": str(out_dir),
        "plane": plane,
        "tip_rb": int(tip_rb),
    }
    if rmse_tip_vs_gt is not None:
        meta["rollout_tip_rmse_vs_gt_mean"] = float(np.mean(rmse_tip_vs_gt))
        meta["rollout_tip_rmse_vs_nn_mean"] = float(np.mean(rmse_tip_vs_nn))
    if fit_ot is not None:
        meta["optitrack_koopman"] = {
            "one_step_rmse": fit_ot["one_rmse"],
            "one_step_rmse_norm": fit_ot["one_rmse_norm"],
            "n_train_pairs": fit_ot["n_train_pairs"],
            "n_val_pairs": fit_ot["n_val_pairs"],
            "sample_rate_hz": float(ot_hz),
            "stride": int(stride_ot),
            "k0": int(k0_ot),
            "num_predict": int(h_ot),
            "rollout_rmse_vs_gt_mean": float(np.mean(rmse_ot_vs_gt)),
            "shape_mean_rmse_vs_gt_m": shape_stats.get("mean_rmse_ot_vs_gt_m"),
            "npz": "koopman_model_optitrack.npz",
        }
    (out_dir / "results.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] artifacts → {out_dir.resolve()}", flush=True)
    print(f"[done]   koopman_model.npz          → {out_dir / 'koopman_model.npz'}", flush=True)
    if fit_ot is not None:
        print(
            f"[done]   koopman_model_optitrack.npz → {out_dir / 'koopman_model_optitrack.npz'}",
            flush=True,
        )
    print(f"[done]   results.json               → {out_dir / 'results.json'}", flush=True)
    if show_figs:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
