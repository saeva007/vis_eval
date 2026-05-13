#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remote one-click inference, baseline evaluation, and journal-style plotting.

This script is intentionally designed for the remote /public/home/putianshu/vis_mlp
environment. It runs the PM10+PM2.5 S2 model, matches the operational IFS
diagnostic visibility baseline, computes paper metrics, and writes publication
figures plus a source manifest.

Core class definition used throughout the paper:
  0: 0 <= visibility < 500 m
  1: 500 <= visibility < 1000 m
  2: visibility >= 1000 m

Examples
--------
Smoke test on remote:
  python vis_eval/run_paper_eval_pm10_pm25_journal.py --mode main --limit_samples 20000

Full run:
  python vis_eval/run_paper_eval_pm10_pm25_journal.py --mode all

Post-process existing tables and remote training logs:
  python vis_eval/run_paper_eval_pm10_pm25_journal.py --mode tables \
    --history_paths /public/home/putianshu/vis_mlp/train/logs/112606205.out,/public/home/putianshu/logs/111696811.out \
    --history_labels "No FE values,Full FE"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
LOCAL_ROOT = VIS_EVAL_DIR.parent

for _p in (str(LOCAL_ROOT), str(VIS_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from metrics_core import (
        compute_rare_event_report,
        pred_from_joint_thresholds,
        pred_from_thresholds,
        pred_from_thresholds_mutual,
    )
except Exception as exc:  # pragma: no cover - import failure should be explicit on remote.
    raise RuntimeError(f"Cannot import vis_eval metrics_core from {VIS_EVAL_DIR}") from exc

try:
    from plot_spatial import (
        run_widespread_event_evaluation,
        detect_widespread_fog_events,
        load_china_shapefile as _load_china_shapefile,
        _draw_event_basemap as _plot_spatial_event_basemap,
    )
except Exception:
    run_widespread_event_evaluation = None
    detect_widespread_fog_events = None
    _load_china_shapefile = None
    _plot_spatial_event_basemap = None


# ---------------------------------------------------------------------------
# Constants and style
# ---------------------------------------------------------------------------


FOG_COLOR = "#2E5A87"
MIST_COLOR = "#E69F00"
CLEAR_COLOR = "#7F7F7F"
PMST_COLOR = "#2E5A87"
IFS_PMST_COLOR = "#2A9D8F"
IFS_DIAG_COLOR = "#5B5B5B"
CLASS_COLORS = [FOG_COLOR, MIST_COLOR, CLEAR_COLOR]
CLASS_NAMES = ["Fog", "Mist", "Clear"]
CLASS_LONG = ["Fog (0-500 m)", "Mist (500-1000 m)", "Clear (>=1000 m)"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CLASS_CMAP.N)
LOCAL_TIME_OFFSET_HOURS = 8
LOCAL_TIME_LABEL = "UTC+8"
TIME_OF_DAY_LOCAL_ORDER = [
    f"Night (00-05 {LOCAL_TIME_LABEL})",
    f"Morning (06-11 {LOCAL_TIME_LABEL})",
    f"Afternoon (12-17 {LOCAL_TIME_LABEL})",
    f"Evening (18-23 {LOCAL_TIME_LABEL})",
]
DEFAULT_S2_RUN_ID = "exp_1776227576_pm10_more_temp_search_utc"
DEFAULT_DATA_DIR = "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_DATA_48H_DIR = "ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime"
DEFAULT_MODEL_PY = "PMST_net_test_11_s2_pm10.py"
DEFAULT_CKPT_PATH = f"checkpoints/{DEFAULT_S2_RUN_ID}_S2_PhaseB_best_score.pt"
DEFAULT_SCALER_PATH = f"checkpoints/robust_scaler_{DEFAULT_S2_RUN_ID}_w12_dyn27_s2_48h_pm10.pkl"
DEFAULT_SEASON_TH_PATH = f"checkpoints/{DEFAULT_S2_RUN_ID}_season_thresholds.pt"
DEFAULT_OUT_DIR = "paper_eval_results_pm10_pm25_journal_utc"
KNOWN_CONVERGENCE_LOG_LABELS = {
    "112606205.out": "No FE values",
    "111696811.out": "Full FE",
}

REGION_DEFS = [
    ("Northeast", 38.5, 54.0, 118.0, 136.0),
    ("North China", 34.0, 42.5, 110.0, 122.5),
    ("East China", 24.0, 34.5, 116.0, 123.5),
    ("Central China", 26.0, 34.5, 108.0, 116.5),
    ("South China", 18.0, 26.5, 108.0, 121.0),
    ("Southwest", 21.0, 34.5, 97.0, 108.5),
    ("Northwest", 34.0, 50.0, 73.0, 110.5),
]


def setup_journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


@dataclass
class Manifest:
    out_dir: Path
    rows: List[Dict[str, object]] = field(default_factory=list)

    def add(
        self,
        figure: str,
        sources: Sequence[str],
        notes: str = "",
        n: Optional[int] = None,
        matched_ifs: Optional[int] = None,
    ) -> None:
        self.rows.append(
            {
                "figure": figure,
                "sources": ";".join(str(s) for s in sources if s),
                "notes": notes,
                "n": "" if n is None else int(n),
                "matched_ifs": "" if matched_ifs is None else int(matched_ifs),
            }
        )

    def write(self) -> None:
        path = self.out_dir / "figure_source_manifest.csv"
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"[manifest] {path}", flush=True)


def save_fig_pair(fig, out_dir: Path, stem: str, manifest: Optional[Manifest] = None,
                  sources: Sequence[str] = (), notes: str = "", n: Optional[int] = None,
                  matched_ifs: Optional[int] = None) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
        print(f"  [Fig] Saved -> {path}", flush=True)
    if manifest is not None:
        manifest.add(f"{stem}.png/pdf", sources, notes=notes, n=n, matched_ifs=matched_ifs)
    plt.close(fig)
    return paths


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# CLI and path helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run PM10+PM2.5 paper evaluation, IFS baseline matching, and journal figures."
    )
    ap.add_argument("--mode", choices=["main", "overlap", "all", "tables"], default="all")
    ap.add_argument("--base", default=os.environ.get("VIS_MLP_ROOT", "/public/home/putianshu/vis_mlp"))
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--data_48h_dir", default=DEFAULT_DATA_48H_DIR)
    ap.add_argument("--ckpt_path", default=DEFAULT_CKPT_PATH)
    ap.add_argument("--scaler_path", default=DEFAULT_SCALER_PATH)
    ap.add_argument("--season_th_path", default=DEFAULT_SEASON_TH_PATH)
    ap.add_argument("--model_py", default=DEFAULT_MODEL_PY)
    ap.add_argument("--ifs_vis_nc", default="VIS_IDW_KDTree_20250101_20251231.nc")
    ap.add_argument("--ifs_48h_nc", default="IFS_VIS_0_48h_stations_2025_00_12.nc")
    ap.add_argument("--ifs_vis_var", default="VIS")
    ap.add_argument("--ifs_48h_var", default="VIS_ifs")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--shp_path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    ap.add_argument("--window_size", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=0, help="0 means infer from X_test.npy")
    ap.add_argument("--extra_feat_dim", type=int, default=0, help="0 means infer from X_test.npy")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--fog_th", type=float, default=0.10)
    ap.add_argument("--mist_th", type=float, default=0.42)
    ap.add_argument("--lead_fog_th", type=float, default=0.10)
    ap.add_argument("--lead_mist_th", type=float, default=0.30)
    ap.add_argument("--threshold_rule", choices=["default", "mutual", "joint"], default="mutual")
    ap.add_argument("--no_calibration", action="store_true")
    ap.add_argument("--use_calibration", action="store_true", help="Load --season_th_path if present.")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--skip_48h", action="store_true")
    ap.add_argument("--event_top_k", type=int, default=3)
    ap.add_argument("--event_window_hours", type=int, default=3)
    ap.add_argument("--event_min_fog_stations", type=int, default=80)
    ap.add_argument("--event_min_regions", type=int, default=3)
    ap.add_argument("--event_min_lon_span", type=float, default=10.0)
    ap.add_argument("--event_min_lat_span", type=float, default=4.0)
    ap.add_argument("--event_gap_hours", type=int, default=24)
    ap.add_argument("--overlap_script", default="")
    ap.add_argument("--overlap_out_dir", default="")
    ap.add_argument("--overlap_extra_args", default="", help="Extra args passed verbatim to overlap evaluator.")
    ap.add_argument("--skip_overlap_bootstrap", action="store_true")
    ap.add_argument(
        "--local_time_offset_hours",
        type=int,
        default=LOCAL_TIME_OFFSET_HOURS,
        help="Offset applied to UTC timestamps for diurnal plots; default 8 converts UTC to UTC+8.",
    )
    ap.add_argument("--history_paths", default="", help="Comma/semicolon-separated training history JSON or stdout .out/.log files.")
    ap.add_argument("--history_labels", default="", help="Optional labels matching --history_paths order.")
    return ap.parse_args()


def abs_under_base(base: Path, path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return base / p


def resolve_model_py(base: Path, explicit: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(abs_under_base(base, explicit))
    candidates.extend(
        [
            base / "PMST_net_test_11_s2_pm10.py",
            base / "vis_mlp" / "PMST_net_test_11_s2_pm10.py",
            base / "train" / "PMST_net_test_11_s2_pm10.py",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("Cannot find PMST_net_test_11_s2_pm10.py; pass --model_py.")


def require_existing_path(path: Path, label: str, expect_dir: bool = False) -> None:
    ok = path.is_dir() if expect_dir else path.is_file()
    if not ok:
        kind = "directory" if expect_dir else "file"
        raise FileNotFoundError(f"{label} {kind} not found: {path}")


def infer_layout_from_x(x_path: Path, window_size: int) -> Tuple[int, int]:
    shape = np.load(x_path, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{x_path} must be 2D [N,D], got {shape}")
    total_dim = int(shape[1])
    rest = total_dim - 6
    for dyn in (27, 26, 25, 24):
        fe = rest - dyn * int(window_size)
        if 20 <= fe <= 64:
            return dyn, fe
    raise ValueError(f"Cannot infer dyn/FE layout from {x_path}: total_dim={total_dim}")


def import_model_class(model_py: Path):
    spec = importlib.util.spec_from_file_location("pmst_model_for_journal_eval", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ImprovedDualStreamPMSTNet


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ---------------------------------------------------------------------------
# Labels, metrics, inference
# ---------------------------------------------------------------------------


def visibility_to_class(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_raw, dtype=np.float32).copy()
    finite = np.isfinite(y)
    if finite.any() and np.nanmax(y[finite]) < 100.0:
        y *= 1000.0
    cls = np.full(len(y), 2, dtype=np.int64)
    cls[y < 1000.0] = 1
    cls[y < 500.0] = 0
    cls[~np.isfinite(y)] = -1
    return cls, y


def pred_from_probs_rule(probs: np.ndarray, fog_th: float, mist_th: float, rule: str) -> np.ndarray:
    if rule == "joint":
        return pred_from_joint_thresholds(probs, fog_th, mist_th)
    if rule == "mutual":
        return pred_from_thresholds_mutual(probs, fog_th, mist_th)
    return pred_from_thresholds(probs, fog_th, mist_th)


def dyn_log_indices(dyn_vars_count: int) -> List[int]:
    idxs = [2, 4, 9]
    if dyn_vars_count >= 27:
        idxs.extend([dyn_vars_count - 2, dyn_vars_count - 1])
    else:
        idxs.append(dyn_vars_count - 1)
    return [i for i in idxs if 0 <= i < dyn_vars_count]


def prepare_batch_rows(
    rows: np.ndarray,
    scaler,
    window_size: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
) -> np.ndarray:
    split_dyn = window_size * dyn_vars_count
    log_mask = np.zeros(split_dyn, dtype=bool)
    for t in range(window_size):
        for i in dyn_log_indices(dyn_vars_count):
            log_mask[t * dyn_vars_count + i] = True

    feats = rows[:, : split_dyn + 5].astype(np.float32, copy=True)
    feats[:, :split_dyn] = np.where(
        log_mask,
        np.log1p(np.maximum(feats[:, :split_dyn], 0.0)),
        feats[:, :split_dyn],
    )
    if scaler is not None:
        if len(scaler.center_) != feats.shape[1]:
            raise ValueError(
                f"Scaler dimension {len(scaler.center_)} does not match feature block {feats.shape[1]}"
            )
        feats = (feats - scaler.center_) / (scaler.scale_ + 1e-6)
    veg = rows[:, split_dyn + 5 : split_dyn + 6].astype(np.float32, copy=False)
    extra = rows[:, split_dyn + 6 :].astype(np.float32, copy=True)
    if extra.shape[1] < extra_feat_dim:
        extra = np.pad(extra, ((0, 0), (0, extra_feat_dim - extra.shape[1])), mode="constant")
    elif extra.shape[1] > extra_feat_dim:
        extra = extra[:, :extra_feat_dim]

    final = np.concatenate([np.clip(feats, -10, 10), veg, np.clip(extra, -10, 10)], axis=1)
    return np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def run_model_inference(
    x_path: Path,
    scaler,
    model,
    device,
    batch_size: int,
    window_size: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    limit_samples: int = 0,
    temperature: Optional[float] = None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    X = np.load(x_path, mmap_mode="r")
    n = len(X) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(X))
    temp = 1.0 if temperature is None else max(float(temperature), 1e-6)
    out = []
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = prepare_batch_rows(rows, scaler, window_size, dyn_vars_count, extra_feat_dim)
        x = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits = model(x)[0]
            probs = F.softmax(logits / temp, dim=1)
        out.append(probs.detach().cpu().numpy())
        if start == 0 or end == n or (start // max(batch_size, 1)) % 20 == 0:
            print(f"  [inference] {end}/{n}", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def load_checkpoint_into_model(model, ckpt_path: Path, device) -> None:
    import torch

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object from {ckpt_path}: {type(state)}")
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing keys: {len(missing)} first={missing[:5]}", flush=True)
    if unexpected:
        print(f"  [ckpt] unexpected keys: {len(unexpected)} first={unexpected[:5]}", flush=True)


def confusion_counts(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    valid = (y_true >= 0) & (y_true <= 2) & (pred >= 0) & (pred <= 2)
    return confusion_matrix(y_true[valid], pred[valid], labels=[0, 1, 2])


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def classification_metrics(y_true: np.ndarray, pred: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    cm = confusion_counts(y_true, pred)
    n = int(cm.sum())
    d: Dict[str, float] = {"n": float(n)}
    for cid, cname in enumerate(("Fog", "Mist", "Clear")):
        tp = float(cm[cid, cid])
        fp = float(cm[:, cid].sum() - cm[cid, cid])
        fn = float(cm[cid, :].sum() - cm[cid, cid])
        support = float(cm[cid, :].sum())
        pred_count = float(cm[:, cid].sum())
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        csi = safe_div(tp, tp + fp + fn)
        far = safe_div(fp, tp + fp)
        prefix = cname if cname != "Clear" else "Clear"
        d[f"{prefix}_P"] = p
        d[f"{prefix}_R"] = r
        d[f"{prefix}_CSI"] = csi
        d[f"{prefix}_FAR"] = far
        d[f"{prefix}_support"] = support
        d[f"pred_{prefix.lower()}"] = pred_count
    true_low = y_true <= 1
    pred_low = pred <= 1
    true_clear = y_true == 2
    low_tp = float((true_low & pred_low).sum())
    low_fp = float((~true_low & pred_low).sum())
    low_fn = float((true_low & ~pred_low).sum())
    low_r = safe_div(low_tp, low_tp + low_fn)
    d["low_vis_csi"] = safe_div(low_tp, low_tp + low_fp + low_fn)
    d["low_vis_recall"] = low_r
    d["false_positive_rate"] = safe_div(float((true_clear & pred_low).sum()), float(true_clear.sum()))
    d["accuracy"] = safe_div(float(np.trace(cm)), float(n))
    if probs is not None and len(probs) == len(y_true):
        try:
            rare_metrics = compute_rare_event_report(probs, y_true, pred=pred)
            drop_keys = {
                k
                for k in rare_metrics
                if k.endswith("_F1")
                or k.endswith("_F2")
                or k.endswith("_POD")
                or k in {"macro_f1", "weighted_f1", "low_vis_precision", "low_vis_f1", "low_vis_pod"}
            }
            d.update({k: v for k, v in rare_metrics.items() if k not in drop_keys})
        except Exception as exc:
            print(f"  [WARN] probability metrics skipped: {exc}", flush=True)
    return d


def metrics_rows(rows: Sequence[Tuple[str, Dict[str, float], Dict[str, object]]]) -> pd.DataFrame:
    out = []
    for source, metrics, extra in rows:
        row = {"source": source}
        row.update(extra)
        row.update(metrics)
        out.append(row)
    return pd.DataFrame(out)


def metric_deltas(
    a: Dict[str, float],
    b: Dict[str, float],
    a_name: str,
    b_name: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    lower_is_better = {"Fog_FAR", "Mist_FAR", "Clear_FAR", "false_positive_rate", "Brier_Fog", "Brier_Mist", "ECE"}
    rows = []
    for m in metrics:
        if m not in a or m not in b:
            continue
        av = float(a[m])
        bv = float(b[m])
        delta = av - bv
        direction = "lower" if m in lower_is_better else "higher"
        better = delta < 0 if direction == "lower" else delta > 0
        rows.append(
            {
                "metric": m,
                f"{a_name}": av,
                f"{b_name}": bv,
                f"delta_{a_name}_minus_{b_name}": delta,
                "preferred_direction": direction,
                f"{a_name}_better": bool(better),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading and IFS matching
# ---------------------------------------------------------------------------


def load_main_data(data_dir: Path, limit_samples: int = 0) -> Tuple[Path, np.ndarray, np.ndarray, pd.DataFrame]:
    x_path = data_dir / "X_test.npy"
    y_path = data_dir / "y_test.npy"
    meta_path = data_dir / "meta_test.csv"
    for p in (x_path, y_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")
    y_raw_all = np.load(y_path)
    n = len(y_raw_all) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(y_raw_all))
    y_cls, y_raw = visibility_to_class(y_raw_all[:n])
    meta = pd.read_csv(meta_path)
    if len(meta) < n:
        raise ValueError(f"{meta_path} has {len(meta)} rows, expected at least {n}")
    meta = meta.iloc[:n].copy().reset_index(drop=True)
    meta["time"] = pd.to_datetime(meta["time"], errors="coerce")
    if "hour" not in meta:
        meta["hour"] = meta["time"].dt.hour
    meta["hour_utc"] = meta["time"].dt.hour
    meta = add_local_time_columns(meta)
    if "month" not in meta:
        meta["month"] = meta["time"].dt.month
    return x_path, y_cls, y_raw, meta


def add_local_time_columns(df: pd.DataFrame, offset_hours: int = LOCAL_TIME_OFFSET_HOURS) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["time_analysis"] = out["time"] + pd.to_timedelta(int(offset_hours), unit="h")
    out["hour_analysis"] = out["time_analysis"].dt.hour
    out["month_analysis"] = out["time_analysis"].dt.month
    return out


def normalize_station_ids(values) -> pd.Series:
    s = pd.Series(values)
    numeric = pd.to_numeric(s, errors="coerce")
    out = s.astype(str)
    m = numeric.notna()
    if m.any():
        out.loc[m] = numeric.loc[m].astype(np.int64).astype(str)
    return out


def classify_visibility_values(vis: np.ndarray) -> np.ndarray:
    arr = np.asarray(vis, dtype=np.float64)
    cls = np.full(arr.shape, 2, dtype=np.int64)
    cls[arr < 1000.0] = 1
    cls[arr < 500.0] = 0
    cls[~np.isfinite(arr)] = -1
    return cls


def load_ifs_diagnostic(
    meta: pd.DataFrame,
    ifs_nc_path: Path,
    vis_var: str = "VIS",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not ifs_nc_path.exists():
        raise FileNotFoundError(f"IFS diagnostic NetCDF not found: {ifs_nc_path}")
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to match IFS diagnostic visibility.") from exc

    ds = xr.open_dataset(ifs_nc_path)
    try:
        if vis_var not in ds:
            raise KeyError(f"Variable {vis_var!r} not found in {ifs_nc_path}; vars={list(ds.data_vars)}")
        station_coord = "station" if "station" in ds.coords else "station_id"
        if "time" not in ds.coords or station_coord not in ds.coords:
            raise KeyError(f"IFS NetCDF must have coords time and station/station_id: {ifs_nc_path}")

        da = ds[vis_var].squeeze()
        if "time" not in da.dims or station_coord not in da.dims:
            raise ValueError(f"{vis_var} must have time and station dims, got dims={da.dims}")
        da = da.transpose("time", station_coord, ...)
        if da.ndim != 2:
            raise ValueError(f"{vis_var} must be 2D after squeeze, got shape={da.shape}")
        vis = np.asarray(da.values, dtype=np.float64)
        times = pd.to_datetime(ds["time"].values)
        stations = pd.Index(normalize_station_ids(ds[station_coord].values))

        time_lookup = pd.Series(np.arange(len(times), dtype=np.int64), index=pd.Index(times))
        station_lookup = pd.Series(np.arange(len(stations), dtype=np.int64), index=stations)

        meta_times = pd.to_datetime(meta["time"], errors="coerce")
        meta_station = normalize_station_ids(meta["station_id"].values)
        time_idx = meta_times.map(time_lookup)
        station_idx = meta_station.map(station_lookup)
        key_valid = time_idx.notna() & station_idx.notna()

        raw = np.full(len(meta), np.nan, dtype=np.float64)
        pred = np.full(len(meta), -1, dtype=np.int64)
        valid = np.zeros(len(meta), dtype=bool)
        if key_valid.any():
            pos = np.flatnonzero(key_valid.to_numpy())
            ti = time_idx.iloc[pos].astype(np.int64).to_numpy()
            si = station_idx.iloc[pos].astype(np.int64).to_numpy()
            matched = vis[ti, si]
            finite = np.isfinite(matched)
            pos_f = pos[finite]
            raw[pos_f] = matched[finite]
            pred[pos_f] = classify_visibility_values(matched[finite])
            valid[pos_f] = True
        print(f"[IFS] matched finite rows: {int(valid.sum())}/{len(meta)} from {ifs_nc_path}", flush=True)
        return pred, raw, valid
    finally:
        ds.close()


def export_per_sample(
    out_path: Path,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_raw: np.ndarray,
    pred: np.ndarray,
    probs: np.ndarray,
    ifs_pred: Optional[np.ndarray] = None,
    ifs_vis: Optional[np.ndarray] = None,
    ifs_valid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "station_id",
            "lat",
            "lon",
            "time",
            "time_analysis",
            "month",
            "month_analysis",
            "hour",
            "hour_utc",
            "hour_analysis",
            "init_time",
            "init_hour",
            "lead_hour",
        ]
        if c in meta
    ]
    df = meta[cols].copy()
    df["y_true"] = y_true
    df["vis_raw_m"] = y_raw
    df["pmst_pred"] = pred
    df["pmst_p_fog"] = probs[:, 0]
    df["pmst_p_mist"] = probs[:, 1]
    df["pmst_p_clear"] = probs[:, 2]
    df["pmst_correct"] = pred == y_true
    if ifs_pred is not None:
        df["ifs_diagnostic_valid"] = ifs_valid
        df["ifs_diagnostic_vis_m"] = ifs_vis
        df["ifs_diagnostic_pred"] = ifs_pred
        df["ifs_diagnostic_correct"] = np.asarray(ifs_valid, dtype=bool) & (ifs_pred == y_true)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[table] {out_path} rows={len(df)}", flush=True)
    return df


def write_report(path: Path, y_true: np.ndarray, pred: np.ndarray, metrics: Dict[str, float],
                 ifs_metrics: Optional[Dict[str, float]] = None) -> None:
    def _write_metric_block(f, title: str, values: Dict[str, float]) -> None:
        f.write(f"{title}\n")
        f.write("  class, support, predicted, CSI, recall, precision, FAR\n")
        for cname in ("Fog", "Mist", "Clear"):
            f.write(
                "  "
                f"{cname}, "
                f"{values.get(f'{cname}_support', 0.0):.0f}, "
                f"{values.get(f'pred_{cname.lower()}', 0.0):.0f}, "
                f"{values.get(f'{cname}_CSI', np.nan):.6f}, "
                f"{values.get(f'{cname}_R', np.nan):.6f}, "
                f"{values.get(f'{cname}_P', np.nan):.6f}, "
                f"{values.get(f'{cname}_FAR', np.nan):.6f}\n"
            )
        f.write(f"  low_vis_csi: {values.get('low_vis_csi', np.nan):.6f}\n")
        f.write(f"  low_vis_recall: {values.get('low_vis_recall', np.nan):.6f}\n")
        f.write(f"  clear_to_low_vis_false_positive_rate: {values.get('false_positive_rate', np.nan):.6f}\n")
        f.write(f"  accuracy: {values.get('accuracy', np.nan):.6f}\n")
        for key in ("balanced_acc", "mcc", "Brier_Fog", "Brier_Mist", "ECE"):
            if key in values:
                f.write(f"  {key}: {float(values[key]):.6f}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("Class definitions:\n")
        f.write("  0: 0 <= visibility < 500 m\n")
        f.write("  1: 500 <= visibility < 1000 m\n")
        f.write("  2: visibility >= 1000 m\n\n")
        _write_metric_block(f, "PMST class metrics (CSI/recall primary):", metrics)
        if ifs_metrics is not None:
            f.write("\n")
            _write_metric_block(f, "IFS diagnostic visibility metrics on matched rows:", ifs_metrics)
    print(f"[report] {path}", flush=True)


# ---------------------------------------------------------------------------
# Scenario and station products
# ---------------------------------------------------------------------------


def add_scenario_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_local_time_columns(df)
    if "hour" not in out:
        out["hour"] = out["time"].dt.hour
    if "hour_utc" not in out:
        out["hour_utc"] = out["time"].dt.hour
    if "month" not in out:
        out["month"] = out["time"].dt.month
    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    season_month = out["month_analysis"] if "month_analysis" in out else out["month"]
    out["season"] = season_month.map(season_map)
    h = out["hour_analysis"].astype(int)
    out["time_of_day"] = np.select(
        [h.between(0, 5), h.between(6, 11), h.between(12, 17), h.between(18, 23)],
        TIME_OF_DAY_LOCAL_ORDER,
        default="Unknown",
    )
    region = np.full(len(out), "Other", dtype=object)
    lats = out["lat"].astype(float).to_numpy()
    lons = out["lon"].astype(float).to_numpy()
    for name, lat_min, lat_max, lon_min, lon_max in REGION_DEFS:
        m = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
        region[m] = name
    out["region"] = region
    if "init_time" in out and "init_hour" not in out:
        init_dt = pd.to_datetime(out["init_time"], errors="coerce")
        out["init_hour"] = init_dt.dt.hour
    return out


def build_scenario_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = add_scenario_columns(eval_df)
    rows = []
    specs = [
        ("All", np.ones(len(df), dtype=bool)),
    ]
    for col, values in (
        ("time_of_day", TIME_OF_DAY_LOCAL_ORDER),
        ("season", ["DJF", "MAM", "JJA", "SON"]),
        ("region", sorted(df["region"].dropna().unique())),
    ):
        for val in values:
            specs.append((f"{col}:{val}", (df[col] == val).to_numpy()))

    for scenario, mask in specs:
        if int(mask.sum()) < 50:
            continue
        y = df.loc[mask, "y_true"].to_numpy(dtype=np.int64)
        pmst = df.loc[mask, "pmst_pred"].to_numpy(dtype=np.int64)
        rows.append({"source": "pmst", "scenario": scenario, **classification_metrics(y, pmst)})
        if "ifs_diagnostic_pred" in df:
            vm = mask & df["ifs_diagnostic_valid"].to_numpy(dtype=bool)
            if int(vm.sum()) >= 50:
                yi = df.loc[vm, "y_true"].to_numpy(dtype=np.int64)
                ip = df.loc[vm, "ifs_diagnostic_pred"].to_numpy(dtype=np.int64)
                rows.append({"source": "ifs_diagnostic", "scenario": scenario, **classification_metrics(yi, ip)})
    return pd.DataFrame(rows)


def aggregate_station_metrics(eval_df: pd.DataFrame, pred_col: str = "pmst_pred") -> pd.DataFrame:
    rows = []
    for sid, sub in eval_df.groupby("station_id", sort=False):
        y = sub["y_true"].to_numpy(dtype=np.int64)
        p = sub[pred_col].to_numpy(dtype=np.int64)
        n_fog = int((y == 0).sum())
        n_mist = int((y == 1).sum())
        n_clear = int((y == 2).sum())
        pred_low = p <= 1
        row = {
            "station_id": sid,
            "lat": float(sub["lat"].iloc[0]),
            "lon": float(sub["lon"].iloc[0]),
            "n_total": int(len(sub)),
            "n_fog": n_fog,
            "n_mist": n_mist,
            "n_clear": n_clear,
            "fog_recall": safe_div(float(((y == 0) & (p == 0)).sum()), float(n_fog)),
            "mist_recall": safe_div(float(((y == 1) & (p == 1)).sum()), float(n_mist)),
            "fpr_fog": safe_div(float(((y == 2) & pred_low).sum()), float(n_clear)),
            "overall_acc": float((y == p).mean()) if len(sub) else math.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_confusion_pmst_vs_ifs(
    y_true: np.ndarray,
    pmst_pred: np.ndarray,
    ifs_pred: Optional[np.ndarray],
    ifs_valid: Optional[np.ndarray],
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> None:
    setup_journal_style()
    if ifs_pred is not None and ifs_valid is not None and int(np.sum(ifs_valid)) > 0:
        y = y_true[ifs_valid]
        pmst = pmst_pred[ifs_valid]
        ifs = ifs_pred[ifs_valid]
        panels = [("PMST", pmst), ("IFS diagnostic VIS", ifs)]
        matched = int(np.sum(ifs_valid))
    else:
        y = y_true
        panels = [("PMST", pmst_pred)]
        matched = None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False)
    for ax, (title, pred) in zip(axes.ravel(), panels):
        cm = confusion_counts(y, pred)
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Observed class")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                txt = f"{cm_norm[i, j]:.2f}\n{cm[i, j]:,}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Row-normalized fraction")
    save_fig_pair(
        fig,
        out_dir,
        "fig3_confusion_matrix_pmst_vs_ifs_diagnostic",
        manifest,
        sources,
        notes="Row-normalized confusion matrix; IFS panel uses matched finite IFS VIS rows.",
        n=len(y),
        matched_ifs=matched,
    )


def plot_csi_recall_pmst_vs_ifs(
    pmst_metrics: Dict[str, float],
    ifs_metrics: Optional[Dict[str, float]],
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    n: int,
    matched_ifs: Optional[int],
) -> None:
    setup_journal_style()
    panels = [
        ("Fog", [("Fog_CSI", "CSI"), ("Fog_R", "Recall")]),
        ("Mist", [("Mist_CSI", "CSI"), ("Mist_R", "Recall")]),
        ("Low visibility", [("low_vis_csi", "CSI"), ("low_vis_recall", "Recall")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharey=False)
    for ax_idx, (ax, (group, sub)) in enumerate(zip(axes, panels)):
        x = np.arange(len(sub))
        width = 0.32
        pmst_vals = [float(pmst_metrics.get(k, np.nan)) for k, _ in sub]
        ax.bar(x - width / 2, pmst_vals, width, label="PMST", color=PMST_COLOR, edgecolor="white", linewidth=0.5)
        if ifs_metrics is not None:
            ifs_vals = [float(ifs_metrics.get(k, np.nan)) for k, _ in sub]
            ax.bar(x + width / 2, ifs_vals, width, label="IFS diagnostic", color=IFS_DIAG_COLOR, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in sub], rotation=20, ha="right")
        ax.set_title(group)
        ax.set_ylim(0, 1.05)
        if ax_idx == 0:
            ax.set_ylabel("Score")
        add_panel_label(ax, "abc"[ax_idx])
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig3_csi_recall_pmst_vs_ifs_diagnostic",
        manifest,
        sources,
        notes="CSI and recall are shown as the primary rare-event metrics for Fog, Mist, and combined low visibility.",
        n=n,
        matched_ifs=matched_ifs,
    )


def plot_ifs_visibility_bias(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    ifs_vis: np.ndarray,
    ifs_valid: np.ndarray,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> None:
    if ifs_vis is None or ifs_valid is None or int(np.sum(ifs_valid)) == 0:
        return
    setup_journal_style()
    m = ifs_valid & np.isfinite(ifs_vis) & np.isfinite(y_raw)
    low = m & (y_true <= 1)
    if int(low.sum()) == 0:
        return
    diff = np.clip(ifs_vis[low] - y_raw[low], -2000, 20000)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    axes[0].hist(diff, bins=50, color=IFS_DIAG_COLOR, alpha=0.88)
    axes[0].axvline(0, color="#111111", lw=1.2)
    axes[0].set_xlabel("IFS VIS - observed VIS (m), true low-vis cases")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("IFS visibility bias in low-vis observations")
    add_panel_label(axes[0], "a")

    data = []
    labels = []
    for cls, label in [(0, "Fog obs"), (1, "Mist obs")]:
        mm = m & (y_true == cls)
        if int(mm.sum()) > 0:
            data.append(np.clip(ifs_vis[mm], 0, 20000))
            labels.append(label)
    axes[1].boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                    boxprops={"facecolor": "#D9D9D9", "color": "#555555"},
                    medianprops={"color": PMST_COLOR, "linewidth": 1.4})
    axes[1].axhline(500, color=FOG_COLOR, ls="--", lw=1.0, label="500 m")
    axes[1].axhline(1000, color=MIST_COLOR, ls="--", lw=1.0, label="1000 m")
    axes[1].set_ylabel("IFS diagnostic visibility (m)")
    axes[1].set_title("IFS VIS distribution by observed class")
    axes[1].legend(frameon=False)
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig3c_ifs_visibility_bias_lowvis",
        manifest,
        sources,
        notes="Diagnostic plot for IFS visibility overestimation in observed low-visibility cases.",
        n=int(low.sum()),
        matched_ifs=int(m.sum()),
    )


def plot_scenario_split(
    scenario_df: pd.DataFrame,
    split: str,
    order: Sequence[str],
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> None:
    setup_journal_style()
    src_df = scenario_df[scenario_df["source"] == "pmst"].copy()
    rows = []
    for name in order:
        sc = f"{split}:{name}"
        sub = src_df[src_df["scenario"] == sc]
        if sub.empty:
            continue
        r = sub.iloc[0].to_dict()
        r["name"] = name
        rows.append(r)
    if not rows:
        print(f"  [WARN] No scenario rows for split={split}", flush=True)
        return
    df = pd.DataFrame(rows)
    metrics = [
        ("Fog_CSI", "Fog CSI"),
        ("Fog_R", "Fog recall"),
        ("Mist_CSI", "Mist CSI"),
        ("Mist_R", "Mist recall"),
        ("low_vis_recall", "Low-vis recall"),
    ]
    colors = [FOG_COLOR, "#6E91B5", MIST_COLOR, "#F0B84A", "#4B5563"]
    x = np.arange(len(df))
    width = min(0.16, 0.80 / len(metrics))
    fig, ax = plt.subplots(figsize=(max(7.5, 0.78 * len(df)), 4.2))
    for i, ((key, label), color) in enumerate(zip(metrics, colors)):
        vals = df[key].astype(float).to_numpy()
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width * 0.94, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=25 if len(df) > 4 else 0, ha="right" if len(df) > 4 else "center")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    title = {"time_of_day": f"Metrics by time of day ({LOCAL_TIME_LABEL})", "season": "Metrics by season", "region": "Metrics by region"}.get(split, split)
    ax.set_title(title)
    ax.legend(ncol=min(5, len(metrics)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    stem = {
        "time_of_day": "fig7_split_time_of_day",
        "season": "fig7_split_season",
        "region": "fig7_split_region",
    }[split]
    save_fig_pair(fig, out_dir, stem, manifest, sources, notes=f"PMST scenario split: {split}.", n=int(df["n"].sum()))


def _group_metric_rows(eval_df: pd.DataFrame, group_col: str, group_order: Sequence[object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for value in group_order:
        sub = eval_df[eval_df[group_col] == value]
        if sub.empty:
            continue
        y = sub["y_true"].to_numpy(dtype=np.int64)
        p = sub["pmst_pred"].to_numpy(dtype=np.int64)
        metrics = classification_metrics(y, p)
        fog_count = int(np.sum(y == 0))
        mist_count = int(np.sum(y == 1))
        low_count = fog_count + mist_count
        row: Dict[str, object] = {
            group_col: value,
            "n": int(len(sub)),
            "fog_count": fog_count,
            "mist_count": mist_count,
            "low_vis_count": low_count,
            "low_vis_rate": safe_div(float(low_count), float(len(sub))),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_diurnal_time_detail(
    eval_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    offset_hours: int = LOCAL_TIME_OFFSET_HOURS,
) -> Optional[Path]:
    setup_journal_style()
    df = add_local_time_columns(eval_df, offset_hours)
    df = df[np.isfinite(df["hour_analysis"])].copy()
    df["hour_analysis"] = df["hour_analysis"].astype(int)
    table = _group_metric_rows(df, "hour_analysis", list(range(24)))
    if table.empty:
        print(f"  [WARN] No rows for diurnal {LOCAL_TIME_LABEL} detail figure.", flush=True)
        return None
    table_path = out_dir / "fig11_diurnal_bjt_metrics_counts.csv"
    table.to_csv(table_path, index=False, float_format="%.6f")
    print(f"[table] {table_path}", flush=True)

    x = table["hour_analysis"].to_numpy(dtype=int)
    fog_k = table["fog_count"].to_numpy(dtype=float) / 1000.0
    mist_k = table["mist_count"].to_numpy(dtype=float) / 1000.0
    low_rate = table["low_vis_rate"].to_numpy(dtype=float) * 100.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.15},
    )
    ax = axes[0]
    ax.bar(x, fog_k, width=0.82, color=FOG_COLOR, label="Observed Fog")
    ax.bar(x, mist_k, bottom=fog_k, width=0.82, color=MIST_COLOR, label="Observed Mist")
    ax.set_ylabel("Samples (x1000)")
    ax.set_title(f"Diurnal low-visibility frequency and PMST skill ({LOCAL_TIME_LABEL})")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(x, low_rate, color="#2A9D8F", lw=1.8, marker="o", ms=3.2, label="Low-vis rate")
    ax2.set_ylabel("Low-vis rate (%)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left")
    add_panel_label(ax, "a")

    metric_specs = [
        ("Fog_CSI", "Fog CSI", FOG_COLOR),
        ("Fog_R", "Fog recall", "#6E91B5"),
        ("Mist_CSI", "Mist CSI", MIST_COLOR),
        ("Mist_R", "Mist recall", "#F0B84A"),
        ("low_vis_recall", "Low-vis recall", "#4B5563"),
    ]
    ax = axes[1]
    for key, label, color in metric_specs:
        ax.plot(x, table[key].to_numpy(dtype=float), lw=1.8, marker="o", ms=3.0, color=color, label=label)
    ax.set_xticks(np.arange(0, 24, 1))
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel(f"Hour ({LOCAL_TIME_LABEL})")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.10)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    add_panel_label(ax, "b")
    for left, right in ((0, 6), (18, 24)):
        for a in axes:
            a.axvspan(left - 0.5, right - 0.5, color="#F3F4F6", alpha=0.55, zorder=-10)
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig11_diurnal_bjt_performance_counts",
        manifest,
        list(sources) + [str(table_path)],
        notes="Hourly PMST skill and observed Fog/Mist counts after converting UTC timestamps to UTC+8.",
        n=int(table["n"].sum()),
    )
    return table_path


def plot_region_detail(
    eval_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> Optional[Path]:
    setup_journal_style()
    df = add_scenario_columns(eval_df)
    order = [r[0] for r in REGION_DEFS] + ["Other"]
    table = _group_metric_rows(df, "region", order)
    if table.empty:
        print("  [WARN] No rows for region detail figure.", flush=True)
        return None
    table_path = out_dir / "fig12_region_metrics_rates.csv"
    table.to_csv(table_path, index=False, float_format="%.6f")
    print(f"[table] {table_path}", flush=True)

    y = np.arange(len(table))
    labels = table["region"].astype(str).tolist()
    low_rate = table["low_vis_rate"].to_numpy(dtype=float) * 100.0
    low_recall = table["low_vis_recall"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.50 * len(table) + 1.6)))
    ax.barh(y, low_rate, color=FOG_COLOR, alpha=0.88, label="Low-vis time share")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Observed low-visibility time share (%)")
    ax.set_title("Regional low-visibility occurrence and PMST recall")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    ax2 = ax.twiny()
    ax2.plot(low_recall, y, color="#4B5563", marker="o", lw=2.0, ms=4.0, label="Low-vis recall")
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Low-vis recall")
    ax2.grid(False)

    handles, legend_labels = ax.get_legend_handles_labels()
    handles2, legend_labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, legend_labels + legend_labels2, frameon=False, loc="lower right")

    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig12_region_performance_counts",
        manifest,
        list(sources) + [str(table_path)],
        notes="Regional low-visibility time share is station-count-normalized by regional sample totals; the overlaid metric is low-visibility recall.",
        n=int(table["n"].sum()),
    )
    return table_path


def plot_time_of_day_detail(
    eval_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    offset_hours: int = LOCAL_TIME_OFFSET_HOURS,
) -> Optional[Path]:
    setup_journal_style()
    df = add_local_time_columns(eval_df, offset_hours)
    df = df[np.isfinite(df["hour_analysis"])].copy()
    h = df["hour_analysis"].astype(int)
    df["time_of_day"] = np.select(
        [h.between(0, 5), h.between(6, 11), h.between(12, 17), h.between(18, 23)],
        TIME_OF_DAY_LOCAL_ORDER,
        default="Unknown",
    )
    table = _group_metric_rows(df, "time_of_day", TIME_OF_DAY_LOCAL_ORDER)
    if table.empty:
        print("  [WARN] No rows for time-of-day detail figure.", flush=True)
        return None
    table_path = out_dir / "fig12_time_of_day_metrics_rates.csv"
    table.to_csv(table_path, index=False, float_format="%.6f")
    print(f"[table] {table_path}", flush=True)

    y = np.arange(len(table))
    labels = table["time_of_day"].astype(str).tolist()
    low_rate = table["low_vis_rate"].to_numpy(dtype=float) * 100.0
    low_recall = table["low_vis_recall"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, max(3.8, 0.60 * len(table) + 1.6)))
    ax.barh(y, low_rate, color=FOG_COLOR, alpha=0.88, label="Low-vis time share")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Observed low-visibility time share (%)")
    ax.set_title(f"Time-of-day low-visibility occurrence and PMST recall ({LOCAL_TIME_LABEL})")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    ax2 = ax.twiny()
    ax2.plot(low_recall, y, color="#4B5563", marker="o", lw=2.0, ms=4.0, label="Low-vis recall")
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Low-vis recall")
    ax2.grid(False)

    handles, legend_labels = ax.get_legend_handles_labels()
    handles2, legend_labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, legend_labels + legend_labels2, frameon=False, loc="lower right")

    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig12_time_of_day_performance_counts",
        manifest,
        list(sources) + [str(table_path)],
        notes="Time-of-day low-visibility time share uses UTC timestamps converted to UTC+8; the overlaid metric is low-visibility recall.",
        n=int(table["n"].sum()),
    )
    return table_path


def discover_history_paths(base: Path, out_dir: Path, explicit: str = "") -> List[Path]:
    if explicit.strip():
        tokens = [t.strip() for chunk in explicit.split(";") for t in chunk.split(",")]
        return [abs_under_base(base, t) for t in tokens if t]
    roots = [out_dir, base / "checkpoints", base]
    seen = set()
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*history.json"):
            key = str(p.resolve())
            if key not in seen:
                paths.append(p)
                seen.add(key)
    return paths


def _history_variant_label(path: Path) -> str:
    name = path.name
    if name in KNOWN_CONVERGENCE_LOG_LABELS:
        return KNOWN_CONVERGENCE_LOG_LABELS[name]
    if "feabl_no_fe_all" in name:
        return "No FE values"
    if "feabl_" in name:
        start = name.find("feabl_")
        chunk = name[start:].split("_S2_", 1)[0]
        return chunk.replace("feabl_", "FE ablation: ").replace("_", " ")
    return "Full FE"


def _split_cli_list(value: str) -> List[str]:
    return [t.strip() for chunk in str(value or "").split(";") for t in chunk.split(",") if t.strip()]


def _history_phase_order(path: Path) -> int:
    name = path.name
    if "S2_PhaseA1" in name:
        return 0
    if "S2_PhaseA2" in name:
        return 1
    if "S2_PhaseB" in name:
        return 2
    return 99


def parse_stdout_loss_history(path: Path, variant: str) -> pd.DataFrame:
    pattern = re.compile(
        r"\[(S2_PhaseA1|S2_PhaseA2|S2_PhaseB)\]\s+Step\s+(\d+)\s*/\s*(\d+)\s+\|\s+Loss=([0-9.eE+-]+)"
    )
    rows: List[Dict[str, object]] = []
    phase_offsets = {"S2_PhaseA1": 0, "S2_PhaseA2": 15000, "S2_PhaseB": 30000}
    phase_names = {"S2_PhaseA1": "A1", "S2_PhaseA2": "A2", "S2_PhaseB": "B"}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                match = pattern.search(raw.replace("\r", "\n"))
                if not match:
                    continue
                phase, step_s, total_s, loss_s = match.groups()
                step = int(step_s)
                total = int(total_s)
                phase_offset = phase_offsets.get(phase, 0)
                # If the phase length differs from the standard 15k/15k/30k schedule,
                # keep the x-axis monotonic while preserving the phase-local step.
                if phase == "S2_PhaseA2":
                    phase_offset = max(phase_offset, total)
                elif phase == "S2_PhaseB":
                    phase_offset = max(phase_offset, 2 * total if total <= 15000 else 30000)
                rows.append(
                    {
                        "variant": variant,
                        "phase": phase_names.get(phase, phase),
                        "phase_step": step,
                        "global_step": phase_offset + step,
                        "train_loss": float(loss_s),
                        "val_score": np.nan,
                        "source_history": str(path),
                    }
                )
    except Exception as exc:
        print(f"  [WARN] Cannot parse stdout log {path}: {exc}", flush=True)
    return pd.DataFrame(rows)


def build_convergence_table(history_paths: Sequence[Path], labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    phase_names = {0: "A1", 1: "A2", 2: "B", 99: "unknown"}
    records: List[Dict[str, object]] = []
    by_variant: Dict[str, List[Path]] = {}
    label_lookup = {
        str(p): str(labels[i]).strip()
        for i, p in enumerate(history_paths)
        if labels is not None and i < len(labels) and str(labels[i]).strip()
    }
    stdout_tables = []
    for p in history_paths:
        if p.exists():
            variant = label_lookup.get(str(p), _history_variant_label(p))
            if p.suffix.lower() in {".out", ".log", ".txt"}:
                stdout_tables.append(parse_stdout_loss_history(p, variant))
            else:
                by_variant.setdefault(variant, []).append(p)
    for variant, paths in by_variant.items():
        offset = 0
        for p in sorted(paths, key=_history_phase_order):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  [WARN] Cannot read history {p}: {exc}", flush=True)
                continue
            steps = payload.get("steps", [])
            train_loss = payload.get("train_loss", [])
            val_score = payload.get("val_score", [])
            n = min(len(steps), len(train_loss))
            phase = _history_phase_order(p)
            for i in range(n):
                step = int(steps[i])
                records.append(
                    {
                        "variant": variant,
                        "phase": phase_names.get(phase, "unknown"),
                        "phase_step": step,
                        "global_step": offset + step,
                        "train_loss": float(train_loss[i]),
                        "val_score": float(val_score[i]) if i < len(val_score) else np.nan,
                        "source_history": str(p),
                    }
                )
            if steps:
                offset += int(max(steps))
    tables = [pd.DataFrame(records)] if records else []
    tables.extend([t for t in stdout_tables if t is not None and not t.empty])
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True).sort_values(["variant", "global_step"]).reset_index(drop=True)


def plot_feature_convergence_from_history(
    base: Path,
    out_dir: Path,
    manifest: Manifest,
    explicit_history_paths: str = "",
    explicit_history_labels: str = "",
) -> Optional[Path]:
    paths = discover_history_paths(base, out_dir, explicit_history_paths)
    labels = _split_cli_list(explicit_history_labels)
    status_path = out_dir / "fig13_feature_convergence_history_status.csv"
    pd.DataFrame(
        [
            {
                "history_path": str(p),
                "exists": bool(p.exists()),
                "variant": labels[i] if i < len(labels) and labels[i] else _history_variant_label(p),
            }
            for i, p in enumerate(paths)
        ]
        or [{"history_path": "", "exists": False, "variant": ""}]
    ).to_csv(status_path, index=False)
    print(f"[table] {status_path}", flush=True)

    table = build_convergence_table(paths, labels=labels)
    if table.empty:
        print("  [WARN] No training history JSON found; convergence loss figure not generated.", flush=True)
        manifest.add(
            status_path.name,
            [str(status_path)],
            notes="No local training history JSON files were found for the FE convergence figure.",
        )
        return None

    table_path = out_dir / "fig13_feature_convergence_history.csv"
    table.to_csv(table_path, index=False, float_format="%.6f")
    print(f"[table] {table_path}", flush=True)

    setup_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    color_map = {"Full FE": PMST_COLOR, "No FE values": IFS_DIAG_COLOR}
    for variant, sub in table.groupby("variant", sort=False):
        color = color_map.get(variant, None)
        sub = sub.sort_values("global_step")
        axes[0].plot(sub["global_step"], sub["train_loss"], lw=1.9, label=variant, color=color)
        if sub["val_score"].notna().any():
            axes[1].plot(sub["global_step"], sub["val_score"], lw=1.9, label=variant, color=color)
    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("S2 training step")
    axes[1].set_title("Validation target score")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("S2 training step")
    for ax_idx, ax in enumerate(axes):
        ax.grid(alpha=0.25)
        add_panel_label(ax, chr(ord("a") + ax_idx))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=max(1, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig13_feature_engineering_convergence",
        manifest,
        sorted(set(table["source_history"].astype(str).tolist())),
        notes="Training-history convergence plot from S2 history JSON files.",
    )
    return table_path


def _has_boundary_object(shp_obj) -> bool:
    if shp_obj is None:
        return False
    if isinstance(shp_obj, dict):
        return bool(shp_obj.get("segments"))
    try:
        return len(shp_obj) > 0
    except Exception:
        return True


def _read_shp_segments(shp_path: Path) -> Optional[Dict[str, object]]:
    """Minimal shapefile polygon/polyline reader used only if geopandas fails."""
    try:
        import struct

        data = shp_path.read_bytes()
        if len(data) < 100:
            return None
        segments = []
        pos = 100
        while pos + 8 <= len(data):
            content_bytes = int(struct.unpack(">i", data[pos + 4 : pos + 8])[0]) * 2
            pos += 8
            rec = data[pos : pos + content_bytes]
            pos += content_bytes
            if len(rec) < 44:
                continue
            shape_type = int(struct.unpack("<i", rec[:4])[0])
            if shape_type == 0:
                continue
            if shape_type not in (3, 5, 13, 15):
                continue
            num_parts, num_points = struct.unpack("<2i", rec[36:44])
            parts_start = 44
            points_start = parts_start + 4 * int(num_parts)
            points_bytes = 16 * int(num_points)
            if num_parts <= 0 or num_points <= 0 or len(rec) < points_start + points_bytes:
                continue
            parts = list(struct.unpack(f"<{int(num_parts)}i", rec[parts_start:points_start]))
            parts.append(int(num_points))
            points = np.frombuffer(
                rec,
                dtype="<f8",
                count=int(num_points) * 2,
                offset=points_start,
            ).reshape(int(num_points), 2)
            for i in range(len(parts) - 1):
                a, b = int(parts[i]), int(parts[i + 1])
                if b > a:
                    seg = points[a:b].copy()
                    segments.append((seg[:, 0], seg[:, 1]))
        if segments:
            return {"kind": "shp_segments", "path": str(shp_path), "segments": segments}
    except Exception as exc:
        print(f"  [WARN] Pure-Python shapefile fallback failed for {shp_path}: {exc}", flush=True)
    return None


def read_shapefile(shp_path: str):
    if not shp_path:
        print("  [WARN] Empty shapefile path; maps will be drawn without boundaries.", flush=True)
        return None
    shp_file = Path(shp_path)
    if not shp_file.exists():
        print(f"  [WARN] Shapefile not found: {shp_file}; maps will be drawn without boundaries.", flush=True)
        return None

    # First follow the original paper-evaluation scripts exactly.
    if _load_china_shapefile is not None:
        shp = _load_china_shapefile(str(shp_file))
        if _has_boundary_object(shp):
            print(f"  [Map] Loaded boundary via plot_spatial.load_china_shapefile: {shp_file}", flush=True)
            return shp

    try:
        import geopandas as gpd

        shp = gpd.read_file(str(shp_file))
        if _has_boundary_object(shp):
            print(f"  [Map] Loaded boundary via geopandas.read_file: {shp_file}", flush=True)
            return shp
    except Exception as exc:
        print(f"  [WARN] Could not read shapefile with geopandas {shp_file}: {exc}", flush=True)

    shp = _read_shp_segments(shp_file)
    if _has_boundary_object(shp):
        print(f"  [Map] Loaded boundary via pure-Python .shp fallback: {shp_file}", flush=True)
        return shp
    print(f"  [WARN] Boundary loading failed for {shp_file}; maps will be drawn without boundaries.", flush=True)
    return None


def draw_boundary(ax, shp_obj=None, color: str = "#333333", linewidth: float = 0.55, zorder: int = 5) -> bool:
    if not _has_boundary_object(shp_obj):
        return False
    if isinstance(shp_obj, dict):
        for xs, ys in shp_obj.get("segments", []):
            ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)
        return True
    try:
        shp_obj.boundary.plot(ax=ax, color=color, linewidth=linewidth, zorder=zorder)
        return True
    except Exception as exc:
        print(f"  [WARN] Could not draw boundary: {exc}", flush=True)
        return False


def draw_basemap(ax, shp_gdf=None, compact: bool = False) -> None:
    if compact and _plot_spatial_event_basemap is not None and not isinstance(shp_gdf, dict):
        _plot_spatial_event_basemap(ax, shp_gdf)
        draw_boundary(ax, shp_gdf, color="#404040", linewidth=0.50, zorder=6)
        return
    draw_boundary(ax, shp_gdf, color="#333333" if not compact else "#404040", linewidth=0.55, zorder=6)
    ax.set_xlim(72, 136)
    ax.set_ylim(17, 54 if compact else 55)
    ax.set_aspect("equal", adjustable="box")
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#8A8A8A")
            spine.set_linewidth(0.6)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.18)


def plot_station_metric_map(
    station_df: pd.DataFrame,
    value_col: str,
    mask_col: str,
    min_count: int,
    title: str,
    stem: str,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    shp_gdf=None,
    cmap: str = "cividis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    setup_journal_style()
    df = station_df.copy()
    df = df[df[mask_col] >= min_count]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    draw_basemap(ax, shp_gdf)
    vals = df[value_col].to_numpy(dtype=float)
    valid = np.isfinite(vals)
    sc = ax.scatter(
        df.loc[valid, "lon"],
        df.loc[valid, "lat"],
        c=vals[valid],
        s=8,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.92,
        zorder=3,
    )
    draw_boundary(ax, shp_gdf, color="#202020", linewidth=0.55, zorder=6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.78)
    cb.set_label(value_col)
    ax.set_title(title)
    fig.tight_layout()
    save_fig_pair(fig, out_dir, stem, manifest, sources, notes=f"Station map masked by {mask_col}>={min_count}.", n=len(df))


def plot_event_peak_grid(
    eval_df: pd.DataFrame,
    event_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    shp_gdf=None,
) -> None:
    if event_df is None or event_df.empty:
        print("  [WARN] No event summary for peak grid.", flush=True)
        return
    setup_journal_style()
    events = event_df.head(3).copy()
    n_cols = len(events)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(3, n_cols, figsize=(4.1 * n_cols, 9.0), squeeze=False)
    row_specs = [
        ("Observed class", "y_true"),
        ("PMST", "pmst_pred"),
        ("IFS diagnostic", "ifs_diagnostic_pred"),
    ]
    df = eval_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col_idx, (_, ev) in enumerate(events.iterrows()):
        peak = pd.Timestamp(ev["peak_time"])
        sub = df[df["time"] == peak]
        for row_idx, (row_label, col) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            draw_basemap(ax, shp_gdf, compact=True)
            if sub.empty or col not in sub:
                ax.set_title(f"Event {int(ev.get('event_rank', col_idx + 1))}\nmissing peak rows")
                continue
            if col == "ifs_diagnostic_pred" and "ifs_diagnostic_valid" in sub:
                valid = sub["ifs_diagnostic_valid"].astype(bool).to_numpy()
                if (~valid).any():
                    ax.scatter(sub.loc[~valid, "lon"], sub.loc[~valid, "lat"], s=4, color="#D3D3D3", alpha=0.45, linewidths=0, zorder=2)
                plot_sub = sub.loc[valid]
            else:
                plot_sub = sub
            ax.scatter(
                plot_sub["lon"],
                plot_sub["lat"],
                c=plot_sub[col].astype(int),
                s=7,
                cmap=CLASS_CMAP,
                norm=CLASS_NORM,
                linewidths=0,
                alpha=0.95,
                zorder=3,
            )
            draw_boundary(ax, shp_gdf, color="#202020", linewidth=0.45, zorder=6)
            if row_idx == 0:
                ax.set_title(f"Event {int(ev.get('event_rank', col_idx + 1))}\n{peak:%Y-%m-%d %H:00 UTC}")
            if col_idx == 0:
                ax.text(-0.18, 0.5, row_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    handles = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(3)]
    handles.append(Patch(facecolor="#D3D3D3", label="IFS missing"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig_pair(
        fig,
        out_dir,
        "fig9_events_peak_grid_3x3",
        manifest,
        sources,
        notes="Rows are observed, PMST, and IFS diagnostic visibility class at event peak times.",
        n=len(eval_df),
    )


def plot_event_footprint(
    hourly_paths: Sequence[Path],
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> None:
    dfs = []
    for p in hourly_paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        print("  [WARN] No event hourly metrics for footprint figure.", flush=True)
        return
    setup_journal_style()
    ymax = 0
    for df in dfs:
        for col in ("obs_low_vis_count", "pmst_low_vis_count", "ifs_low_vis_count"):
            if col in df:
                ymax = max(ymax, int(np.nanmax(df[col].to_numpy(dtype=float))))
    fig, axes = plt.subplots(1, len(dfs), figsize=(4.3 * len(dfs), 3.7), sharey=True, squeeze=False)
    for i, (ax, df) in enumerate(zip(axes.ravel(), dfs), start=1):
        x = df["hour_offset"].to_numpy(dtype=float)
        ax.plot(x, df["obs_fog_count"], color="#111111", marker="o", lw=1.9, label="Obs fog")
        ax.plot(x, df["obs_low_vis_count"], color="#111111", marker="o", lw=1.2, ls="--", label="Obs low-vis")
        ax.plot(x, df["pmst_fog_count"], color=PMST_COLOR, marker="s", lw=1.9, label="PMST fog")
        ax.plot(x, df["pmst_low_vis_count"], color=PMST_COLOR, marker="s", lw=1.2, ls="--", label="PMST low-vis")
        if "ifs_fog_count" in df:
            ax.plot(x, df["ifs_fog_count"], color=IFS_DIAG_COLOR, marker="^", lw=1.9, label="IFS fog")
            ax.plot(x, df["ifs_low_vis_count"], color=IFS_DIAG_COLOR, marker="^", lw=1.2, ls="--", label="IFS low-vis")
        ax.axvline(0, color="#333333", lw=1.0, ls=":")
        ax.set_title(f"Event {i}")
        ax.set_xlabel("Hour relative to peak")
        ax.set_ylim(0, max(10, ymax * 1.08))
        if i == 1:
            ax.set_ylabel("Station count")
        add_panel_label(ax, chr(ord("a") + i - 1))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig9_events_footprint_evolution_1x3",
        manifest,
        sources,
        notes="Event footprint growth/decay from hourly station counts.",
    )


def plot_overlap_fig10(overlap_out: Path, out_dir: Path, manifest: Manifest) -> None:
    metrics_path = overlap_out / "ifs_diagnostic_matched_metrics.csv"
    if not metrics_path.exists():
        metrics_path = overlap_out / "overall_metrics.csv"
    if not metrics_path.exists():
        print(f"  [WARN] Overlap metrics not found under {overlap_out}; skip fig10.", flush=True)
        return
    df = pd.read_csv(metrics_path)
    if "source" not in df:
        print(f"  [WARN] Overlap metrics missing source column: {metrics_path}", flush=True)
        return

    if "low_vis_recall" not in df and "low_vis_pod" in df:
        df["low_vis_recall"] = pd.to_numeric(df["low_vis_pod"], errors="coerce")

    def _adaptive_ylim(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 1.0
        vmax = float(np.nanmax(arr))
        if vmax <= 0:
            return 0.10
        padded = vmax * 1.22
        if padded >= 0.92:
            return 1.0
        step = 0.02 if padded <= 0.20 else 0.05 if padded <= 0.50 else 0.10
        return min(1.0, max(step * 3, math.ceil(padded / step) * step))

    setup_journal_style()
    source_order = [s for s in ("tianji", "ifs", "ifs_diagnostic") if s in set(df["source"].astype(str))]
    if not source_order:
        return
    labels = {"tianji": "Tianji-input PMST", "ifs": "IFS-input PMST", "ifs_diagnostic": "IFS diagnostic VIS"}
    colors = {"tianji": PMST_COLOR, "ifs": IFS_PMST_COLOR, "ifs_diagnostic": IFS_DIAG_COLOR}
    panels = [
        ("Fog", [("fog_csi", "CSI"), ("fog_pod", "Recall")]),
        ("Mist", [("mist_csi", "CSI"), ("mist_pod", "Recall")]),
        ("Low visibility", [("low_vis_csi", "CSI"), ("low_vis_recall", "Recall")]),
    ]
    row_by_src = {str(r["source"]): r for _, r in df.iterrows()}
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), sharey=False)
    for ax_idx, (ax, (title, metric_specs)) in enumerate(zip(axes, panels)):
        x = np.arange(len(metric_specs))
        width = min(0.28, 0.80 / max(len(source_order), 1))
        panel_values = []
        for si, src in enumerate(source_order):
            vals = [float(row_by_src[src].get(m, np.nan)) for m, _ in metric_specs]
            vals_plot = [0.0 if not np.isfinite(v) else v for v in vals]
            panel_values.extend(vals)
            ax.bar(
                x + (si - (len(source_order) - 1) / 2) * width,
                vals_plot,
                width * 0.92,
                color=colors.get(src, "#777777"),
                label=labels.get(src, src) if ax_idx == 0 else None,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in metric_specs], rotation=25, ha="right")
        ax.set_ylim(0, _adaptive_ylim(panel_values))
        ax.set_ylabel("Score")
        add_panel_label(ax, chr(ord("a") + ax_idx))
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig10_overlap_forecast_source_comparison",
        manifest,
        [str(metrics_path)],
        notes="Controlled overlap-variable experiment using CSI/recall, including combined low-visibility CSI and recall.",
    )


def plot_fig11_lead_init(
    lead_pooled: pd.DataFrame,
    lead00: pd.DataFrame,
    lead12: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> None:
    setup_journal_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    specs = [
        ("Fog_CSI", "Fog CSI"),
        ("Fog_R", "Fog recall"),
        ("Mist_CSI", "Mist CSI"),
        ("low_vis_recall", "Low-vis recall"),
    ]
    for ax, (metric, title), letter in zip(axes.ravel(), specs, "abcd"):
        if metric in lead_pooled:
            ax.plot(lead_pooled["lead_hour"], lead_pooled[metric], color=PMST_COLOR, lw=2.2, label="Pooled")
        if metric in lead00 and not lead00.empty:
            ax.plot(lead00["lead_hour"], lead00[metric], "o-", color="#4C78A8", lw=1.4, ms=3, label="00Z")
        if metric in lead12 and not lead12.empty:
            ax.plot(lead12["lead_hour"], lead12[metric], "s-", color="#F58518", lw=1.4, ms=3, label="12Z")
        ax.set_title(title)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        add_panel_label(ax, letter)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        "fig11_48h_lead_init_00Z_12Z",
        manifest,
        sources,
        notes="Model-only 48h lead diagnostics stratified by forecast initialization hour.",
    )


# ---------------------------------------------------------------------------
# 48h lead diagnostics
# ---------------------------------------------------------------------------


def parse_init_hour(meta: pd.DataFrame) -> pd.Series:
    if "init_hour" in meta:
        return pd.to_numeric(meta["init_hour"], errors="coerce")
    if "init_time" not in meta:
        return pd.Series(np.nan, index=meta.index)
    s = meta["init_time"]
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        raw = s.astype(str).str.replace(r"\.0$", "", regex=True)
        dt = pd.to_datetime(raw, format="%Y%m%d%H", errors="coerce")
    return dt.dt.hour


def lead_metrics_table(
    y: np.ndarray,
    pred: np.ndarray,
    probs: Optional[np.ndarray],
    lead: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_n: int = 50,
) -> pd.DataFrame:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    rows = []
    lead_arr = np.asarray(lead, dtype=float)
    finite_lead = np.isfinite(lead_arr)
    mask = mask & finite_lead
    lead_i = np.full(len(lead_arr), -9999, dtype=np.int32)
    lead_i[finite_lead] = np.rint(lead_arr[finite_lead]).astype(np.int32)
    for h in sorted(np.unique(lead_i[mask])):
        m = mask & (lead_i == h)
        if int(m.sum()) < min_n:
            continue
        met = classification_metrics(y[m], pred[m], probs=probs[m] if probs is not None else None)
        row = {"lead_hour": int(h)}
        row.update(met)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lead_hour").reset_index(drop=True) if rows else pd.DataFrame()


def recover_ifs_48h_valid_time_grid(ds, n_rec: int, n_lead: int, lead_vals: np.ndarray) -> np.ndarray:
    lead_vals = np.asarray(lead_vals).reshape(-1).astype(np.int32)
    if lead_vals.shape[0] != n_lead:
        raise ValueError(f"lead_hour len={lead_vals.shape[0]} != n_lead={n_lead}")

    for coord_name in ("valid_time", "time"):
        if coord_name not in ds.coords:
            continue
        raw = np.asarray(ds[coord_name].values)
        if raw.ndim == 2:
            if raw.shape == (n_rec, n_lead):
                return pd.to_datetime(raw.reshape(-1)).values.astype("datetime64[ns]").reshape(n_rec, n_lead)
            if raw.shape == (n_lead, n_rec):
                return pd.to_datetime(raw.T.reshape(-1)).values.astype("datetime64[ns]").reshape(n_rec, n_lead)
            raise ValueError(f"{coord_name} shape={raw.shape}, expected {(n_rec, n_lead)}")
        flat = raw.reshape(-1)
        if flat.shape[0] == n_rec * n_lead:
            return pd.to_datetime(flat).values.astype("datetime64[ns]").reshape(n_rec, n_lead)
        if flat.shape[0] == n_rec:
            base = pd.to_datetime(flat).values.astype("datetime64[ns]")
            return base[:, None] + lead_vals.astype("timedelta64[h]")[None, :]

    if "forecast_reference_time" in ds.coords:
        base_raw = np.asarray(ds["forecast_reference_time"].values).reshape(-1)
        if base_raw.shape[0] != n_rec:
            raise ValueError(f"forecast_reference_time len={base_raw.shape[0]} != n_rec={n_rec}")
        base = pd.to_datetime(base_raw).values.astype("datetime64[ns]")
        return base[:, None] + lead_vals.astype("timedelta64[h]")[None, :]

    raise ValueError(
        "Cannot recover IFS 48h valid-time grid; need valid_time, time, or forecast_reference_time coordinates."
    )


def load_ifs_48h_diagnostic(
    meta: pd.DataFrame,
    ifs_nc_path: Path,
    vis_var: str = "VIS_ifs",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if not ifs_nc_path.exists():
        raise FileNotFoundError(f"IFS 48h NetCDF not found: {ifs_nc_path}")
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to match IFS 48h visibility.") from exc

    ds = xr.open_dataset(ifs_nc_path)
    try:
        if vis_var not in ds:
            raise KeyError(f"Variable {vis_var!r} not found in {ifs_nc_path}; vars={list(ds.data_vars)}")
        da = ds[vis_var].squeeze()
        if da.ndim != 3:
            raise ValueError(f"{vis_var} must be 3D after squeeze, got dims={da.dims}, shape={da.shape}")

        station_coord = "station" if "station" in ds.coords else "station_id" if "station_id" in ds.coords else ""
        if not station_coord:
            raise KeyError(f"IFS 48h NetCDF must have station or station_id coordinate: {ifs_nc_path}")
        station_dim = station_coord if station_coord in da.dims else next((d for d in da.dims if "station" in d.lower()), None)
        lead_dim = "lead_hour" if "lead_hour" in da.dims else next((d for d in da.dims if "lead" in d.lower()), None)
        if station_dim is None or lead_dim is None:
            raise ValueError(f"Cannot identify station/lead dims for {vis_var}: dims={da.dims}")
        record_dims = [d for d in da.dims if d not in {station_dim, lead_dim}]
        if len(record_dims) != 1:
            raise ValueError(f"Cannot identify single record dim for {vis_var}: dims={da.dims}")
        da = da.transpose(record_dims[0], lead_dim, station_dim)
        vis_arr = np.asarray(da.values)
        n_rec, n_lead, n_st = vis_arr.shape

        if "lead_hour" not in ds.coords:
            raise KeyError(f"IFS 48h NetCDF must have lead_hour coordinate: {ifs_nc_path}")
        lead_vals_raw = np.asarray(ds["lead_hour"].values)
        lead_vals = np.rint(lead_vals_raw.reshape(-1).astype(float)).astype(np.int32)
        if lead_vals.shape[0] != n_lead:
            raise ValueError(f"lead_hour len={lead_vals.shape[0]} != n_lead={n_lead}, raw shape={lead_vals_raw.shape}")

        stations = normalize_station_ids(ds[station_coord].values)
        if len(stations) != n_st:
            raise ValueError(f"{station_coord} len={len(stations)} != station dimension={n_st}")
        valid_time_grid = recover_ifs_48h_valid_time_grid(ds, n_rec, n_lead, lead_vals)
        if valid_time_grid.shape != (n_rec, n_lead):
            raise ValueError(f"valid_time_grid shape={valid_time_grid.shape}, expected {(n_rec, n_lead)}")

        pair_time_ns = valid_time_grid.reshape(-1).astype("datetime64[ns]").astype(np.int64)
        pair_lead = np.tile(lead_vals, n_rec).astype(np.int32)
        pair_index = pd.MultiIndex.from_arrays([pair_time_ns, pair_lead])
        pair_values = np.arange(pair_index.size, dtype=np.int64)
        if not pair_index.is_unique:
            keep = ~pair_index.duplicated(keep="first")
            pair_index = pair_index[keep]
            pair_values = pair_values[keep]
        meta_time = pd.to_datetime(meta["time"], errors="coerce").values.astype("datetime64[ns]").astype(np.int64)
        meta_lead = pd.to_numeric(meta["lead_hour"], errors="coerce").to_numpy(dtype=float)
        finite_lead = np.isfinite(meta_lead)
        meta_lead_i = np.full(len(meta), -9999, dtype=np.int32)
        meta_lead_i[finite_lead] = np.rint(meta_lead[finite_lead]).astype(np.int32)
        meta_index = pd.MultiIndex.from_arrays([meta_time, meta_lead_i])
        pair_pos = pd.Series(pair_values, index=pair_index).reindex(meta_index).to_numpy()

        station_lookup = pd.Series(np.arange(len(stations), dtype=np.int64), index=stations)
        meta_station = normalize_station_ids(meta["station_id"].values)
        station_pos = meta_station.map(station_lookup).to_numpy()

        raw = np.full(len(meta), np.nan, dtype=np.float64)
        pred = np.full(len(meta), -1, dtype=np.int64)
        valid = np.zeros(len(meta), dtype=bool)
        key_valid = pd.notna(pair_pos) & pd.notna(station_pos)
        if np.any(key_valid):
            pos = np.flatnonzero(key_valid)
            flat_pair = pair_pos[pos].astype(np.int64)
            rec_idx = flat_pair // n_lead
            lead_idx = flat_pair % n_lead
            st_idx = station_pos[pos].astype(np.int64)
            matched = np.asarray(vis_arr[rec_idx, lead_idx, st_idx], dtype=np.float64)
            finite = np.isfinite(matched)
            pos_f = pos[finite]
            raw[pos_f] = matched[finite]
            pred[pos_f] = classify_visibility_values(matched[finite])
            valid[pos_f] = True

        diag = {
            "n_model_rows": float(len(meta)),
            "n_ifs_records": float(n_rec),
            "n_ifs_leads": float(n_lead),
            "n_ifs_stations": float(n_st),
            "n_exact_key_matches": float(np.sum(key_valid)),
            "n_finite_matches": float(np.sum(valid)),
            "exact_match_ratio": safe_div(float(np.sum(key_valid)), float(len(meta))),
            "finite_match_ratio": safe_div(float(np.sum(valid)), float(len(meta))),
        }
        print(
            f"[48h IFS] matched finite rows: {int(valid.sum())}/{len(meta)} from {ifs_nc_path}",
            flush=True,
        )
        return pred, raw, valid, diag
    finally:
        ds.close()


def plot_fig11_48h_model_vs_ifs(
    cmp_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
) -> None:
    if cmp_df.empty:
        print("  [WARN] Empty 48h model-vs-IFS lead table; skip figure.", flush=True)
        return
    setup_journal_style()
    specs = [
        ("Fog_CSI", "Fog CSI"),
        ("Mist_CSI", "Mist CSI"),
        ("low_vis_csi", "Low-vis CSI"),
        ("low_vis_recall", "Low-vis recall"),
    ]

    def _adaptive_ylim(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 1.0
        vmax = float(np.nanmax(arr))
        if vmax <= 0:
            return 0.10
        padded = vmax * 1.18
        if padded >= 0.92:
            return 1.0
        step = 0.02 if padded <= 0.20 else 0.05 if padded <= 0.50 else 0.10
        return min(1.0, max(step * 3, math.ceil(padded / step) * step))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    for ax, (metric, title), letter in zip(axes.ravel(), specs, "abcd"):
        model_col = f"{metric}_model"
        ifs_col = f"{metric}_ifs"
        panel_values: List[float] = []
        if model_col in cmp_df:
            y_model = cmp_df[model_col].to_numpy(dtype=float)
            panel_values.extend(y_model.tolist())
            ax.plot(cmp_df["lead_hour"], y_model, color=PMST_COLOR, lw=2.2, label="PMST")
        if ifs_col in cmp_df:
            y_ifs = cmp_df[ifs_col].to_numpy(dtype=float)
            panel_values.extend(y_ifs.tolist())
            ax.plot(cmp_df["lead_hour"], y_ifs, color=IFS_DIAG_COLOR, lw=1.9, ls="--", marker="o", ms=2.5, label="IFS diagnostic")
        ax.set_title(title)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, _adaptive_ylim(panel_values))
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.10)
        add_panel_label(ax, letter)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    n_val = int(cmp_df["n_ifs"].sum()) if "n_ifs" in cmp_df else None
    save_fig_pair(
        fig,
        out_dir,
        "fig11_48h_model_vs_ifs_by_lead",
        manifest,
        sources,
        notes="48h lead diagnostics on exact UTC valid-time, station, and lead-hour matches between PMST samples and IFS 0-48h visibility.",
        n=n_val,
    )


def run_48h_optional(
    args: argparse.Namespace,
    base: Path,
    out_dir: Path,
    model,
    device,
    scaler,
    manifest: Manifest,
) -> None:
    if args.skip_48h:
        print("[48h] skipped by --skip_48h", flush=True)
        return
    data_48h = abs_under_base(base, args.data_48h_dir)
    if not data_48h.is_dir():
        fallback = base / "ml_dataset_fe_12h_48h_pm10_pm25"
        if fallback.is_dir():
            data_48h = fallback
        else:
            print(f"[48h] data dir not found: {data_48h}; skip fig11.", flush=True)
            return
    try:
        x_path, y_cls, _, meta = load_main_data(data_48h, args.limit_samples)
        dyn, fe = infer_layout_from_x(x_path, args.window_size)
        raw_model = model.module if hasattr(model, "module") else model
        model_dyn = int(getattr(raw_model, "dyn_vars", dyn))
        model_extra = 0
        try:
            if getattr(raw_model, "extra_encoder", None) is not None:
                model_extra = int(raw_model.extra_encoder[0].in_features)
        except Exception:
            model_extra = fe
        if dyn != model_dyn:
            print(f"[48h] dyn layout {dyn} differs from loaded model; skip.", flush=True)
            return
        if fe != model_extra:
            print(f"[48h] extra layout {fe} differs from loaded model extra {model_extra}; skip.", flush=True)
            return
        if "lead_hour" not in meta:
            print("[48h] meta_test.csv has no lead_hour; skip fig11.", flush=True)
            return
        init_hour = parse_init_hour(meta)
        if init_hour.isna().all():
            print("[48h] meta_test.csv has no parseable init_time/init_hour; skip fig11.", flush=True)
            return
        probs = run_model_inference(
            x_path,
            scaler,
            model,
            device,
            args.batch_size,
            args.window_size,
            dyn,
            fe,
            limit_samples=args.limit_samples,
            temperature=None,
        )
        pred = pred_from_probs_rule(probs, args.lead_fog_th, args.lead_mist_th, args.threshold_rule)
        lead = pd.to_numeric(meta["lead_hour"], errors="coerce").to_numpy(dtype=float)
        pooled = lead_metrics_table(y_cls, pred, probs, lead)
        lead00 = lead_metrics_table(y_cls, pred, probs, lead, mask=(init_hour.to_numpy() == 0))
        lead12 = lead_metrics_table(y_cls, pred, probs, lead, mask=(init_hour.to_numpy() == 12))
        pooled.to_csv(out_dir / "metrics_by_lead_hour_48h_model.csv", index=False)
        lead00.to_csv(out_dir / "metrics_by_lead_hour_init00Z.csv", index=False)
        lead12.to_csv(out_dir / "metrics_by_lead_hour_init12Z.csv", index=False)
        plot_fig11_lead_init(
            pooled,
            lead00,
            lead12,
            out_dir,
            manifest,
            [str(x_path), str(data_48h / "meta_test.csv")],
        )
        ifs_48h_nc = abs_under_base(base, args.ifs_48h_nc)
        if ifs_48h_nc.exists():
            try:
                ifs_pred, _, ifs_valid, ifs_diag = load_ifs_48h_diagnostic(meta, ifs_48h_nc, args.ifs_48h_var)
                diag_path = out_dir / "lead_eval_alignment_diagnostics_48h_ifs.csv"
                pd.DataFrame([ifs_diag]).to_csv(diag_path, index=False, float_format="%.6f")
                print(f"[table] {diag_path}", flush=True)
                matched_mask = np.asarray(ifs_valid, dtype=bool)
                if int(matched_mask.sum()) >= 50:
                    model_matched = lead_metrics_table(y_cls, pred, probs, lead, mask=matched_mask)
                    ifs_lead = lead_metrics_table(y_cls, ifs_pred, None, lead, mask=matched_mask)
                    model_matched_path = out_dir / "model_metrics_by_lead_hour_48h_ifs_matched.csv"
                    ifs_lead_path = out_dir / "ifs_metrics_by_lead_hour_48h.csv"
                    cmp_path = out_dir / "model_vs_ifs_metrics_by_lead_hour_48h.csv"
                    model_matched.to_csv(model_matched_path, index=False, float_format="%.6f")
                    ifs_lead.to_csv(ifs_lead_path, index=False, float_format="%.6f")
                    cmp_df = model_matched.merge(
                        ifs_lead,
                        on="lead_hour",
                        how="inner",
                        suffixes=("_model", "_ifs"),
                    ).sort_values("lead_hour").reset_index(drop=True)
                    for metric in (
                        "Fog_CSI",
                        "Fog_R",
                        "Mist_CSI",
                        "Mist_R",
                        "low_vis_csi",
                        "low_vis_recall",
                        "false_positive_rate",
                    ):
                        model_col = f"{metric}_model"
                        ifs_col = f"{metric}_ifs"
                        if model_col in cmp_df and ifs_col in cmp_df:
                            cmp_df[f"{metric}_diff_model_minus_ifs"] = cmp_df[model_col] - cmp_df[ifs_col]
                    cmp_df.to_csv(cmp_path, index=False, float_format="%.6f")
                    print(f"[table] {model_matched_path}", flush=True)
                    print(f"[table] {ifs_lead_path}", flush=True)
                    print(f"[table] {cmp_path}", flush=True)
                    plot_fig11_48h_model_vs_ifs(
                        cmp_df,
                        out_dir,
                        manifest,
                        [str(x_path), str(data_48h / "meta_test.csv"), str(ifs_48h_nc), str(cmp_path)],
                    )
                else:
                    print("[48h IFS] fewer than 50 matched rows; skip model-vs-IFS lead figure.", flush=True)
            except Exception as exc:
                print(f"[48h IFS] skipped after error: {exc}", flush=True)
        else:
            print(f"[48h IFS] NetCDF not found: {ifs_48h_nc}; skip model-vs-IFS lead figure.", flush=True)
    except Exception as exc:
        print(f"[48h] skipped after error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main and overlap execution
# ---------------------------------------------------------------------------


def run_overlap_subprocess(args: argparse.Namespace, base: Path, out_dir: Path, manifest: Manifest) -> Optional[Path]:
    overlap_script = abs_under_base(base, args.overlap_script) if args.overlap_script else base / "ifs_baseline" / "test_PMST_overlap_forecast_source_s2.py"
    if not overlap_script.is_file():
        print(f"[overlap] script not found: {overlap_script}; skip.", flush=True)
        return None
    overlap_out = abs_under_base(base, args.overlap_out_dir) if args.overlap_out_dir else out_dir / "overlap_forecast_source"
    cmd = [
        sys.executable,
        str(overlap_script),
        "--out_dir",
        str(overlap_out),
        "--ifs_forecast_nc",
        str(abs_under_base(base, args.ifs_vis_nc)),
        "--ifs_forecast_var",
        args.ifs_vis_var,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.limit_samples and args.limit_samples > 0:
        cmd.extend(["--limit_samples", str(args.limit_samples)])
        cmd.extend(["--bootstrap", "50", "--bootstrap_size", str(min(args.limit_samples, 20000))])
    if args.skip_overlap_bootstrap:
        cmd.append("--skip_bootstrap")
    if args.overlap_extra_args.strip():
        cmd.extend(args.overlap_extra_args.strip().split())
    print("[overlap] running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[overlap] evaluator failed with exit code {exc.returncode}; continuing.", flush=True)
        return overlap_out
    plot_overlap_fig10(overlap_out, out_dir, manifest)
    # Also copy the overlap script's own figure when present for traceability.
    for name in ("fig_forecast_source_key_metrics.png", "fig_forecast_source_key_metrics.pdf"):
        src = overlap_out / name
        if src.exists():
            dst = out_dir / f"source_{name}"
            shutil.copy2(src, dst)
    return overlap_out


def run_main(args: argparse.Namespace, base: Path, out_dir: Path, manifest: Manifest):
    import torch
    import joblib

    data_dir = abs_under_base(base, args.data_dir)
    ckpt_path = abs_under_base(base, args.ckpt_path)
    scaler_path = abs_under_base(base, args.scaler_path)
    season_th_path = abs_under_base(base, args.season_th_path)
    ifs_nc = abs_under_base(base, args.ifs_vis_nc)
    model_py = resolve_model_py(base, args.model_py)
    print(f"[profile] S2 run id: {DEFAULT_S2_RUN_ID}", flush=True)
    print(f"[paths] data_dir       : {data_dir}", flush=True)
    print(f"[paths] checkpoint     : {ckpt_path}", flush=True)
    print(f"[paths] scaler         : {scaler_path}", flush=True)
    print(f"[paths] season_th      : {season_th_path}", flush=True)
    print(f"[paths] model_py       : {model_py}", flush=True)
    print(f"[paths] output_dir     : {out_dir}", flush=True)
    require_existing_path(data_dir, "data_dir", expect_dir=True)
    require_existing_path(ckpt_path, "checkpoint")
    require_existing_path(scaler_path, "scaler")
    require_existing_path(model_py, "model_py")
    x_path, y_cls, y_raw, meta = load_main_data(data_dir, args.limit_samples)
    dyn_inferred, fe_inferred = infer_layout_from_x(x_path, args.window_size)
    dyn = args.dyn_vars_count or dyn_inferred
    fe = args.extra_feat_dim or fe_inferred
    print(f"[layout] dyn_vars_count={dyn}, extra_feat_dim={fe}", flush=True)

    model_cls = import_model_class(model_py)
    device = resolve_device(args.device)
    scaler = joblib.load(scaler_path)
    model = model_cls(
        dyn_vars_count=dyn,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        extra_feat_dim=fe,
    ).to(device)
    load_checkpoint_into_model(model, ckpt_path, device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[model] DataParallel on {torch.cuda.device_count()} devices", flush=True)
    model.eval()

    temperature = None
    if args.use_calibration and season_th_path.exists():
        try:
            try:
                cal = torch.load(season_th_path, map_location="cpu", weights_only=True)
            except TypeError:
                cal = torch.load(season_th_path, map_location="cpu")
            temperature = cal.get("temperature")
            if temperature is not None:
                temperature = float(temperature)
            print(f"[calibration] temperature={temperature} from {season_th_path}", flush=True)
        except Exception as exc:
            print(f"[calibration] could not load {season_th_path}: {exc}", flush=True)

    probs = run_model_inference(
        x_path,
        scaler,
        model,
        device,
        args.batch_size,
        args.window_size,
        dyn,
        fe,
        limit_samples=args.limit_samples,
        temperature=temperature,
    )
    pmst_pred = pred_from_probs_rule(probs, args.fog_th, args.mist_th, args.threshold_rule)
    pmst_metrics = classification_metrics(y_cls, pmst_pred, probs=probs)
    overall_df = metrics_rows(
        [
            (
                "pmst",
                pmst_metrics,
                {
                    "sample_scope": "test",
                    "fog_threshold": args.fog_th,
                    "mist_threshold": args.mist_th,
                    "threshold_rule": args.threshold_rule,
                    "checkpoint": str(ckpt_path),
                    "data_dir": str(data_dir),
                },
            )
        ]
    )

    ifs_pred = ifs_vis = ifs_valid = None
    ifs_metrics = None
    try:
        ifs_pred, ifs_vis, ifs_valid = load_ifs_diagnostic(meta, ifs_nc, args.ifs_vis_var)
        if int(np.sum(ifs_valid)) > 0:
            ifs_metrics = classification_metrics(y_cls[ifs_valid], ifs_pred[ifs_valid])
            matched_pmst_metrics = classification_metrics(y_cls[ifs_valid], pmst_pred[ifs_valid], probs=probs[ifs_valid])
            ifs_df = metrics_rows(
                [
                    ("pmst", matched_pmst_metrics, {"sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid))}),
                    ("ifs_diagnostic", ifs_metrics, {"sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid)), "ifs_forecast_nc": str(ifs_nc)}),
                ]
            )
            ifs_df.to_csv(out_dir / "ifs_diagnostic_matched_metrics.csv", index=False)
            delta_df = metric_deltas(
                matched_pmst_metrics,
                ifs_metrics,
                "pmst",
                "ifs_diagnostic",
                [
                    "Fog_CSI",
                    "Fog_R",
                    "Fog_P",
                    "Fog_FAR",
                    "Mist_CSI",
                    "Mist_R",
                    "Mist_P",
                    "Mist_FAR",
                    "low_vis_csi",
                    "low_vis_recall",
                    "false_positive_rate",
                    "accuracy",
                ],
            )
            delta_df.to_csv(out_dir / "metric_deltas_pmst_minus_ifs_diagnostic.csv", index=False)
            overall_df = pd.concat([overall_df, ifs_df], ignore_index=True)
    except Exception as exc:
        print(f"[IFS] diagnostic baseline skipped: {exc}", flush=True)
        ifs_valid = np.zeros(len(y_cls), dtype=bool)

    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)
    eval_df = export_per_sample(
        out_dir / "per_sample_eval.csv",
        meta,
        y_cls,
        y_raw,
        pmst_pred,
        probs,
        ifs_pred=ifs_pred,
        ifs_vis=ifs_vis,
        ifs_valid=ifs_valid,
    )
    write_report(out_dir / "rare_event_report.txt", y_cls, pmst_pred, pmst_metrics, ifs_metrics)

    station_df = aggregate_station_metrics(eval_df, "pmst_pred")
    station_df.to_csv(out_dir / "station_metrics.csv", index=False)
    scenario_df = build_scenario_metrics(eval_df)
    scenario_df.to_csv(out_dir / "scenario_metrics.csv", index=False)

    setup_journal_style()
    sources_main = [str(x_path), str(data_dir / "y_test.npy"), str(data_dir / "meta_test.csv"), str(ckpt_path), str(scaler_path)]
    if ifs_nc.exists():
        sources_main.append(str(ifs_nc))
    plot_confusion_pmst_vs_ifs(y_cls, pmst_pred, ifs_pred, ifs_valid, out_dir, manifest, sources_main)
    plot_csi_recall_pmst_vs_ifs(
        pmst_metrics if ifs_metrics is None else classification_metrics(y_cls[ifs_valid], pmst_pred[ifs_valid], probs=probs[ifs_valid]),
        ifs_metrics,
        out_dir,
        manifest,
        sources_main,
        n=len(y_cls),
        matched_ifs=int(np.sum(ifs_valid)) if ifs_valid is not None else None,
    )
    if ifs_vis is not None and ifs_valid is not None:
        plot_ifs_visibility_bias(y_cls, y_raw, ifs_vis, ifs_valid, out_dir, manifest, sources_main)

    for split, order in (
        ("time_of_day", TIME_OF_DAY_LOCAL_ORDER),
        ("season", ["DJF", "MAM", "JJA", "SON"]),
        ("region", [r[0] for r in REGION_DEFS] + ["Other"]),
    ):
        plot_scenario_split(scenario_df, split, order, out_dir, manifest, [str(out_dir / "scenario_metrics.csv")])
    plot_time_of_day_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")], offset_hours=args.local_time_offset_hours)
    plot_region_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")])

    shp_gdf = read_shapefile(args.shp_path)
    plot_station_metric_map(
        station_df,
        "fog_recall",
        "n_fog",
        5,
        "Station Fog Recall",
        "fig8_station_fog_recall",
        out_dir,
        manifest,
        [str(out_dir / "station_metrics.csv")],
        shp_gdf=shp_gdf,
        cmap="cividis",
        vmin=0,
        vmax=1,
    )
    plot_station_metric_map(
        station_df,
        "mist_recall",
        "n_mist",
        5,
        "Station Mist Recall",
        "fig8_station_mist_recall",
        out_dir,
        manifest,
        [str(out_dir / "station_metrics.csv")],
        shp_gdf=shp_gdf,
        cmap="cividis",
        vmin=0,
        vmax=1,
    )
    plot_station_metric_map(
        station_df,
        "fpr_fog",
        "n_clear",
        20,
        "Station Low-Visibility False Positive Rate",
        "fig8_station_fpr",
        out_dir,
        manifest,
        [str(out_dir / "station_metrics.csv")],
        shp_gdf=shp_gdf,
        cmap="magma_r",
        vmin=0,
        vmax=0.2,
    )

    event_df = pd.DataFrame()
    if run_widespread_event_evaluation is not None:
        try:
            event_df = run_widespread_event_evaluation(
                meta=meta,
                y_true=y_cls,
                y_true_raw=y_raw,
                pmst_pred=pmst_pred,
                output_dir=str(out_dir),
                shp_path=args.shp_path,
                ifs_nc_path=str(ifs_nc),
                top_k=args.event_top_k,
                window_hours=args.event_window_hours,
                min_fog_stations=args.event_min_fog_stations,
                min_regions=args.event_min_regions,
                min_lon_span=args.event_min_lon_span,
                min_lat_span=args.event_min_lat_span,
                gap_hours=args.event_gap_hours,
            )
        except Exception as exc:
            print(f"[events] run_widespread_event_evaluation failed: {exc}", flush=True)
    event_path = out_dir / "event_case_summary.csv"
    if event_path.exists():
        event_df = pd.read_csv(event_path, parse_dates=["peak_time", "start_time", "end_time"])
    elif event_df is not None and not event_df.empty:
        event_df.to_csv(event_path, index=False)
    if "ifs_diagnostic_pred" not in eval_df and ifs_pred is not None:
        eval_df["ifs_diagnostic_pred"] = ifs_pred
        eval_df["ifs_diagnostic_valid"] = ifs_valid
    plot_event_peak_grid(eval_df, event_df, out_dir, manifest, [str(event_path), str(out_dir / "per_sample_eval.csv")], shp_gdf=shp_gdf)
    hourly_paths = [out_dir / f"fig9_event_{k}_hourly_metrics.csv" for k in (1, 2, 3)]
    plot_event_footprint(hourly_paths, out_dir, manifest, [str(p) for p in hourly_paths])

    run_48h_optional(args, base, out_dir, model, device, scaler, manifest)

    run_config = {
        "args": vars(args),
        "base": str(base),
        "data_dir": str(data_dir),
        "x_path": str(x_path),
        "ckpt_path": str(ckpt_path),
        "scaler_path": str(scaler_path),
        "ifs_vis_nc": str(ifs_nc),
        "dyn_vars_count": int(dyn),
        "extra_feat_dim": int(fe),
        "n_samples": int(len(y_cls)),
        "ifs_matched_rows": int(np.sum(ifs_valid)) if ifs_valid is not None else 0,
        "class_definition": {
            "0": "0 <= visibility < 500 m",
            "1": "500 <= visibility < 1000 m",
            "2": "visibility >= 1000 m",
        },
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    return model, device, scaler


def run_tables_mode(args: argparse.Namespace, base: Path, out_dir: Path, manifest: Manifest) -> None:
    per_sample_path = out_dir / "per_sample_eval.csv"
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing posthoc table input: {per_sample_path}")
    eval_df = pd.read_csv(per_sample_path)
    required = {"time", "lat", "lon", "y_true", "pmst_pred"}
    missing = sorted(required - set(eval_df.columns))
    if missing:
        raise ValueError(f"{per_sample_path} is missing required columns: {missing}")
    eval_df["time"] = pd.to_datetime(eval_df["time"], errors="coerce")
    eval_df = add_scenario_columns(eval_df)

    scenario_df = build_scenario_metrics(eval_df)
    scenario_path = out_dir / "scenario_metrics.csv"
    scenario_df.to_csv(scenario_path, index=False)
    print(f"[table] {scenario_path}", flush=True)

    sources = [str(per_sample_path)]
    plot_scenario_split(scenario_df, "time_of_day", TIME_OF_DAY_LOCAL_ORDER, out_dir, manifest, [str(scenario_path)])
    plot_diurnal_time_detail(eval_df, out_dir, manifest, sources, offset_hours=args.local_time_offset_hours)
    plot_time_of_day_detail(eval_df, out_dir, manifest, sources, offset_hours=args.local_time_offset_hours)
    plot_region_detail(eval_df, out_dir, manifest, sources)
    plot_feature_convergence_from_history(base, out_dir, manifest, args.history_paths, args.history_labels)


def main() -> None:
    args = parse_args()
    setup_journal_style()
    base = Path(args.base).resolve()
    out_dir = abs_under_base(base, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir)

    print("=" * 72, flush=True)
    print("PM10+PM2.5 journal paper evaluation", flush=True)
    print(f"mode     : {args.mode}", flush=True)
    print(f"base     : {base}", flush=True)
    print(f"out_dir  : {out_dir}", flush=True)
    print("=" * 72, flush=True)

    if args.mode == "tables":
        run_tables_mode(args, base, out_dir, manifest)
    if args.mode in ("main", "all"):
        run_main(args, base, out_dir, manifest)
    if args.mode in ("overlap", "all"):
        run_overlap_subprocess(args, base, out_dir, manifest)

    manifest.write()
    print("=" * 72, flush=True)
    print(f"[OK] journal evaluation outputs written to: {out_dir}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
