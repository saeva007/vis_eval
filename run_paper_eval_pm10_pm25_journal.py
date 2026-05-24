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
import shlex
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
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
    from paper_eval_config import DEFAULT_CONFIG_NAME, apply_paper_eval_config
except Exception as exc:  # pragma: no cover - import failure should be explicit on remote.
    raise RuntimeError(f"Cannot import paper_eval_config from {VIS_EVAL_DIR}") from exc

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
        plot_three_events_footprint_row,
        plot_three_events_peak_row,
        load_china_shapefile as _load_china_shapefile,
        _draw_event_basemap as _plot_spatial_event_basemap,
    )
except Exception:
    run_widespread_event_evaluation = None
    detect_widespread_fog_events = None
    plot_three_events_footprint_row = None
    plot_three_events_peak_row = None
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
SEASON_MAP = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
LOCAL_TIME_LABEL = "UTC+8"
TIME_OF_DAY_LOCAL_ORDER = [
    f"Night (00-05 {LOCAL_TIME_LABEL})",
    f"Morning (06-11 {LOCAL_TIME_LABEL})",
    f"Afternoon (12-17 {LOCAL_TIME_LABEL})",
    f"Evening (18-23 {LOCAL_TIME_LABEL})",
]
DEFAULT_S2_RUN_ID = "exp_1778563813_pm10_more_temp_search_utc"
DEFAULT_DATA_DIR = "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_DATA_48H_DIR = "ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime"
DEFAULT_MODEL_PY = "PMST_net_test_11_s2_pm10.py"
DEFAULT_CKPT_PATH = f"checkpoints/{DEFAULT_S2_RUN_ID}_S2_PhaseB_best_score.pt"
DEFAULT_SCALER_PATH = f"checkpoints/robust_scaler_{DEFAULT_S2_RUN_ID}_w12_dyn27_s2_48h_pm10.pkl"
DEFAULT_SEASON_TH_PATH = f"checkpoints/{DEFAULT_S2_RUN_ID}_season_thresholds.pt"
DEFAULT_OUT_DIR = "paper_eval_results_pm10_pm25_journal_utc"
DEFAULT_FOG_TH = 0.34
DEFAULT_MIST_TH = 0.56
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
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
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
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
        print(f"  [Fig] Saved -> {path}", flush=True)
    if manifest is not None:
        manifest.add(f"{stem}.png/pdf/svg", sources, notes=notes, n=n, matched_ifs=matched_ifs)
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
    ap.add_argument(
        "--config_json",
        default=os.environ.get("PAPER_EVAL_CONFIG", str(VIS_EVAL_DIR / DEFAULT_CONFIG_NAME)),
        help="Central JSON run configuration. Pass 'none' to use hard-coded CLI defaults only.",
    )
    ap.add_argument("--mode", choices=["main", "overlap", "all", "tables", "lead48"], default="all")
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
    ap.add_argument("--fog_th", type=float, default=DEFAULT_FOG_TH)
    ap.add_argument("--mist_th", type=float, default=DEFAULT_MIST_TH)
    ap.add_argument("--lead_fog_th", type=float, default=DEFAULT_FOG_TH)
    ap.add_argument("--lead_mist_th", type=float, default=DEFAULT_MIST_TH)
    ap.add_argument("--threshold_rule", choices=["default", "mutual", "joint"], default="mutual")
    ap.add_argument(
        "--decision_rule",
        choices=["fine_threshold", "binary_gate"],
        default="fine_threshold",
        help="Primary PMST decision rule for paper figures and tables.",
    )
    ap.add_argument(
        "--lowvis_gate_th",
        type=float,
        default=0.81,
        help="Binary low-visibility gate threshold used when --decision_rule=binary_gate.",
    )
    ap.add_argument(
        "--lead_lowvis_gate_th",
        type=float,
        default=0.81,
        help="Binary gate threshold for optional 48h lead evaluation.",
    )
    ap.add_argument("--experiment_id", default="", help="Traceable experiment id written to run_config.json.")
    ap.add_argument(
        "--selection_result_json",
        default="",
        help="Optional low-vis gate selection JSON used to justify --lowvis_gate_th.",
    )
    ap.add_argument(
        "--write_decision_comparison",
        action="store_true",
        help="Write fine-threshold, binary-gate, and saved-season-threshold comparison tables.",
    )
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
    ap.add_argument("--overlap_tianji_ckpt", default="", help="Optional Tianji-source overlap checkpoint path.")
    ap.add_argument("--overlap_ifs_ckpt", default="", help="Optional IFS-source overlap checkpoint path.")
    ap.add_argument("--overlap_tianji_scaler", default="", help="Optional Tianji-source overlap scaler path.")
    ap.add_argument("--overlap_ifs_scaler", default="", help="Optional IFS-source overlap scaler path.")
    ap.add_argument("--overlap_feature_importance_csv", default="", help="Optional feature-importance CSV used by overlap feature-replacement analysis.")
    ap.add_argument("--overlap_feature_swap_top_k", type=int, default=0, help="Run overlap feature replacement for top-K dynamic variables when >0.")
    ap.add_argument(
        "--overlap_feature_swap_features",
        default="RH2M,Q_1000,DP_1000,RH_925,PRECIP",
        help="Comma/semicolon feature names for overlap feature replacement.",
    )
    ap.add_argument("--skip_overlap_bootstrap", action="store_true")
    ap.add_argument("--run_variable_quality", action="store_true", help="Run Tianji-vs-IFS forecast-variable quality analysis when inputs exist.")
    ap.add_argument("--variable_quality_script", default="", help="Optional path to analyze_key_variable_quality.py.")
    ap.add_argument("--obs_root", default="/public/home/putianshu/vis_mlp/auto_station", help="Root directory for station observation CSV files.")
    ap.add_argument("--quality_tianji_data_dir", default="ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_baseline")
    ap.add_argument("--quality_ifs_data_dir", default="ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_baseline")
    ap.add_argument("--quality_out_dir", default="", help="Optional output dir for forecast-variable quality analysis.")
    ap.add_argument("--quality_features", default="RH2M,T2M,WSPD10,MSLP,PRECIP", help="Comma/semicolon key variables for forecast-quality analysis.")
    ap.add_argument(
        "--local_time_offset_hours",
        type=int,
        default=LOCAL_TIME_OFFSET_HOURS,
        help="Offset applied to UTC timestamps for diurnal plots; default 8 converts UTC to UTC+8.",
    )
    ap.add_argument(
        "--meta_time_shift_hours",
        type=float,
        default=0.0,
        help=(
            "Add this many hours to meta_test.csv time before UTC-indexed evaluation. "
            "Use -8 only for legacy datasets whose meta time is BJT-labelled rather than UTC; "
            "rebuilding the dataset with UTC splitting is preferred for event windows."
        ),
    )
    ap.add_argument("--history_paths", default="", help="Comma/semicolon-separated training history JSON or stdout .out/.log files.")
    ap.add_argument("--history_labels", default="", help="Optional labels matching --history_paths order.")
    args = ap.parse_args()
    return apply_paper_eval_config(args, "journal", default_dir=VIS_EVAL_DIR)


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


def binary_gate_pred(probs: np.ndarray, low_prob: np.ndarray, low_th: float) -> np.ndarray:
    """Predict low visibility with the binary head, then split Fog/Mist by fine-head argmax."""

    if low_prob is None:
        raise ValueError("binary_gate decision requires low-visibility probabilities from the model.")
    probs = np.asarray(probs, dtype=np.float64)
    low_prob = np.asarray(low_prob, dtype=np.float64).reshape(-1)
    if len(low_prob) != len(probs):
        raise ValueError(f"low_prob length {len(low_prob)} does not match probs length {len(probs)}")
    pred = np.full(len(low_prob), 2, dtype=np.int64)
    low = low_prob >= float(low_th)
    low_class = np.where(probs[:, 0] >= probs[:, 1], 0, 1)
    pred[low] = low_class[low]
    return pred


def pred_from_decision_rule(
    probs: np.ndarray,
    low_prob: Optional[np.ndarray],
    decision_rule: str,
    fog_th: float,
    mist_th: float,
    threshold_rule: str,
    lowvis_gate_th: float,
) -> np.ndarray:
    if decision_rule == "binary_gate":
        return binary_gate_pred(probs, low_prob, lowvis_gate_th)
    return pred_from_probs_rule(probs, fog_th, mist_th, threshold_rule)


def load_saved_season_thresholds(season_th_path: Path) -> Tuple[Optional[dict], Optional[float]]:
    if not season_th_path or not season_th_path.exists():
        return None, None
    try:
        import torch

        try:
            payload = torch.load(season_th_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(season_th_path, map_location="cpu")
    except Exception as exc:
        print(f"[season] could not load saved thresholds from {season_th_path}: {exc}", flush=True)
        return None, None
    if not isinstance(payload, dict):
        return None, None
    temperature = payload.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
    return payload.get("season_thresholds"), temperature


def thresholds_from_saved_seasons(
    meta: pd.DataFrame,
    season_thresholds: dict,
    default_fog: float,
    default_mist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if "month_analysis" in meta:
        months = pd.to_numeric(meta["month_analysis"], errors="coerce")
    elif "month" in meta:
        months = pd.to_numeric(meta["month"], errors="coerce")
    else:
        months = pd.to_datetime(meta["time"], errors="coerce").dt.month
    fog = np.full(len(meta), float(default_fog), dtype=np.float64)
    mist = np.full(len(meta), float(default_mist), dtype=np.float64)
    for i, month in enumerate(months):
        if pd.isna(month):
            continue
        season = SEASON_MAP.get(int(month))
        rec = season_thresholds.get(season, {}) if season else {}
        if isinstance(rec, dict):
            fog[i] = float(rec.get("fog_th", rec.get("fog", default_fog)))
            mist[i] = float(rec.get("mist_th", rec.get("mist", default_mist)))
    return fog, mist


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
    return_low_prob: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import torch
    import torch.nn.functional as F

    X = np.load(x_path, mmap_mode="r")
    n = len(X) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(X))
    temp = 1.0 if temperature is None else max(float(temperature), 1e-6)
    out = []
    low_out = []
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = prepare_batch_rows(rows, scaler, window_size, dyn_vars_count, extra_feat_dim)
        x = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            model_out = model(x)
            logits = model_out[0]
            probs = F.softmax(logits / temp, dim=1)
            low_logit = model_out[2] if len(model_out) >= 3 else None
        out.append(probs.detach().cpu().numpy())
        if return_low_prob:
            if low_logit is None:
                raise ValueError("Model forward output does not include low_vis_detector logits.")
            low_out.append(torch.sigmoid(low_logit).detach().cpu().numpy().reshape(-1))
        if start == 0 or end == n or (start // max(batch_size, 1)) % 20 == 0:
            print(f"  [inference] {end}/{n}", flush=True)
    probs_all = np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)
    if return_low_prob:
        low_all = np.concatenate(low_out, axis=0) if low_out else np.zeros((0,), dtype=np.float32)
        return probs_all, low_all
    return probs_all, None


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
    d["low_vis_precision"] = safe_div(low_tp, low_tp + low_fp)
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


def canonicalize_meta_time(meta: pd.DataFrame, meta_time_shift_hours: float = 0.0) -> pd.DataFrame:
    out = meta.copy()
    parsed = pd.to_datetime(out["time"], errors="coerce")
    out["time_utc_original"] = parsed
    if float(meta_time_shift_hours or 0.0) != 0.0:
        parsed = parsed + pd.to_timedelta(float(meta_time_shift_hours), unit="h")
        out["meta_time_shift_hours"] = float(meta_time_shift_hours)
    out["time"] = parsed
    out["time_utc"] = parsed
    return out


def load_main_data(
    data_dir: Path,
    limit_samples: int = 0,
    meta_time_shift_hours: float = 0.0,
) -> Tuple[Path, np.ndarray, np.ndarray, pd.DataFrame]:
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
    meta = canonicalize_meta_time(meta, meta_time_shift_hours)
    if float(meta_time_shift_hours or 0.0) != 0.0:
        print(
            f"[time] Applied meta_time_shift_hours={float(meta_time_shift_hours):+g}; "
            "meta['time'] is now UTC for matching and event plots.",
            flush=True,
        )
    meta["hour"] = meta["time"].dt.hour
    meta["hour_utc"] = meta["time"].dt.hour
    meta = add_local_time_columns(meta)
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
    low_prob: Optional[np.ndarray] = None,
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
            "time_utc",
            "time_utc_original",
            "time_analysis",
            "meta_time_shift_hours",
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
    if low_prob is not None:
        df["pmst_p_lowvis_binary"] = np.asarray(low_prob, dtype=np.float64).reshape(-1)
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
        f.write(f"  low_vis_precision: {values.get('low_vis_precision', np.nan):.6f}\n")
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
        true_low = y <= 1
        fog_tp = float(((y == 0) & (p == 0)).sum())
        fog_fp = float(((y != 0) & (p == 0)).sum())
        fog_fn = float(((y == 0) & (p != 0)).sum())
        mist_tp = float(((y == 1) & (p == 1)).sum())
        mist_fp = float(((y != 1) & (p == 1)).sum())
        mist_fn = float(((y == 1) & (p != 1)).sum())
        low_tp = float((true_low & pred_low).sum())
        low_fp = float((~true_low & pred_low).sum())
        low_fn = float((true_low & ~pred_low).sum())
        row = {
            "station_id": sid,
            "lat": float(sub["lat"].iloc[0]),
            "lon": float(sub["lon"].iloc[0]),
            "n_total": int(len(sub)),
            "n_fog": n_fog,
            "n_mist": n_mist,
            "n_low_vis": n_fog + n_mist,
            "n_clear": n_clear,
            "fog_recall": safe_div(fog_tp, float(n_fog)),
            "fog_csi": safe_div(fog_tp, fog_tp + fog_fp + fog_fn),
            "mist_recall": safe_div(mist_tp, float(n_mist)),
            "mist_csi": safe_div(mist_tp, mist_tp + mist_fp + mist_fn),
            "low_vis_recall": safe_div(low_tp, low_tp + low_fn),
            "low_vis_csi": safe_div(low_tp, low_tp + low_fp + low_fn),
            "fpr_fog": safe_div(float(((y == 2) & pred_low).sum()), float(n_clear)),
            "overall_acc": float((y == p).mean()) if len(sub) else math.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_station_model_vs_ifs_metrics(
    eval_df: pd.DataFrame,
    model_col: str = "pmst_pred",
    ifs_col: str = "ifs_diagnostic_pred",
    valid_col: str = "ifs_diagnostic_valid",
) -> pd.DataFrame:
    """Per-station PMST-vs-IFS deltas on exactly matched IFS rows."""

    required = {"station_id", "lat", "lon", "y_true", model_col, ifs_col, valid_col}
    missing = sorted(required - set(eval_df.columns))
    if missing:
        print(f"  [WARN] Cannot build station PMST-vs-IFS table; missing columns: {missing}", flush=True)
        return pd.DataFrame()

    df = eval_df[np.asarray(eval_df[valid_col], dtype=bool)].copy()
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for sid, sub in df.groupby("station_id", sort=False):
        y = sub["y_true"].to_numpy(dtype=np.int64)
        p_model = sub[model_col].to_numpy(dtype=np.int64)
        p_ifs = sub[ifs_col].to_numpy(dtype=np.int64)
        n_fog = int((y == 0).sum())
        n_mist = int((y == 1).sum())
        n_low = int((y <= 1).sum())
        if n_low <= 0:
            continue
        met_model = classification_metrics(y, p_model)
        met_ifs = classification_metrics(y, p_ifs)
        row: Dict[str, object] = {
            "station_id": sid,
            "lat": float(sub["lat"].iloc[0]),
            "lon": float(sub["lon"].iloc[0]),
            "n_total_matched": int(len(sub)),
            "n_fog": n_fog,
            "n_mist": n_mist,
            "n_low_vis": n_low,
            "n_clear": int((y == 2).sum()),
        }
        metric_names = {
            "Fog_R": "fog_recall",
            "Mist_R": "mist_recall",
            "low_vis_recall": "low_vis_recall",
            "Fog_CSI": "fog_csi",
            "Mist_CSI": "mist_csi",
            "low_vis_csi": "low_vis_csi",
            "false_positive_rate": "false_positive_rate",
            "accuracy": "accuracy",
        }
        for metric, safe in metric_names.items():
            pm = float(met_model.get(metric, np.nan))
            iv = float(met_ifs.get(metric, np.nan))
            row[f"pmst_{safe}"] = pm
            row[f"ifs_{safe}"] = iv
            row[f"delta_{safe}"] = pm - iv
            row[f"delta_{safe}_pctpt"] = 100.0 * (pm - iv)
        for safe in ("fog_recall", "mist_recall", "low_vis_recall"):
            delta = float(row.get(f"delta_{safe}", np.nan))
            row[f"pmst_better_{safe}"] = bool(np.isfinite(delta) and delta > 0)
            row[f"ifs_better_{safe}"] = bool(np.isfinite(delta) and delta < 0)
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
        ("low_vis_csi", "Low-vis CSI"),
        ("low_vis_recall", "Low-vis recall"),
    ]
    colors = [FOG_COLOR, "#6E91B5", MIST_COLOR, "#F0B84A", "#334155", "#64748B"]
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
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
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
        ("low_vis_csi", "Low-vis CSI", "#334155"),
        ("low_vis_recall", "Low-vis recall", "#64748B"),
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


def _nice_delta_limit(values: np.ndarray, quantile: float = 98.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.20
    raw = float(np.nanpercentile(np.abs(vals), quantile))
    if not np.isfinite(raw) or raw <= 0:
        raw = float(np.nanmax(np.abs(vals))) if vals.size else 0.20
    return min(1.0, max(0.10, math.ceil(raw / 0.05) * 0.05))


def _nice_hist_limits(values: np.ndarray, min_abs: float) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -min_abs, min_abs
    lo = math.floor(float(np.nanmin(vals)) / 0.10) * 0.10
    hi = math.ceil(float(np.nanmax(vals)) / 0.10) * 0.10
    lo = min(lo, -min_abs)
    hi = max(hi, min_abs)
    if lo == hi:
        lo -= 0.10
        hi += 0.10
    return lo, hi


def _gaussian_density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.array([], dtype=float)
    std = float(np.nanstd(vals, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return np.array([], dtype=float)
    bw = 1.06 * std * (vals.size ** (-1.0 / 5.0))
    if not np.isfinite(bw) or bw <= 0:
        return np.array([], dtype=float)
    z = (grid[:, None] - vals[None, :]) / bw
    return np.exp(-0.5 * z * z).mean(axis=1) / (bw * math.sqrt(2.0 * math.pi))


def plot_station_recall_delta_map(
    station_delta_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    shp_gdf=None,
    min_count: int = 5,
) -> None:
    """Hero map: where PMST improves station recall over IFS diagnostic visibility."""

    if station_delta_df is None or station_delta_df.empty:
        print("  [WARN] No station PMST-vs-IFS delta rows; skip station delta map.", flush=True)
        return
    setup_journal_style()
    metric_specs = [
        ("delta_fog_recall", "Fog recall", "n_fog"),
        ("delta_mist_recall", "Mist recall", "n_mist"),
        ("delta_low_vis_recall", "Low-vis recall", "n_low_vis"),
    ]
    available = [(m, label, count) for m, label, count in metric_specs if m in station_delta_df and count in station_delta_df]
    if not available:
        print("  [WARN] Station delta table lacks recall-delta columns; skip station delta map.", flush=True)
        return

    metric_lookup = {metric: (label, count_col) for metric, label, count_col in available}
    metric = "delta_low_vis_recall" if "delta_low_vis_recall" in metric_lookup else available[0][0]
    label, count_col = metric_lookup[metric]

    df = station_delta_df.copy()
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[(df[count_col] >= min_count) & np.isfinite(df[metric])].copy()
    if df.empty:
        print("  [WARN] Station delta map has no finite values after count mask.", flush=True)
        return

    vals = df[metric].to_numpy(dtype=float)
    lim = _nice_delta_limit(vals)
    hist_lo, hist_hi = _nice_hist_limits(vals, lim)
    cmap = LinearSegmentedColormap.from_list(
        "pmst_ifs_recall_delta",
        [
            (0.00, "#9E1F36"),
            (0.24, "#D6604D"),
            (0.50, "#FFFFFF"),
            (0.76, "#67A9CF"),
            (1.00, "#08306B"),
        ],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    draw_basemap(ax, shp_gdf, compact=True)
    count_vals = np.clip(df[count_col].to_numpy(dtype=float), 0, None)
    count_ref = float(np.nanpercentile(count_vals, 95)) if count_vals.size else 1.0
    count_ref = max(1.0, count_ref)
    sizes = 14.0 + 28.0 * np.sqrt(np.minimum(count_vals, count_ref) / count_ref)
    order = np.argsort(np.abs(vals))
    df_plot = df.iloc[order]
    sc = ax.scatter(
        df_plot["lon"],
        df_plot["lat"],
        c=df_plot[metric],
        s=sizes[order],
        marker="D",
        cmap=cmap,
        norm=norm,
        linewidths=0.45,
        edgecolors="#6F6F6F",
        alpha=0.95,
        zorder=3,
    )
    draw_boundary(ax, shp_gdf, color="#1F2937", linewidth=0.60, zorder=6)

    n_station = int(len(df))
    better = int((df[metric] > 0).sum())
    worse = int((df[metric] < 0).sum())
    median_delta = float(np.nanmedian(vals))
    mean_delta = float(np.nanmean(vals))
    ax.set_title(f"Station-level {label.lower()} difference", fontsize=11, fontweight="bold", pad=8)
    ax.text(
        0.02,
        0.98,
        f"PMST better: {safe_div(better * 100.0, n_station):.0f}%\n"
        f"IFS better: {safe_div(worse * 100.0, n_station):.0f}%\n"
        f"median delta={median_delta:+.2f}; n={n_station:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1F2937",
        bbox={"facecolor": "white", "edgecolor": "#B8B8B8", "linewidth": 0.4, "alpha": 0.88, "pad": 3.0},
        zorder=8,
    )

    hist_ax = ax.inset_axes([0.055, 0.055, 0.35, 0.26])
    bins = np.linspace(hist_lo, hist_hi, 19)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    for center, height, width in zip(centers, hist, widths):
        hist_ax.bar(
            center,
            height,
            width=width * 0.94,
            color=cmap(norm(np.clip(center, -lim, lim))),
            edgecolor="#383838",
            linewidth=0.25,
            alpha=0.92,
        )
    grid = np.linspace(hist_lo, hist_hi, 240)
    density = _gaussian_density_curve(vals, grid)
    if density.size:
        hist_ax.plot(grid, density, color="#1F2937", linewidth=1.1)
    hist_ax.axvline(0.0, color="#C62828", linestyle="--", linewidth=1.0)
    hist_ax.axvline(mean_delta, color="#1F2937", linestyle="-", linewidth=0.9)
    hist_ax.set_xlim(hist_lo, hist_hi)
    hist_ax.set_xlabel("Delta recall", fontsize=7)
    hist_ax.set_ylabel("Density", fontsize=7)
    hist_ax.tick_params(axis="both", labelsize=7, length=2.5)
    hist_ax.grid(alpha=0.18)
    for spine in hist_ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.5)

    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.055, pad=0.045, extend="both")
    cb.set_ticks(np.linspace(-lim, lim, 5))
    cb.set_label("Recall difference (PMST - IFS diagnostic VIS)")
    cb.ax.text(
        0.0,
        -1.35,
        "IFS better",
        transform=cb.ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#9E1F36",
    )
    cb.ax.text(
        1.0,
        -1.35,
        "PMST better",
        transform=cb.ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#08306B",
    )
    fig.tight_layout()
    notes = (
        f"Station-level {label.lower()} deltas on finite IFS-matched rows; "
        "blue means PMST recall exceeds IFS diagnostic visibility recall, "
        "red means the reverse. Inset histogram is density-normalized."
    )
    save_fig_pair(
        fig,
        out_dir,
        "fig8_station_model_vs_ifs_recall_delta",
        manifest,
        sources,
        notes=notes,
        n=int(len(station_delta_df)),
    )


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
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ymax = max(ymax, int(np.nanmax(vals)))
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
    if lead_pooled.empty and lead00.empty and lead12.empty:
        print("  [WARN] Empty 48h lead tables; skip init-hour figure.", flush=True)
        return
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    specs = [
        ("Fog_CSI", "Fog CSI"),
        ("Fog_R", "Fog recall"),
        ("Mist_CSI", "Mist CSI"),
        ("Mist_R", "Mist recall"),
        ("low_vis_csi", "Low-vis CSI"),
        ("low_vis_recall", "Low-vis recall"),
    ]
    for ax, (metric, title), letter in zip(axes.ravel(), specs, "abcdef"):
        plotted = False
        plotted = _plot_lead_metric_series(ax, lead_pooled, metric, "#111827", "Pooled", "o", 2.2, 3) or plotted
        plotted = _plot_lead_metric_series(ax, lead00, metric, "#0F766E", "00Z", "o", 1.5, 4) or plotted
        plotted = _plot_lead_metric_series(ax, lead12, metric, "#C2410C", "12Z", "s", 1.5, 4) or plotted
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="#6B7280")
        ax.set_title(title)
        ax.set_xlabel("Display lead time (h)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, 48.5)
        ax.axvspan(-0.5, 12.0, color="#EEF7F0", alpha=0.55, zorder=-10)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.10)
        add_panel_label(ax, letter)
    handles, labels = [], []
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(
        0.995,
        0.01,
        "Dotted 0-12 h segment is filled from the previous initialization's 12-24 h verification window.",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#3F4A3F",
    )
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


def parse_compact_datetime(values, index=None) -> pd.Series:
    s = pd.Series(values, index=index)
    raw = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact_mask = raw.str.match(r"^\d{10}$", na=False)
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if compact_mask.any():
        dt.loc[compact_mask] = pd.to_datetime(raw.loc[compact_mask], format="%Y%m%d%H", errors="coerce")
    if (~compact_mask).any():
        dt.loc[~compact_mask] = pd.to_datetime(raw.loc[~compact_mask], errors="coerce")
    return dt


def parse_init_hour(meta: pd.DataFrame) -> pd.Series:
    if "init_hour" in meta:
        return pd.to_numeric(meta["init_hour"], errors="coerce")
    if "init_time" not in meta:
        return pd.Series(np.nan, index=meta.index)
    return parse_compact_datetime(meta["init_time"], index=meta.index).dt.hour


def infer_init_cycle_hour(
    meta: pd.DataFrame,
    local_time_offset_hours: int = LOCAL_TIME_OFFSET_HOURS,
) -> Tuple[pd.Series, str]:
    """Return a 00Z/12Z forecast-cycle hour, tolerating local 08/20 encodings."""

    raw = pd.to_numeric(parse_init_hour(meta), errors="coerce")
    candidates: List[Tuple[str, pd.Series]] = [("raw", raw)]
    offset = int(local_time_offset_hours or 0) % 24
    if offset:
        candidates.append((f"raw_minus_{offset}h", (raw - offset) % 24))
        candidates.append((f"raw_plus_{offset}h", (raw + offset) % 24))

    best_name, best_series, best_score = candidates[0][0], candidates[0][1], -1
    for name, series in candidates:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(arr)
        rounded = np.full(len(arr), -9999, dtype=np.int32)
        rounded[finite] = (np.rint(arr[finite]).astype(np.int32) % 24)
        score = int(np.isin(rounded[finite], [0, 12]).sum())
        if score > best_score:
            best_name, best_series, best_score = name, series, score

    arr = pd.to_numeric(best_series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    finite = np.isfinite(arr)
    out[finite] = np.rint(arr[finite]) % 24
    return pd.Series(out, index=meta.index, name="init_cycle_hour"), best_name


def init_cycle_mask(init_cycle_hour: pd.Series, target_hour: int) -> np.ndarray:
    arr = pd.to_numeric(init_cycle_hour, errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(arr), dtype=bool)
    finite = np.isfinite(arr)
    mask[finite] = (np.rint(arr[finite]).astype(np.int32) % 24) == int(target_hour)
    return mask


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


def build_display_lead_table(
    native: pd.DataFrame,
    fill_from: Optional[pd.DataFrame] = None,
    fill_source: str = "previous_init_12_24h",
    fill_min_hour: int = 12,
    fill_max_hour: int = 24,
) -> pd.DataFrame:
    """Add display lead 0-12 h by shifting another table's 12-24 h rows.

    The underlying 48 h dataset starts at 12 h. For a continuous lead-time figure,
    the previous initialization's 12-24 h segment is a defensible proxy for the
    current initialization's 0-12 h display segment.
    """

    if native is None or native.empty or "lead_hour" not in native:
        return pd.DataFrame()
    out = native.copy()
    out["native_lead_hour"] = pd.to_numeric(out["lead_hour"], errors="coerce")
    out["display_lead_hour"] = out["native_lead_hour"]
    out["lead_fill_source"] = "native_12_48h"
    src = native if fill_from is None else fill_from
    fill = pd.DataFrame()
    if src is not None and not src.empty and "lead_hour" in src:
        fill = src.copy()
        raw = pd.to_numeric(fill["lead_hour"], errors="coerce")
        fill = fill[(raw >= float(fill_min_hour)) & (raw < float(fill_max_hour))].copy()
        if not fill.empty:
            fill["native_lead_hour"] = pd.to_numeric(fill["lead_hour"], errors="coerce")
            fill["display_lead_hour"] = fill["native_lead_hour"] - float(fill_min_hour)
            fill["lead_hour"] = fill["display_lead_hour"]
            fill["lead_fill_source"] = fill_source
    if fill.empty:
        return out.sort_values(["display_lead_hour", "lead_fill_source"]).reset_index(drop=True)
    combined = pd.concat([fill, out], ignore_index=True, sort=False)
    return combined.sort_values(["display_lead_hour", "lead_fill_source"]).reset_index(drop=True)


def _lead_x(df: pd.DataFrame) -> pd.Series:
    if "display_lead_hour" in df:
        return pd.to_numeric(df["display_lead_hour"], errors="coerce")
    return pd.to_numeric(df["lead_hour"], errors="coerce")


def _plot_lead_metric_series(
    ax,
    df: pd.DataFrame,
    metric: str,
    color: str,
    label: str,
    marker: str,
    lw: float,
    zorder: int,
    linestyle: str = "-",
) -> bool:
    if df is None or df.empty or metric not in df:
        return False
    plotted = False
    d = df.copy()
    d["_x"] = _lead_x(d)
    d["_y"] = pd.to_numeric(d[metric], errors="coerce")
    d = d[np.isfinite(d["_x"]) & np.isfinite(d["_y"])].sort_values("_x")
    if d.empty:
        return False
    fill_col = d["lead_fill_source"].astype(str) if "lead_fill_source" in d else pd.Series("native_12_48h", index=d.index)
    native = d[fill_col == "native_12_48h"]
    front = d[fill_col != "native_12_48h"]
    if not front.empty:
        ax.plot(
            front["_x"],
            front["_y"],
            color=color,
            lw=max(1.1, lw - 0.4),
            ls=":",
            marker=marker,
            ms=2.8,
            mfc="white",
            mec=color,
            alpha=0.82,
            label=None,
            zorder=zorder - 1,
        )
        plotted = True
    if not native.empty:
        ax.plot(
            native["_x"],
            native["_y"],
            color=color,
            lw=lw,
            ls=linestyle,
            marker=marker,
            ms=3.0,
            label=label,
            zorder=zorder,
        )
        plotted = True
    elif not d.empty:
        ax.plot(d["_x"], d["_y"], color=color, lw=lw, ls=linestyle, marker=marker, ms=3.0, label=label, zorder=zorder)
        plotted = True
    return plotted


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

        pair_lead = np.tile(lead_vals, n_rec).astype(np.int32)
        meta_lead = pd.to_numeric(meta["lead_hour"], errors="coerce").to_numpy(dtype=float)
        finite_lead = np.isfinite(meta_lead)
        meta_lead_i = np.full(len(meta), -9999, dtype=np.int32)
        meta_lead_i[finite_lead] = np.rint(meta_lead[finite_lead]).astype(np.int32)

        station_lookup = pd.Series(np.arange(len(stations), dtype=np.int64), index=stations)
        meta_station = normalize_station_ids(meta["station_id"].values)
        station_pos = meta_station.map(station_lookup).to_numpy()

        pair_values_full = np.arange(n_rec * n_lead, dtype=np.int64)
        candidates: List[Dict[str, object]] = []

        def _add_match_candidate(label: str, ifs_key_ns: np.ndarray, meta_key_ns: np.ndarray) -> None:
            pair_index = pd.MultiIndex.from_arrays([ifs_key_ns, pair_lead])
            pair_values = pair_values_full
            if not pair_index.is_unique:
                keep = ~pair_index.duplicated(keep="first")
                pair_index = pair_index[keep]
                pair_values = pair_values[keep]
            meta_index = pd.MultiIndex.from_arrays([meta_key_ns, meta_lead_i])
            pair_pos_cand = pd.Series(pair_values, index=pair_index).reindex(meta_index).to_numpy()
            pair_match = pd.notna(pair_pos_cand)
            exact_key = pair_match & pd.notna(station_pos)
            candidates.append(
                {
                    "label": label,
                    "pair_pos": pair_pos_cand,
                    "n_pair_matches": int(np.sum(pair_match)),
                    "n_exact_key_matches": int(np.sum(exact_key)),
                }
            )

        if "forecast_reference_time" in ds.coords and "init_time" in meta:
            init_raw = np.asarray(ds["forecast_reference_time"].values).reshape(-1)
            if init_raw.shape[0] == n_rec:
                ifs_init_ns = np.repeat(pd.to_datetime(init_raw).values.astype("datetime64[ns]").astype(np.int64), n_lead)
                meta_init_dt = parse_compact_datetime(meta["init_time"], index=meta.index)
                meta_init_ns = meta_init_dt.values.astype("datetime64[ns]").astype(np.int64)
                if not meta_init_dt.isna().all():
                    _add_match_candidate("init_time+lead", ifs_init_ns, meta_init_ns)

        valid_time_grid = recover_ifs_48h_valid_time_grid(ds, n_rec, n_lead, lead_vals)
        if valid_time_grid.shape != (n_rec, n_lead):
            raise ValueError(f"valid_time_grid shape={valid_time_grid.shape}, expected {(n_rec, n_lead)}")
        pair_time_ns = valid_time_grid.reshape(-1).astype("datetime64[ns]").astype(np.int64)
        meta_time = pd.to_datetime(meta["time"], errors="coerce").values.astype("datetime64[ns]").astype(np.int64)
        _add_match_candidate("valid_time+lead", pair_time_ns, meta_time)

        if not candidates:
            raise ValueError("Cannot build any 48h IFS match keys from init_time or valid_time coordinates.")
        best = max(candidates, key=lambda c: (int(c["n_exact_key_matches"]), int(c["n_pair_matches"])))
        pair_pos = np.asarray(best["pair_pos"])
        key_valid = pd.notna(pair_pos) & pd.notna(station_pos)

        raw = np.full(len(meta), np.nan, dtype=np.float64)
        pred = np.full(len(meta), -1, dtype=np.int64)
        valid = np.zeros(len(meta), dtype=bool)
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
            "match_key": str(best["label"]),
            "n_pair_matches": float(best["n_pair_matches"]),
            "n_exact_key_matches": float(np.sum(key_valid)),
            "n_finite_matches": float(np.sum(valid)),
            "pair_match_ratio": safe_div(float(best["n_pair_matches"]), float(len(meta))),
            "exact_match_ratio": safe_div(float(np.sum(key_valid)), float(len(meta))),
            "finite_match_ratio": safe_div(float(np.sum(valid)), float(len(meta))),
        }
        print(
            f"[48h IFS] match key={best['label']}; matched finite rows: {int(valid.sum())}/{len(meta)} from {ifs_nc_path}",
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
        ("Fog_R", "Fog recall"),
        ("Mist_CSI", "Mist CSI"),
        ("Mist_R", "Mist recall"),
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

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    for ax, (metric, title), letter in zip(axes.ravel(), specs, "abcdef"):
        model_col = f"{metric}_model"
        ifs_col = f"{metric}_ifs"
        panel_values: List[float] = []
        plotted = False
        if model_col in cmp_df:
            y_model = cmp_df[model_col].to_numpy(dtype=float)
            panel_values.extend(y_model.tolist())
            plotted = _plot_lead_metric_series(ax, cmp_df, model_col, PMST_COLOR, "PMST", "o", 2.2, 4) or plotted
        if ifs_col in cmp_df:
            y_ifs = cmp_df[ifs_col].to_numpy(dtype=float)
            panel_values.extend(y_ifs.tolist())
            plotted = _plot_lead_metric_series(ax, cmp_df, ifs_col, IFS_DIAG_COLOR, "IFS diagnostic", "s", 1.9, 4, linestyle="--") or plotted
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="#6B7280")
        ax.set_title(title)
        ax.set_xlabel("Display lead time (h)")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, _adaptive_ylim(panel_values))
        ax.set_xlim(-0.5, 48.5)
        ax.axvspan(-0.5, 12.0, color="#EEF7F0", alpha=0.55, zorder=-10)
        ax.axhline(0, color="#D1D5DB", lw=0.8, zorder=1)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.10)
        add_panel_label(ax, letter)
    handles, labels = [], []
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(
        0.995,
        0.01,
        "Dotted 0-12 h segment is filled from the previous initialization's 12-24 h verification window.",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#3F4A3F",
    )
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
    print(f"[48h] data_dir: {data_48h}", flush=True)
    print("[48h] expected figures: fig11_48h_lead_init_00Z_12Z and fig11_48h_model_vs_ifs_by_lead", flush=True)
    try:
        x_path, y_cls, _, meta = load_main_data(
            data_48h, args.limit_samples, getattr(args, "meta_time_shift_hours", 0.0)
        )
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
        init_cycle_hour, init_cycle_source = infer_init_cycle_hour(meta, args.local_time_offset_hours)
        if init_cycle_hour.isna().all():
            print("[48h] meta_test.csv has no parseable init_time/init_hour; skip fig11.", flush=True)
            return
        probs, low_prob = run_model_inference(
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
            return_low_prob=(args.decision_rule == "binary_gate"),
        )
        pred = pred_from_decision_rule(
            probs,
            low_prob,
            args.decision_rule,
            args.lead_fog_th,
            args.lead_mist_th,
            args.threshold_rule,
            args.lead_lowvis_gate_th,
        )
        lead = pd.to_numeric(meta["lead_hour"], errors="coerce").to_numpy(dtype=float)
        pooled = lead_metrics_table(y_cls, pred, probs, lead)
        mask00 = init_cycle_mask(init_cycle_hour, 0)
        mask12 = init_cycle_mask(init_cycle_hour, 12)
        print(
            f"[48h] init cycle source={init_cycle_source}; rows 00Z={int(mask00.sum())}, 12Z={int(mask12.sum())}",
            flush=True,
        )
        lead00 = lead_metrics_table(y_cls, pred, probs, lead, mask=mask00)
        lead12 = lead_metrics_table(y_cls, pred, probs, lead, mask=mask12)
        pooled.to_csv(out_dir / "metrics_by_lead_hour_48h_model.csv", index=False)
        lead00.to_csv(out_dir / "metrics_by_lead_hour_init00Z.csv", index=False)
        lead12.to_csv(out_dir / "metrics_by_lead_hour_init12Z.csv", index=False)
        pooled_display = build_display_lead_table(pooled, pooled, "pooled_previous_init_12_24h")
        lead00_display = build_display_lead_table(lead00, lead12, "previous_12Z_init_12_24h")
        lead12_display = build_display_lead_table(lead12, lead00, "previous_00Z_init_12_24h")
        pooled_display.to_csv(out_dir / "metrics_by_display_lead_hour_48h_model.csv", index=False)
        lead00_display.to_csv(out_dir / "metrics_by_display_lead_hour_init00Z.csv", index=False)
        lead12_display.to_csv(out_dir / "metrics_by_display_lead_hour_init12Z.csv", index=False)
        plot_fig11_lead_init(
            pooled_display,
            lead00_display,
            lead12_display,
            out_dir,
            manifest,
            [
                str(x_path),
                str(data_48h / "meta_test.csv"),
                str(out_dir / "metrics_by_display_lead_hour_48h_model.csv"),
            ],
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
                    print(
                        f"[48h IFS] lead table rows: model={len(model_matched)}, ifs={len(ifs_lead)}, merged={len(cmp_df)}",
                        flush=True,
                    )
                    for metric in (
                        "Fog_CSI",
                        "Fog_R",
                        "Mist_CSI",
                        "Mist_R",
                        "low_vis_csi",
                        "low_vis_recall",
                    ):
                        model_col = f"{metric}_model"
                        ifs_col = f"{metric}_ifs"
                        if model_col in cmp_df and ifs_col in cmp_df:
                            cmp_df[f"{metric}_diff_model_minus_ifs"] = cmp_df[model_col] - cmp_df[ifs_col]
                    cmp_df.to_csv(cmp_path, index=False, float_format="%.6f")
                    cmp_display = build_display_lead_table(cmp_df, cmp_df, "matched_previous_init_12_24h")
                    cmp_display_path = out_dir / "model_vs_ifs_metrics_by_display_lead_hour_48h.csv"
                    cmp_display.to_csv(cmp_display_path, index=False, float_format="%.6f")
                    print(f"[table] {model_matched_path}", flush=True)
                    print(f"[table] {ifs_lead_path}", flush=True)
                    print(f"[table] {cmp_path}", flush=True)
                    print(f"[table] {cmp_display_path}", flush=True)
                    plot_fig11_48h_model_vs_ifs(
                        cmp_display,
                        out_dir,
                        manifest,
                        [str(x_path), str(data_48h / "meta_test.csv"), str(ifs_48h_nc), str(cmp_display_path)],
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


def _probe_overlap_cli_options(script: Path, options: Sequence[str]) -> Dict[str, bool]:
    """Return whether an overlap evaluator advertises optional CLI switches."""
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        help_text = proc.stdout or ""
        if not help_text:
            return {opt: True for opt in options}
        return {opt: opt in help_text for opt in options}
    except Exception as exc:
        print(f"[overlap] could not inspect evaluator CLI ({exc}); pass optional arguments as configured.", flush=True)
        return {opt: True for opt in options}


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
    supported = _probe_overlap_cli_options(
        overlap_script,
        ["--feature_importance_csv", "--feature_swap_top_k", "--feature_swap_features"],
    )
    if args.limit_samples and args.limit_samples > 0:
        cmd.extend(["--limit_samples", str(args.limit_samples)])
        cmd.extend(["--bootstrap", "50", "--bootstrap_size", str(min(args.limit_samples, 20000))])
    for arg_name, cli_name in (
        ("overlap_tianji_ckpt", "--tianji_ckpt"),
        ("overlap_ifs_ckpt", "--ifs_ckpt"),
        ("overlap_tianji_scaler", "--tianji_scaler"),
        ("overlap_ifs_scaler", "--ifs_scaler"),
    ):
        value = str(getattr(args, arg_name, "") or "").strip()
        if value:
            cmd.extend([cli_name, str(abs_under_base(base, value))])
    if str(getattr(args, "overlap_feature_importance_csv", "") or "").strip():
        if supported.get("--feature_importance_csv", True):
            cmd.extend(
                [
                    "--feature_importance_csv",
                    str(abs_under_base(base, args.overlap_feature_importance_csv)),
                ]
            )
        else:
            print("[overlap] target evaluator lacks --feature_importance_csv; skip that optional argument.", flush=True)
    if int(getattr(args, "overlap_feature_swap_top_k", 0) or 0) > 0:
        if supported.get("--feature_swap_top_k", True):
            cmd.extend(["--feature_swap_top_k", str(int(args.overlap_feature_swap_top_k))])
        else:
            print("[overlap] target evaluator lacks --feature_swap_top_k; source comparison will run without feature replacement.", flush=True)
    if str(getattr(args, "overlap_feature_swap_features", "") or "").strip():
        if supported.get("--feature_swap_features", True):
            cmd.extend(["--feature_swap_features", str(args.overlap_feature_swap_features)])
        else:
            print("[overlap] target evaluator lacks --feature_swap_features; skip explicit replacement feature list.", flush=True)
    if args.skip_overlap_bootstrap:
        cmd.append("--skip_bootstrap")
    if args.overlap_extra_args.strip():
        cmd.extend(shlex.split(args.overlap_extra_args.strip()))
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


def run_key_variable_quality_subprocess(
    args: argparse.Namespace,
    base: Path,
    out_dir: Path,
    manifest: Optional[Manifest] = None,
) -> Optional[Path]:
    if not bool(getattr(args, "run_variable_quality", False)):
        return None
    script = (
        abs_under_base(base, getattr(args, "variable_quality_script", ""))
        if str(getattr(args, "variable_quality_script", "") or "").strip()
        else VIS_EVAL_DIR / "analyze_key_variable_quality.py"
    )
    tianji_dir = abs_under_base(base, getattr(args, "quality_tianji_data_dir", ""))
    ifs_dir = abs_under_base(base, getattr(args, "quality_ifs_data_dir", ""))
    obs_root = abs_under_base(base, getattr(args, "obs_root", ""))
    q_out = (
        abs_under_base(base, getattr(args, "quality_out_dir", ""))
        if str(getattr(args, "quality_out_dir", "") or "").strip()
        else out_dir / "key_variable_quality"
    )
    missing = []
    if not script.is_file():
        missing.append(f"script={script}")
    if not tianji_dir.is_dir():
        missing.append(f"tianji_data_dir={tianji_dir}")
    if not ifs_dir.is_dir():
        missing.append(f"ifs_data_dir={ifs_dir}")
    if not obs_root.is_dir():
        missing.append(f"obs_root={obs_root}")
    if missing:
        print("[variable-quality] skipped; missing " + "; ".join(missing), flush=True)
        return None
    cmd = [
        sys.executable,
        str(script),
        "--tianji_data_dir",
        str(tianji_dir),
        "--ifs_data_dir",
        str(ifs_dir),
        "--obs_root",
        str(obs_root),
        "--out_dir",
        str(q_out),
        "--features",
        str(getattr(args, "quality_features", "RH2M,T2M,WSPD10,MSLP,PRECIP")),
        "--window",
        str(getattr(args, "window_size", 12)),
        "--limit_samples",
        str(getattr(args, "limit_samples", 0) or 0),
    ]
    print("[variable-quality] running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[variable-quality] failed with exit code {exc.returncode}; continuing.", flush=True)
        return q_out
    if manifest is not None:
        sources = [
            str(q_out / "key_variable_quality_metrics.csv"),
            str(q_out / "key_variable_quality_samples.csv"),
            str(tianji_dir / "meta_test.csv"),
            str(ifs_dir / "meta_test.csv"),
        ]
        fig_path = q_out / "fig_key_variable_quality_tianji_vs_ifs.png"
        if fig_path.exists():
            manifest.add(
                fig_path.name,
                sources,
                notes="Key meteorological-variable forecast quality against station observations; observation time zone is auto-selected by match coverage.",
            )
    return q_out


def run_main(args: argparse.Namespace, base: Path, out_dir: Path, manifest: Manifest):
    import torch
    import joblib

    data_dir = abs_under_base(base, args.data_dir)
    ckpt_path = abs_under_base(base, args.ckpt_path)
    scaler_path = abs_under_base(base, args.scaler_path)
    season_th_path = abs_under_base(base, args.season_th_path)
    ifs_nc = abs_under_base(base, args.ifs_vis_nc)
    model_py = resolve_model_py(base, args.model_py)
    print(f"[profile] S2 run id: {getattr(args, 'config_s2_run_id', DEFAULT_S2_RUN_ID)}", flush=True)
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
    x_path, y_cls, y_raw, meta = load_main_data(
        data_dir, args.limit_samples, getattr(args, "meta_time_shift_hours", 0.0)
    )
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

    season_thresholds, season_temp = load_saved_season_thresholds(season_th_path)
    temperature = None
    if args.use_calibration and season_temp is not None:
        temperature = season_temp
        print(f"[calibration] temperature={temperature} from {season_th_path}", flush=True)

    probs, low_prob = run_model_inference(
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
        return_low_prob=(args.decision_rule == "binary_gate" or args.write_decision_comparison),
    )
    pmst_pred = pred_from_decision_rule(
        probs,
        low_prob,
        args.decision_rule,
        args.fog_th,
        args.mist_th,
        args.threshold_rule,
        args.lowvis_gate_th,
    )
    pmst_metrics = classification_metrics(y_cls, pmst_pred, probs=probs)
    selection_json_path = abs_under_base(base, args.selection_result_json) if args.selection_result_json else None
    primary_decision_meta = {
        "decision_rule": args.decision_rule,
        "fog_threshold": args.fog_th,
        "mist_threshold": args.mist_th,
        "threshold_rule": args.threshold_rule,
        "lowvis_gate_threshold": args.lowvis_gate_th if args.decision_rule == "binary_gate" else np.nan,
        "selection_result_json": str(selection_json_path) if selection_json_path else "",
    }
    if args.write_decision_comparison:
        comparison_rows = [
            (
                "current_fine_threshold",
                classification_metrics(
                    y_cls,
                    pred_from_probs_rule(probs, args.fog_th, args.mist_th, args.threshold_rule),
                    probs=probs,
                ),
                {
                    "decision_rule": "fine_threshold",
                    "fog_threshold": args.fog_th,
                    "mist_threshold": args.mist_th,
                    "threshold_rule": args.threshold_rule,
                },
            )
        ]
        if low_prob is not None:
            comparison_rows.append(
                (
                    "binary_gate_fine_argmax",
                    classification_metrics(y_cls, binary_gate_pred(probs, low_prob, args.lowvis_gate_th), probs=probs),
                    {
                        "decision_rule": "binary_gate",
                        "binary_threshold": args.lowvis_gate_th,
                        "selection_result_json": str(selection_json_path) if selection_json_path else "",
                    },
                )
            )
        if season_thresholds:
            fog_arr, mist_arr = thresholds_from_saved_seasons(meta, season_thresholds, args.fog_th, args.mist_th)
            comparison_rows.append(
                (
                    "saved_season_thresholds",
                    classification_metrics(y_cls, pred_from_thresholds_mutual(probs, fog_arr, mist_arr), probs=probs),
                    {
                        "decision_rule": "season_thresholds",
                        "threshold_rule": "mutual",
                        "season_threshold_path": str(season_th_path),
                    },
                )
            )
        comparison_df = metrics_rows(comparison_rows)
        comparison_df.to_csv(out_dir / "decision_comparison_metrics.csv", index=False)
        comparison_json = {
            "experiment_id": args.experiment_id or getattr(args, "config_run_tag", ""),
            "s2_run_id": getattr(args, "config_s2_run_id", DEFAULT_S2_RUN_ID),
            "primary_decision": primary_decision_meta,
            "comparison_source": {
                "data_dir": str(data_dir),
                "checkpoint": str(ckpt_path),
                "scaler": str(scaler_path),
                "season_threshold_path": str(season_th_path),
            },
            "comparison_rows": comparison_df.to_dict(orient="records"),
        }
        with open(out_dir / "decision_comparison_metrics.json", "w", encoding="utf-8") as f:
            json.dump(comparison_json, f, ensure_ascii=False, indent=2)
        print(f"[table] {out_dir / 'decision_comparison_metrics.csv'}", flush=True)
        print(f"[json] {out_dir / 'decision_comparison_metrics.json'}", flush=True)
    overall_df = metrics_rows(
        [
            (
                "pmst",
                pmst_metrics,
                {
                    "sample_scope": "test",
                    **primary_decision_meta,
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
                    ("pmst", matched_pmst_metrics, {"sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid)), **primary_decision_meta}),
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
                    "Mist_CSI",
                    "Mist_R",
                    "low_vis_csi",
                    "low_vis_recall",
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
        low_prob=low_prob,
        ifs_pred=ifs_pred,
        ifs_vis=ifs_vis,
        ifs_valid=ifs_valid,
    )
    write_report(out_dir / "rare_event_report.txt", y_cls, pmst_pred, pmst_metrics, ifs_metrics)

    station_df = aggregate_station_metrics(eval_df, "pmst_pred")
    station_df.to_csv(out_dir / "station_metrics.csv", index=False)
    station_delta_df = aggregate_station_model_vs_ifs_metrics(eval_df)
    if station_delta_df is not None and not station_delta_df.empty:
        station_delta_df.to_csv(out_dir / "station_model_vs_ifs_metrics.csv", index=False, float_format="%.6f")
        print(f"[table] {out_dir / 'station_model_vs_ifs_metrics.csv'}", flush=True)
    scenario_df = build_scenario_metrics(eval_df)
    scenario_df.to_csv(out_dir / "scenario_metrics.csv", index=False)

    setup_journal_style()
    sources_main = [str(x_path), str(data_dir / "y_test.npy"), str(data_dir / "meta_test.csv"), str(ckpt_path), str(scaler_path)]
    if selection_json_path:
        sources_main.append(str(selection_json_path))
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
    if station_delta_df is not None and not station_delta_df.empty:
        plot_station_recall_delta_map(
            station_delta_df,
            out_dir,
            manifest,
            [str(out_dir / "station_model_vs_ifs_metrics.csv")],
            shp_gdf=shp_gdf,
            min_count=5,
        )
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
        "low_vis_csi",
        "n_low_vis",
        5,
        "Station Low-Visibility CSI",
        "fig8_station_low_vis_csi",
        out_dir,
        manifest,
        [str(out_dir / "station_metrics.csv")],
        shp_gdf=shp_gdf,
        cmap="cividis",
        vmin=0,
        vmax=1,
    )

    run_48h_optional(args, base, out_dir, model, device, scaler, manifest)

    event_df = pd.DataFrame()
    event_eval_completed = False
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
            event_eval_completed = True
        except Exception as exc:
            print(f"[events] run_widespread_event_evaluation failed: {exc}", flush=True)
    event_path = out_dir / "event_case_summary.csv"
    if event_eval_completed and event_path.exists():
        event_df = pd.read_csv(event_path, parse_dates=["peak_time", "start_time", "end_time"])
    elif event_df is not None and not event_df.empty:
        event_df.to_csv(event_path, index=False)
    elif (not event_eval_completed) and event_path.exists():
        print(f"[events] ignoring stale event summary after failed/skipped event evaluation: {event_path}", flush=True)
    if "ifs_diagnostic_pred" not in eval_df and ifs_pred is not None:
        eval_df["ifs_diagnostic_pred"] = ifs_pred
        eval_df["ifs_diagnostic_valid"] = ifs_valid
    if event_df is not None and not event_df.empty:
        event_sources = [str(event_path), str(out_dir / "per_sample_eval.csv")]
        if plot_three_events_footprint_row is not None:
            three_footprint_path = out_dir / "fig_three_events_footprint_row.png"
            if plot_three_events_footprint_row(
                meta,
                y_raw,
                pmst_pred,
                event_df,
                str(three_footprint_path),
                shp_gdf=shp_gdf,
                window_hours=args.event_window_hours,
            ) is not None:
                manifest.add(
                    three_footprint_path.name,
                    event_sources,
                    notes="Three selected widespread fog events with complete test-set windows where available.",
                    n=int(len(y_cls)),
                    matched_ifs=int(np.sum(ifs_valid)) if ifs_valid is not None else None,
                )
        if plot_three_events_peak_row is not None:
            three_peak_path = out_dir / "fig_three_events_peak_row.png"
            if plot_three_events_peak_row(
                meta,
                y_raw,
                pmst_pred,
                event_df,
                str(three_peak_path),
                shp_gdf=shp_gdf,
            ) is not None:
                manifest.add(
                    three_peak_path.name,
                    event_sources,
                    notes="Observed visibility at the peak hour for the same three selected widespread fog events.",
                    n=int(len(y_cls)),
                )
    plot_event_peak_grid(eval_df, event_df, out_dir, manifest, [str(event_path), str(out_dir / "per_sample_eval.csv")], shp_gdf=shp_gdf)
    hourly_paths = [out_dir / f"fig9_event_{k}_hourly_metrics.csv" for k in (1, 2, 3)]
    plot_event_footprint(hourly_paths, out_dir, manifest, [str(p) for p in hourly_paths])
    run_key_variable_quality_subprocess(args, base, out_dir, manifest)

    run_config = {
        "experiment_id": args.experiment_id or getattr(args, "config_run_tag", ""),
        "args": vars(args),
        "config_json": getattr(args, "config_json", ""),
        "config_run_tag": getattr(args, "config_run_tag", ""),
        "primary_decision": primary_decision_meta,
        "decision_comparison_metrics_json": str(out_dir / "decision_comparison_metrics.json")
        if args.write_decision_comparison
        else "",
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


def run_lead48_mode(args: argparse.Namespace, base: Path, out_dir: Path, manifest: Manifest) -> None:
    import torch
    import joblib

    ckpt_path = abs_under_base(base, args.ckpt_path)
    scaler_path = abs_under_base(base, args.scaler_path)
    model_py = resolve_model_py(base, args.model_py)
    data_48h = abs_under_base(base, args.data_48h_dir)
    if not data_48h.is_dir():
        fallback = base / "ml_dataset_fe_12h_48h_pm10_pm25"
        if fallback.is_dir():
            data_48h = fallback
    require_existing_path(data_48h, "data_48h_dir", expect_dir=True)
    require_existing_path(ckpt_path, "checkpoint")
    require_existing_path(scaler_path, "scaler")
    require_existing_path(model_py, "model_py")

    x_path = data_48h / "X_test.npy"
    dyn, fe = infer_layout_from_x(x_path, args.window_size)
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
    run_48h_optional(args, base, out_dir, model, device, scaler, manifest)


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
    if getattr(args, "config_loaded", False):
        print(f"config   : {args.config_json}", flush=True)
        print(f"run_tag  : {getattr(args, 'config_run_tag', '')}", flush=True)
    print("=" * 72, flush=True)

    if args.mode == "tables":
        run_tables_mode(args, base, out_dir, manifest)
    if args.mode == "lead48":
        run_lead48_mode(args, base, out_dir, manifest)
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
