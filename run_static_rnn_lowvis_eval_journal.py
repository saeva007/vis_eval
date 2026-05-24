#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Static-MLP + RNN low-visibility checkpoints with journal figures.

This script is the evaluation companion for
``train_static_rnn_lowvis.py``. It reuses the paper-evaluation metric and
figure helpers from ``run_paper_eval_pm10_pm25_journal.py`` while loading the
compact Static-MLP + GRU/LSTM checkpoints produced in the training repository.

Typical remote usage
--------------------
Main model:
  python paper_eval/run_static_rnn_lowvis_eval_journal.py \
    --mode main \
    --main_run_id exp_113669104_static_mlp_gru_main

Matrix ablations:
  python paper_eval/run_static_rnn_lowvis_eval_journal.py \
    --mode matrix \
    --matrix_run_prefix exp_static_rnn_matrix_from_113669104

Class definitions are identical to the project guide:
  0: 0 <= visibility < 500 m
  1: 500 <= visibility < 1000 m
  2: visibility >= 1000 m
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent

for _p in (str(VIS_EVAL_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from paper_eval_config import DEFAULT_CONFIG_NAME, apply_paper_eval_config
    from feature_catalog_pm10_pm25 import catalog_rows, permutation_groups, write_catalog
    from metrics_core import pred_from_thresholds_mutual
    from run_paper_eval_pm10_pm25_journal import (
        Manifest,
        TIME_OF_DAY_LOCAL_ORDER,
        REGION_DEFS,
        add_scenario_columns,
        aggregate_station_model_vs_ifs_metrics,
        aggregate_station_metrics,
        build_display_lead_table,
        build_scenario_metrics,
        CLASS_CMAP,
        CLASS_NORM,
        classification_metrics,
        classify_visibility_values,
        draw_basemap,
        draw_boundary,
        infer_init_cycle_hour,
        init_cycle_mask,
        lead_metrics_table,
        load_main_data,
        load_ifs_diagnostic,
        load_ifs_48h_diagnostic,
        plot_csi_recall_pmst_vs_ifs,
        plot_confusion_pmst_vs_ifs,
        plot_diurnal_time_detail,
        plot_event_footprint,
        plot_event_peak_grid,
        plot_fig11_48h_model_vs_ifs,
        plot_fig11_48h_model_vs_ifs_delta_heatmap,
        plot_fig11_lead_init,
        plot_ifs_visibility_bias,
        plot_region_detail,
        plot_scenario_split,
        plot_station_recall_delta_map,
        plot_station_metric_map,
        plot_three_events_footprint_row,
        plot_three_events_peak_row,
        plot_time_of_day_detail,
        read_shapefile,
        run_overlap_subprocess,
        run_key_variable_quality_subprocess,
        run_widespread_event_evaluation,
        save_fig_pair,
        setup_journal_style,
        write_report,
        export_per_sample,
    )
except Exception as exc:  # pragma: no cover - import failure must be explicit remotely.
    raise RuntimeError(f"Cannot import journal evaluation helpers from {VIS_EVAL_DIR}") from exc


DEFAULT_BASE = Path("/public/home/putianshu/vis_mlp")
DEFAULT_TRAIN_DIR = DEFAULT_BASE / "train"
DEFAULT_DATA_DIR = "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_OUT_DIR = "static_rnn_eval_results"
DEFAULT_STAGE_TAG = "S2_PhaseB"
IMPORTANCE_METRICS = [
    "low_vis_recall",
    "low_vis_csi",
    "low_vis_precision",
    "Fog_R",
    "Fog_CSI",
    "Mist_R",
    "Mist_CSI",
    "false_positive_rate",
    "accuracy",
]
LOWER_IS_BETTER = {"false_positive_rate", "Fog_FAR", "Mist_FAR", "Clear_FAR", "ECE", "Brier_Fog", "Brier_Mist"}


@dataclass(frozen=True)
class VariantSpec:
    exp_id: int
    name: str
    encoder: str = "gru"
    hidden_dim: int = 256
    static_hidden_dim: int = 96
    fe_hidden_dim: int = 128
    fusion_hidden_dim: int = 256
    veg_emb_dim: int = 16
    rnn_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: str = "mean"
    use_fe: bool = True
    use_pm: bool = True


VARIANTS: Dict[int, VariantSpec] = {
    0: VariantSpec(0, "static_mlp_gru_main"),
    1: VariantSpec(1, "static_mlp_lstm", encoder="lstm"),
    2: VariantSpec(2, "static_mlp_gru_no_fe", use_fe=False),
    3: VariantSpec(3, "static_mlp_gru_no_pm", use_pm=False),
    4: VariantSpec(4, "static_mlp_gru_aux_reg"),
    5: VariantSpec(5, "static_mlp_gru_attention", pooling="attention"),
    6: VariantSpec(6, "static_mlp_bigru", hidden_dim=192, bidirectional=True),
    7: VariantSpec(7, "static_mlp_gru_csi_select"),
}
DEFAULT_MATRIX_IDS = [1, 2, 3, 4, 5, 6, 7]


@dataclass
class EvalTarget:
    label: str
    run_id: str
    checkpoint: Path
    variant: VariantSpec
    scaler_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Static-MLP + RNN checkpoints and draw journal figures.")
    p.add_argument("--config_json", default=os.environ.get("PAPER_EVAL_CONFIG", str(VIS_EVAL_DIR / DEFAULT_CONFIG_NAME)))
    p.add_argument("--mode", choices=["main", "matrix", "single"], default="main")
    p.add_argument("--base", default=str(DEFAULT_BASE))
    p.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--data_48h_dir", default="ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--stage_tag", default=DEFAULT_STAGE_TAG)

    p.add_argument("--main_run_id", default="")
    p.add_argument("--main_ckpt", default="")
    p.add_argument("--matrix_run_prefix", default="")
    p.add_argument("--matrix_experiments", default="1:2:3:4:5:6:7")
    p.add_argument("--ckpt_path", default="", help="Single-checkpoint path for --mode single.")
    p.add_argument("--run_id", default="", help="Run id for --mode single; defaults to checkpoint stem.")
    p.add_argument("--variant_id", type=int, default=0, help="Variant id for --mode single.")

    p.add_argument("--threshold_source", choices=["checkpoint", "cli", "argmax"], default="checkpoint")
    p.add_argument("--fog_th", type=float, default=0.5)
    p.add_argument("--mist_th", type=float, default=0.5)
    p.add_argument("--ifs_vis_nc", default="VIS_IDW_KDTree_20250101_20251231.nc")
    p.add_argument("--ifs_vis_var", default="VIS")
    p.add_argument("--ifs_48h_nc", default="IFS_VIS_0_48h_stations_2025_00_12.nc")
    p.add_argument("--ifs_48h_var", default="VIS_ifs")
    p.add_argument("--skip_48h", action="store_true")
    p.add_argument("--local_time_offset_hours", type=int, default=8)
    p.add_argument(
        "--meta_time_shift_hours",
        type=float,
        default=0.0,
        help=(
            "Add this many hours to meta_test.csv time before UTC-indexed evaluation. "
            "Use -8 only for legacy BJT-labelled datasets; rebuilding with UTC split is preferred."
        ),
    )
    p.add_argument("--event_top_k", type=int, default=3)
    p.add_argument("--event_window_hours", type=int, default=3)
    p.add_argument("--event_min_fog_stations", type=int, default=80)
    p.add_argument("--event_min_regions", type=int, default=3)
    p.add_argument("--event_min_lon_span", type=float, default=10.0)
    p.add_argument("--event_min_lat_span", type=float, default=4.0)
    p.add_argument("--event_gap_hours", type=int, default=24)
    p.add_argument("--event_env_source", choices=["grid", "none"], default="grid")
    p.add_argument("--event_env_max_events", type=int, default=3)
    p.add_argument("--event_env_rh2m_var", default="rh2m")
    p.add_argument("--event_env_rh2m_vmin", type=float, default=40.0)
    p.add_argument("--event_env_rh2m_vmax", type=float, default=100.0)
    p.add_argument(
        "--event_env_tianji_template",
        default="/tj01/sd3op/userpp/pp_data/{init_yyyymmddhh}/stage26Q/multi_model_sources/{init_yyyymmddhh}/{variable}.nc",
        help="Template for raw Tianji gridded input fields; supports {base}, {variable}, {init}, and {init_yyyymmddhh}.",
    )
    p.add_argument("--event_env_pm10_dir", default="pm10_data")
    p.add_argument("--event_env_pm10_var", default="pm10")
    p.add_argument("--event_env_pm10_vmin", type=float, default=0.0)
    p.add_argument("--event_env_pm10_vmax", type=float, default=240.0)
    p.add_argument("--plots", choices=["none", "core", "all"], default="core")
    p.add_argument("--shp_path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp", help="Optional China boundary shapefile for station/event maps when --plots all.")
    p.add_argument("--allow_missing", action="store_true", help="Skip missing matrix checkpoints instead of failing.")
    p.add_argument("--run_feature_importance", action="store_true", help="Run grouped permutation feature importance for each evaluated target.")
    p.add_argument("--importance_repeats", type=int, default=3)
    p.add_argument("--importance_seed", type=int, default=42)
    p.add_argument("--importance_max_fog", type=int, default=8000)
    p.add_argument("--importance_max_mist", type=int, default=8000)
    p.add_argument("--importance_max_clear", type=int, default=20000)
    p.add_argument("--importance_max_groups", type=int, default=0)
    p.add_argument("--importance_sort_metric", default="low_vis_recall")
    p.add_argument("--run_variable_quality", action="store_true", help="Run Tianji-vs-IFS forecast-variable quality analysis when overlap data and observations exist.")
    p.add_argument("--variable_quality_script", default="")
    p.add_argument("--obs_root", default="/public/home/putianshu/vis_mlp/auto_station")
    p.add_argument("--quality_tianji_data_dir", default="ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_baseline")
    p.add_argument("--quality_ifs_data_dir", default="ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_baseline")
    p.add_argument("--quality_out_dir", default="")
    p.add_argument("--quality_features", default="RH2M,Q_1000,DP_1000,RH_925,PRECIP")
    p.add_argument("--run_overlap_source_comparison", action="store_true", help="Run paired Tianji-vs-IFS overlap source comparison for the main target.")
    p.add_argument("--overlap_script", default="")
    p.add_argument("--overlap_out_dir", default="")
    p.add_argument("--overlap_extra_args", default="--model_arch static_rnn")
    p.add_argument("--overlap_tianji_ckpt", default="")
    p.add_argument("--overlap_ifs_ckpt", default="")
    p.add_argument("--overlap_tianji_scaler", default="")
    p.add_argument("--overlap_ifs_scaler", default="")
    p.add_argument("--overlap_feature_importance_csv", default="")
    p.add_argument("--overlap_feature_swap_top_k", type=int, default=0)
    p.add_argument("--overlap_feature_swap_features", default="RH2M,Q_1000,DP_1000,RH_925,PRECIP")
    p.add_argument("--skip_overlap_bootstrap", action="store_true")
    return p.parse_args()


def as_abs_under(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 1000):
        cand = path.with_name(f"{path.name}_r{idx:02d}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not allocate non-overwriting output directory near {path}")


def parse_id_list(value: str) -> List[int]:
    value = (value or "").replace(",", ":").replace(" ", ":")
    out = []
    for token in value.split(":"):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out or list(DEFAULT_MATRIX_IDS)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def infer_layout_from_x(x_path: Path, window_size: int):
    shape = np.load(x_path, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{x_path} must be 2D [N,D], got {shape}")
    rest = int(shape[1]) - 6
    for dyn in (27, 26, 25, 24):
        fe = rest - dyn * int(window_size)
        if 20 <= fe <= 64:
            return dyn, fe
    raise ValueError(f"Cannot infer static-RNN layout from {x_path}: shape={shape}, window={window_size}")


def load_static_rnn_module(train_dir: Path):
    model_py = train_dir / "train_static_rnn_lowvis.py"
    if not model_py.exists():
        raise FileNotFoundError(f"Missing static-RNN training script: {model_py}")
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    spec = importlib.util.spec_from_file_location("static_rnn_lowvis_for_eval", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {model_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint_payload(path: Path, device):
    import torch

    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        meta = payload.get("metadata", {})
    elif isinstance(payload, dict):
        state = payload
        meta = {}
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(payload)} from {path}")
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    return state, meta if isinstance(meta, dict) else {}


def resolve_scaler_path(base: Path, ckpt_dir: Path, run_id: str, layout, use_pm: bool, explicit: Optional[Path]) -> Path:
    if explicit:
        path = explicit if explicit.is_absolute() else base / explicit
        if not path.exists():
            raise FileNotFoundError(f"Explicit scaler path not found: {path}")
        return path
    tag = "pm" if use_pm else "nopm"
    exact = ckpt_dir / f"robust_scaler_{run_id}_s2_w{layout.window_size}_dyn{layout.dyn_vars}_{tag}.pkl"
    if exact.exists():
        return exact
    candidates = sorted(ckpt_dir.glob(f"robust_scaler_{run_id}_s2_w{layout.window_size}_dyn{layout.dyn_vars}_*.pkl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot find scaler for run_id={run_id} in {ckpt_dir}")


def prepare_static_rows(rows: np.ndarray, scaler, layout, mod, spec: VariantSpec) -> np.ndarray:
    log_mask = mod.build_dyn_log_mask(layout)
    core = rows[:, : layout.core_dim].astype(np.float32, copy=True)
    core = mod.apply_core_transform(core, layout, spec.use_pm, log_mask)
    if scaler is not None:
        if len(scaler.center_) != core.shape[1]:
            raise ValueError(f"Scaler dim={len(scaler.center_)} but core dim={core.shape[1]}")
        core = (core - scaler.center_) / (scaler.scale_ + 1e-6)
    core = np.clip(core, -10.0, 10.0)
    veg = rows[:, layout.split_dyn + 5 : layout.split_dyn + 6].astype(np.float32, copy=False)
    parts = [core, veg]
    if spec.use_fe:
        fe = rows[:, layout.split_dyn + 6 : layout.split_dyn + 6 + layout.fe_dim].astype(np.float32, copy=True)
        parts.append(np.clip(fe, -10.0, 10.0))
    final = np.concatenate(parts, axis=1)
    return np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def run_static_inference(
    x_path: Path,
    scaler,
    model,
    device,
    batch_size: int,
    layout,
    mod,
    spec: VariantSpec,
    limit_samples: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    X = np.load(x_path, mmap_mode="r")
    n = len(X) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(X))
    out = []
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = prepare_static_rows(rows, scaler, layout, mod, spec)
        bx = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits, _ = model(bx)
            probs = F.softmax(logits, dim=1)
        out.append(probs.detach().cpu().numpy())
        if start == 0 or end == n or (start // max(batch_size, 1)) % 20 == 0:
            print(f"  [inference:{spec.name}] {end}/{n}", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def predict_static_rows(
    rows: np.ndarray,
    scaler,
    model,
    device,
    batch_size: int,
    layout,
    mod,
    spec: VariantSpec,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    out = []
    model.eval()
    for start in range(0, len(rows), int(batch_size)):
        end = min(start + int(batch_size), len(rows))
        final = prepare_static_rows(rows[start:end], scaler, layout, mod, spec)
        bx = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits, _ = model(bx)
            probs = F.softmax(logits, dim=1)
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def threshold_from_checkpoint(meta: Dict[str, object]) -> Optional[Tuple[float, float]]:
    th = meta.get("thresholds", {}) if isinstance(meta, dict) else {}
    if isinstance(th, dict) and "fog" in th and "mist" in th:
        return float(th["fog"]), float(th["mist"])
    return None


def predict_from_probs(args: argparse.Namespace, probs: np.ndarray, meta: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
    if args.threshold_source == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64), {"threshold_source": "argmax"}
    ckpt_th = threshold_from_checkpoint(meta)
    if args.threshold_source == "checkpoint" and ckpt_th is not None:
        fog_th, mist_th = ckpt_th
        source = "checkpoint"
    else:
        fog_th, mist_th = float(args.fog_th), float(args.mist_th)
        source = "cli_fallback" if args.threshold_source == "checkpoint" else "cli"
    pred = pred_from_thresholds_mutual(probs, fog_th, mist_th).astype(np.int64)
    return pred, {"threshold_source": source, "threshold_rule": "mutual", "fog_th": fog_th, "mist_th": mist_th}


def make_model(mod, layout, spec: VariantSpec, device):
    return mod.StaticRNNLowVisNet(
        layout=layout,
        encoder=spec.encoder,
        hidden_dim=spec.hidden_dim,
        static_hidden_dim=spec.static_hidden_dim,
        fe_hidden_dim=spec.fe_hidden_dim,
        fusion_hidden_dim=spec.fusion_hidden_dim,
        veg_emb_dim=spec.veg_emb_dim,
        rnn_layers=spec.rnn_layers,
        dropout=spec.dropout,
        bidirectional=spec.bidirectional,
        pooling=spec.pooling,
        use_fe=spec.use_fe,
    ).to(device)


def find_main_checkpoint(ckpt_dir: Path, stage_tag: str, main_run_id: str, main_ckpt: str, base: Path) -> Tuple[str, Path]:
    if main_ckpt:
        path = as_abs_under(base, main_ckpt)
        return (main_run_id or path.name.replace(f"_{stage_tag}_best_score.pt", "")), path
    if main_run_id:
        return main_run_id, ckpt_dir / f"{main_run_id}_{stage_tag}_best_score.pt"
    candidates = sorted(ckpt_dir.glob(f"*static_mlp_gru_main_{stage_tag}_best_score.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No main static-RNN checkpoint found; pass --main_run_id or --main_ckpt.")
    path = candidates[0]
    return path.name.replace(f"_{stage_tag}_best_score.pt", ""), path


def build_targets(args: argparse.Namespace, base: Path, ckpt_dir: Path) -> List[EvalTarget]:
    targets: List[EvalTarget] = []
    if args.mode == "main":
        run_id, path = find_main_checkpoint(ckpt_dir, args.stage_tag, args.main_run_id, args.main_ckpt, base)
        targets.append(EvalTarget("main", run_id, path, VARIANTS[0]))
    elif args.mode == "single":
        if not args.ckpt_path:
            raise ValueError("--mode single requires --ckpt_path")
        path = as_abs_under(base, args.ckpt_path)
        run_id = args.run_id or path.name.replace(f"_{args.stage_tag}_best_score.pt", "")
        targets.append(EvalTarget(VARIANTS[args.variant_id].name, run_id, path, VARIANTS[args.variant_id]))
    else:
        if not args.matrix_run_prefix:
            raise ValueError("--mode matrix requires --matrix_run_prefix")
        for exp_id in parse_id_list(args.matrix_experiments):
            if exp_id == 0:
                raise ValueError("Matrix evaluation excludes experiment 0; evaluate the main model with --mode main.")
            spec = VARIANTS[exp_id]
            run_id = f"{args.matrix_run_prefix}_{exp_id}_{spec.name}"
            path = ckpt_dir / f"{run_id}_{args.stage_tag}_best_score.pt"
            if not path.exists() and args.allow_missing:
                print(f"[skip] missing checkpoint: {path}", flush=True)
                continue
            targets.append(EvalTarget(spec.name, run_id, path, spec))
    for target in targets:
        if not target.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {target.label}: {target.checkpoint}")
    return targets


def plot_summary_bar(summary: pd.DataFrame, out_dir: Path, manifest: Manifest, sources: Sequence[str]) -> None:
    if summary.empty:
        return
    setup_journal_style()
    metrics = [
        ("Fog_CSI", "Fog CSI"),
        ("Fog_R", "Fog recall"),
        ("Mist_CSI", "Mist CSI"),
        ("Mist_R", "Mist recall"),
        ("low_vis_csi", "Low-vis CSI"),
        ("low_vis_recall", "Low-vis recall"),
    ]
    labels = summary["label"].astype(str).tolist()
    x = np.arange(len(labels))
    width = min(0.12, 0.78 / len(metrics))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.82 * len(labels)), 4.6))
    for i, (key, label) in enumerate(metrics):
        vals = summary[key].astype(float).to_numpy()
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width * 0.96, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Static-RNN checkpoint comparison on the S2 test set")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig_pair(fig, out_dir, "fig_static_rnn_model_matrix_summary", manifest, sources, notes="Static-RNN main/ablation checkpoint comparison.")


def predict_for_lead(probs: np.ndarray, decision_meta: Dict[str, object]) -> np.ndarray:
    if decision_meta.get("threshold_source") == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64)
    fog_th = float(decision_meta.get("fog_th", 0.5))
    mist_th = float(decision_meta.get("mist_th", 0.5))
    return pred_from_thresholds_mutual(probs, fog_th, mist_th).astype(np.int64)


def importance_delta(metric: str, baseline: float, perturbed: float) -> float:
    if metric in LOWER_IS_BETTER:
        return float(perturbed - baseline)
    return float(baseline - perturbed)


def sample_importance_indices(y_cls: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(int(args.importance_seed))
    parts = []
    for cls, max_n in (
        (0, int(args.importance_max_fog)),
        (1, int(args.importance_max_mist)),
        (2, int(args.importance_max_clear)),
    ):
        idx = np.flatnonzero(y_cls == cls)
        if max_n > 0 and len(idx) > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
        parts.append(idx)
    out = np.unique(np.concatenate(parts)) if parts else np.arange(len(y_cls))
    out.sort()
    if len(out) == 0:
        raise ValueError("No rows available for feature importance.")
    return out


def score_static_probabilities(y_true: np.ndarray, probs: np.ndarray, decision_meta: Dict[str, object]) -> Dict[str, float]:
    pred = predict_for_lead(probs, decision_meta)
    metrics = classification_metrics(y_true, pred, probs=probs)
    return {k: float(metrics.get(k, np.nan)) for k in IMPORTANCE_METRICS}


def plot_static_feature_importance(imp_df: pd.DataFrame, out_dir: Path, sort_metric: str) -> None:
    col = f"importance_{sort_metric}"
    if imp_df.empty or col not in imp_df:
        return
    setup_journal_style()
    top = imp_df.sort_values(col, ascending=False).head(24).iloc[::-1].copy()
    color_map = {
        "dynamic_12h": "#2E5A87",
        "static": "#6E91B5",
        "static_category": "#6E91B5",
        "feature_engineering": "#E69F00",
        "feature_engineering_vera_optional": "#2A9D8F",
    }
    colors = top["block"].map(color_map).fillna("#7F7F7F")
    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.28 * len(top) + 1.1)))
    ax.barh(top["feature"].astype(str), top[col].astype(float), color=colors)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_xlabel(f"Grouped permutation importance ({sort_metric})")
    ax.set_title("Feature groups that sustain low-visibility skill")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=7, label=label)
        for label, color in (
            ("12 h dynamic", "#2E5A87"),
            ("Station/static", "#6E91B5"),
            ("Engineered", "#E69F00"),
        )
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"fig_static_rnn_feature_importance_{sort_metric}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    plt.close(fig)


def run_static_feature_importance(
    args: argparse.Namespace,
    target: EvalTarget,
    data_dir: Path,
    out_dir: Path,
    manifest: Manifest,
    x_path: Path,
    y_cls: np.ndarray,
    scaler,
    model,
    device,
    layout,
    mod,
    decision_meta: Dict[str, object],
) -> Optional[pd.DataFrame]:
    if not bool(getattr(args, "run_feature_importance", False)):
        return None
    imp_dir = out_dir / "feature_importance"
    imp_dir.mkdir(parents=True, exist_ok=True)
    rows_catalog = catalog_rows(layout.dyn_vars, layout.fe_dim)
    catalog_csv = imp_dir / "feature_catalog_pm10_pm25.csv"
    catalog_md = imp_dir / "feature_catalog_pm10_pm25.md"
    write_catalog(rows_catalog, catalog_csv, catalog_md)

    idx = sample_importance_indices(y_cls, args)
    X = np.load(x_path, mmap_mode="r")
    rows = np.asarray(X[idx], dtype=np.float32)
    y_sample = y_cls[idx]
    print(
        f"[importance:{target.label}] rows={len(idx)} "
        f"fog={int(np.sum(y_sample == 0))} mist={int(np.sum(y_sample == 1))} clear={int(np.sum(y_sample == 2))}",
        flush=True,
    )
    base_probs = predict_static_rows(rows, scaler, model, device, args.batch_size, layout, mod, target.variant)
    baseline = score_static_probabilities(y_sample, base_probs, decision_meta)
    with open(imp_dir / "feature_importance_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_metrics": baseline,
                "sample_size": int(len(idx)),
                "checkpoint": str(target.checkpoint),
                "data_dir": str(data_dir),
                "decision": decision_meta,
                "layout": asdict(layout),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    groups = permutation_groups(layout.window_size, layout.dyn_vars, layout.fe_dim)
    if int(args.importance_max_groups or 0) > 0:
        groups = groups[: int(args.importance_max_groups)]
    rng = np.random.default_rng(int(args.importance_seed))
    records: List[Dict[str, object]] = []
    for gi, group in enumerate(groups, start=1):
        cols = np.asarray(group["columns"], dtype=np.int64)
        repeat_metrics = []
        print(f"[importance:{target.label}] {gi}/{len(groups)} {group['block']}::{group['feature']}", flush=True)
        for _ in range(int(args.importance_repeats)):
            perm = rng.permutation(len(rows))
            perturbed = rows.copy()
            perturbed[:, cols] = rows[perm][:, cols]
            probs_p = predict_static_rows(perturbed, scaler, model, device, args.batch_size, layout, mod, target.variant)
            repeat_metrics.append(score_static_probabilities(y_sample, probs_p, decision_meta))
        row: Dict[str, object] = {
            "feature": group["feature"],
            "block": group["block"],
            "n_columns": int(group["n_columns"]),
            "repeats": int(args.importance_repeats),
        }
        for metric in IMPORTANCE_METRICS:
            vals = np.asarray([m.get(metric, np.nan) for m in repeat_metrics], dtype=float)
            mean_val = float(np.nanmean(vals))
            std_val = float(np.nanstd(vals))
            row[f"baseline_{metric}"] = float(baseline.get(metric, np.nan))
            row[f"permuted_{metric}_mean"] = mean_val
            row[f"permuted_{metric}_std"] = std_val
            row[f"importance_{metric}"] = importance_delta(metric, float(baseline.get(metric, np.nan)), mean_val)
        records.append(row)
    imp_df = pd.DataFrame(records)
    sort_metric = str(args.importance_sort_metric)
    sort_col = f"importance_{sort_metric}"
    if sort_col not in imp_df:
        sort_metric = "low_vis_recall"
        sort_col = f"importance_{sort_metric}"
    if sort_col in imp_df:
        imp_df = imp_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    imp_path = imp_dir / "feature_importance_permutation_static_rnn.csv"
    imp_df.to_csv(imp_path, index=False, float_format="%.8f")
    print(f"[table] {imp_path}", flush=True)
    plot_static_feature_importance(imp_df, imp_dir, sort_metric)
    manifest.add(
        f"fig_static_rnn_feature_importance_{sort_metric}.png/pdf/svg",
        [str(imp_path), str(catalog_csv), str(x_path), str(target.checkpoint)],
        notes="Grouped permutation feature importance on a stratified S2 test subset for the current Static-RNN model.",
        n=int(len(idx)),
    )
    return imp_df


def run_static_48h_optional(
    args: argparse.Namespace,
    target: EvalTarget,
    base: Path,
    data_dir: Path,
    out_dir: Path,
    manifest: Manifest,
    scaler,
    model,
    device,
    layout,
    mod,
    decision_meta: Dict[str, object],
) -> None:
    if args.skip_48h:
        print("[48h] skipped by --skip_48h", flush=True)
        return
    data_48h = as_abs_under(base, args.data_48h_dir)
    if not data_48h.is_dir():
        print(f"[48h] data dir not found: {data_48h}; skip fig11.", flush=True)
        return
    try:
        x_path, y_cls, _, meta = load_main_data(
            data_48h, args.limit_samples, getattr(args, "meta_time_shift_hours", 0.0)
        )
        dyn, fe = infer_layout_from_x(x_path, args.window_size)
        if dyn != layout.dyn_vars or fe != layout.fe_dim:
            print(
                f"[48h] layout dyn/fe=({dyn},{fe}) differs from checkpoint layout "
                f"({layout.dyn_vars},{layout.fe_dim}); skip fig11.",
                flush=True,
            )
            return
        if "lead_hour" not in meta:
            print("[48h] meta_test.csv has no lead_hour; skip fig11.", flush=True)
            return
        init_cycle_hour, init_cycle_source = infer_init_cycle_hour(meta, args.local_time_offset_hours)
        if init_cycle_hour.isna().all():
            print("[48h] meta_test.csv has no parseable init_time/init_hour; skip fig11.", flush=True)
            return
        print(f"[48h] data_dir: {data_48h}", flush=True)
        probs = run_static_inference(
            x_path,
            scaler,
            model,
            device,
            args.batch_size,
            layout,
            mod,
            target.variant,
            args.limit_samples,
        )
        pred = predict_for_lead(probs, decision_meta)
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
        pooled_path = out_dir / "metrics_by_lead_hour_48h_model.csv"
        lead00_path = out_dir / "metrics_by_lead_hour_init00Z.csv"
        lead12_path = out_dir / "metrics_by_lead_hour_init12Z.csv"
        pooled.to_csv(pooled_path, index=False)
        lead00.to_csv(lead00_path, index=False)
        lead12.to_csv(lead12_path, index=False)
        pooled_display = build_display_lead_table(pooled, pooled, "pooled_previous_init_12_24h")
        lead00_display = build_display_lead_table(lead00, lead12, "previous_12Z_init_12_24h")
        lead12_display = build_display_lead_table(lead12, lead00, "previous_00Z_init_12_24h")
        pooled_display_path = out_dir / "metrics_by_display_lead_hour_48h_model.csv"
        lead00_display_path = out_dir / "metrics_by_display_lead_hour_init00Z.csv"
        lead12_display_path = out_dir / "metrics_by_display_lead_hour_init12Z.csv"
        pooled_display.to_csv(pooled_display_path, index=False)
        lead00_display.to_csv(lead00_display_path, index=False)
        lead12_display.to_csv(lead12_display_path, index=False)
        plot_fig11_lead_init(
            pooled_display,
            lead00_display,
            lead12_display,
            out_dir,
            manifest,
            [str(x_path), str(data_48h / "meta_test.csv"), str(target.checkpoint), str(pooled_display_path)],
        )
        ifs_48h_nc = as_abs_under(base, args.ifs_48h_nc)
        if not ifs_48h_nc.exists():
            print(f"[48h IFS] NetCDF not found: {ifs_48h_nc}; skip model-vs-IFS lead figure.", flush=True)
            return
        try:
            ifs_pred, _, ifs_valid, ifs_diag = load_ifs_48h_diagnostic(meta, ifs_48h_nc, args.ifs_48h_var)
            diag_path = out_dir / "lead_eval_alignment_diagnostics_48h_ifs.csv"
            pd.DataFrame([ifs_diag]).to_csv(diag_path, index=False, float_format="%.6f")
            matched_mask = np.asarray(ifs_valid, dtype=bool)
            if int(matched_mask.sum()) < 50:
                print("[48h IFS] fewer than 50 matched rows; skip model-vs-IFS lead figure.", flush=True)
                return
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
                mc = f"{metric}_model"
                ic = f"{metric}_ifs"
                if mc in cmp_df and ic in cmp_df:
                    cmp_df[f"{metric}_diff_model_minus_ifs"] = cmp_df[mc] - cmp_df[ic]
            cmp_df.to_csv(cmp_path, index=False, float_format="%.6f")
            cmp_display = build_display_lead_table(cmp_df, cmp_df, "matched_previous_init_12_24h")
            cmp_display_path = out_dir / "model_vs_ifs_metrics_by_display_lead_hour_48h.csv"
            cmp_display.to_csv(cmp_display_path, index=False, float_format="%.6f")
            plot_fig11_48h_model_vs_ifs(
                cmp_display,
                out_dir,
                manifest,
                [str(x_path), str(data_48h / "meta_test.csv"), str(ifs_48h_nc), str(cmp_display_path)],
            )
            plot_fig11_48h_model_vs_ifs_delta_heatmap(
                cmp_display,
                out_dir,
                manifest,
                [str(x_path), str(data_48h / "meta_test.csv"), str(ifs_48h_nc), str(cmp_display_path)],
            )
        except Exception as exc:
            print(f"[48h IFS] skipped after error: {exc}", flush=True)
    except Exception as exc:
        print(f"[48h] skipped after error: {exc}", flush=True)


@dataclass
class EventGridField:
    values: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    source: str


def _event_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.floor("h")


def _datetime_index_from_values(values) -> pd.DatetimeIndex:
    arr = np.asarray(values).reshape(-1)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(arr, errors="coerce"))
    if np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmedian(np.abs(finite))) > 1.0e8:
            return pd.DatetimeIndex(pd.to_datetime(arr, unit="s", origin="unix", errors="coerce"))
    return pd.DatetimeIndex(pd.to_datetime(arr, errors="coerce"))


def _coord_name(obj, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    low = {str(k).lower(): str(k) for k in list(obj.coords) + list(obj.dims)}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    return None


def _data_var_name(ds, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    preferred_low = preferred.lower()
    for name in ds.data_vars:
        if str(name).lower() == preferred_low:
            return str(name)
    if len(ds.data_vars) == 1:
        return str(next(iter(ds.data_vars)))
    raise KeyError(f"Variable {preferred!r} not found; available={list(ds.data_vars)}")


def _crop_grid_field(da, bounds: Tuple[float, float, float, float] = (72.0, 136.0, 17.0, 54.0)):
    lon_min, lon_max, lat_min, lat_max = bounds
    lat_name = _coord_name(da, ("grid_yt", "latitude", "lat", "y"))
    lon_name = _coord_name(da, ("grid_xt", "longitude", "lon", "x"))
    if lat_name is None or lon_name is None:
        raise KeyError(f"Cannot infer latitude/longitude coords from dims={da.dims}")

    da = da.squeeze(drop=True)
    lat_vals = np.asarray(da[lat_name].values)
    lon_vals = np.asarray(da[lon_name].values)
    if lat_vals.ndim == 1:
        lat_slice = slice(lat_min, lat_max) if lat_vals[0] <= lat_vals[-1] else slice(lat_max, lat_min)
        da = da.sel({lat_name: lat_slice})
    if lon_vals.ndim == 1:
        da = da.sel({lon_name: slice(lon_min, lon_max)})
    da = da.squeeze(drop=True)

    lat_vals = np.asarray(da[lat_name].values)
    lon_vals = np.asarray(da[lon_name].values)
    arr = np.asarray(da.values, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    return arr, lat_vals, lon_vals


def _tianji_candidate_init_times(valid_time: pd.Timestamp) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    for lead in range(12, 25):
        init = valid_time - pd.Timedelta(hours=lead)
        if init.minute == 0 and init.second == 0 and init.hour in (0, 12):
            out.append(init)
    return out


def _render_tianji_grid_path(base: Path, template: str, variable: str, init_time: pd.Timestamp) -> Path:
    text = str(template).format(
        base=str(base),
        variable=variable,
        init=init_time.to_pydatetime(),
        init_yyyymmddhh=init_time.strftime("%Y%m%d%H"),
    )
    path = Path(text)
    return path if path.is_absolute() else base / path


def _nearest_time_position(times: pd.DatetimeIndex, target: pd.Timestamp, tolerance_minutes: float = 30.0) -> Optional[int]:
    if len(times) == 0:
        return None
    if pd.isna(target):
        return None
    t_ns = times.asi8
    finite = t_ns != pd.NaT.value
    if not finite.any():
        return None
    delta = np.abs(t_ns[finite] - int(target.value))
    finite_pos = np.flatnonzero(finite)
    pos = int(finite_pos[int(np.argmin(delta))])
    delta_min = float(delta.min()) / 1.0e9 / 60.0
    return pos if delta_min <= float(tolerance_minutes) else None


def load_tianji_event_grid_fields(
    args: argparse.Namespace,
    base: Path,
    event_times: Sequence[pd.Timestamp],
) -> Tuple[Dict[pd.Timestamp, EventGridField], List[str]]:
    try:
        import xarray as xr
    except ImportError as exc:
        print(f"[events] xarray not available; skip Tianji grid fields: {exc}", flush=True)
        return {}, []

    out: Dict[pd.Timestamp, EventGridField] = {}
    sources: List[str] = []
    ds_cache = {}
    try:
        for t in event_times:
            target = _event_timestamp(t)
            for init in _tianji_candidate_init_times(target):
                path = _render_tianji_grid_path(base, args.event_env_tianji_template, args.event_env_rh2m_var, init)
                if not path.exists():
                    continue
                key = str(path)
                if key not in ds_cache:
                    ds_cache[key] = xr.open_dataset(path)
                ds = ds_cache[key]
                try:
                    var_name = _data_var_name(ds, args.event_env_rh2m_var)
                    if "time" not in ds.coords and "time" not in ds.dims:
                        continue
                    times = _datetime_index_from_values(ds["time"].values)
                    pos = _nearest_time_position(times, target)
                    if pos is None:
                        continue
                    da = ds[var_name].isel({"time": pos})
                    arr, lats, lons = _crop_grid_field(da)
                    out[target] = EventGridField(arr, lats, lons, key)
                    sources.append(key)
                    break
                except Exception as exc:
                    print(f"[events] Tianji grid read failed for {path}: {exc}", flush=True)
            if target not in out:
                print(f"[events] Tianji RH2m grid missing for {target:%Y-%m-%d %H:00} UTC", flush=True)
    finally:
        for ds in ds_cache.values():
            try:
                ds.close()
            except Exception:
                pass
    return out, sorted(set(sources))


def _normalize_pm10_units(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.maximum(arr, 0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanpercentile(finite, 95)) < 0.1:
        arr = arr * 1.0e12
    return arr


def load_pm10_event_grid_fields(
    args: argparse.Namespace,
    base: Path,
    event_times: Sequence[pd.Timestamp],
) -> Tuple[Dict[pd.Timestamp, EventGridField], List[str]]:
    try:
        import xarray as xr
    except ImportError as exc:
        print(f"[events] xarray not available; skip PM10 grid fields: {exc}", flush=True)
        return {}, []

    pm10_dir = Path(args.event_env_pm10_dir)
    if not pm10_dir.is_absolute():
        pm10_dir = base / pm10_dir

    out: Dict[pd.Timestamp, EventGridField] = {}
    sources: List[str] = []
    by_year: Dict[int, List[pd.Timestamp]] = {}
    for t in event_times:
        target = _event_timestamp(t)
        by_year.setdefault(target.year, []).append(target)

    for year, times_needed in sorted(by_year.items()):
        path = pm10_dir / f"{year}.nc"
        if not path.exists():
            for target in times_needed:
                print(f"[events] PM10 grid file missing for {target:%Y-%m-%d %H:00} UTC: {path}", flush=True)
            continue
        try:
            ds = xr.open_dataset(path, decode_cf=False)
        except Exception as exc:
            print(f"[events] PM10 grid open failed for {path}: {exc}", flush=True)
            continue
        try:
            var_name = _data_var_name(ds, args.event_env_pm10_var)
            if "valid_time" not in ds:
                raise KeyError(f"{path} has no valid_time coordinate/variable")
            valid_values = np.asarray(ds["valid_time"].values)
            valid_times = _datetime_index_from_values(valid_values)
            valid_shape = valid_values.shape
            valid_dims = tuple(ds["valid_time"].dims)
            for target in times_needed:
                pos = _nearest_time_position(valid_times, target)
                if pos is None:
                    print(f"[events] PM10 valid time missing for {target:%Y-%m-%d %H:00} UTC", flush=True)
                    continue
                selector = {}
                if valid_dims and len(valid_shape) == len(valid_dims):
                    unraveled = np.unravel_index(pos, valid_shape)
                    selector = {dim: int(ix) for dim, ix in zip(valid_dims, unraveled)}
                selector = {dim: ix for dim, ix in selector.items() if dim in ds[var_name].dims}
                da = ds[var_name].isel(selector) if selector else ds[var_name].isel({ds[var_name].dims[0]: pos})
                arr, lats, lons = _crop_grid_field(da)
                out[target] = EventGridField(_normalize_pm10_units(arr), lats, lons, str(path))
            sources.append(str(path))
        except Exception as exc:
            print(f"[events] PM10 grid read failed for {path}: {exc}", flush=True)
        finally:
            ds.close()
    return out, sorted(set(sources))


def _grid_lon_lat(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
        return lon2d, lat2d
    return lons, lats


def _draw_visibility_class_panel(ax, sub: pd.DataFrame, value_col: str, shp_gdf=None, valid_col: str = "") -> None:
    draw_basemap(ax, shp_gdf, compact=True)
    if sub.empty or value_col not in sub:
        ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center", va="center", color="#6B7280", fontsize=7)
        return
    plot_df = sub
    if valid_col and valid_col in sub:
        valid = sub[valid_col].astype(bool).to_numpy()
        if (~valid).any():
            ax.scatter(sub.loc[~valid, "lon"], sub.loc[~valid, "lat"], s=3.2, color="#D2D6DC", alpha=0.55, linewidths=0, zorder=2)
        plot_df = sub.loc[valid]
    if plot_df.empty:
        ax.text(0.5, 0.5, "No matched IFS", transform=ax.transAxes, ha="center", va="center", color="#6B7280", fontsize=7)
        return
    if value_col.endswith("_m"):
        vals = classify_visibility_values(plot_df[value_col].to_numpy(dtype=float))
    else:
        vals = pd.to_numeric(plot_df[value_col], errors="coerce").to_numpy(dtype=float)
    valid_vals = np.isfinite(vals) & (vals >= 0)
    if valid_vals.any():
        ax.scatter(
            plot_df.loc[valid_vals, "lon"],
            plot_df.loc[valid_vals, "lat"],
            c=vals[valid_vals].astype(int),
            s=5.2,
            cmap=CLASS_CMAP,
            norm=CLASS_NORM,
            linewidths=0,
            alpha=0.93,
            zorder=4,
        )
    draw_boundary(ax, shp_gdf, color="#1F2933", linewidth=0.45, zorder=7)


def _draw_grid_panel(
    ax,
    field: Optional[EventGridField],
    shp_gdf,
    cmap: str,
    norm: Normalize,
    missing_label: str,
) -> None:
    draw_basemap(ax, shp_gdf, compact=True)
    if field is None:
        ax.text(0.5, 0.5, missing_label, transform=ax.transAxes, ha="center", va="center", color="#6B7280", fontsize=7)
        return
    lon2d, lat2d = _grid_lon_lat(field.lats, field.lons)
    ax.pcolormesh(lon2d, lat2d, field.values, cmap=cmap, norm=norm, shading="auto", zorder=2)
    draw_boundary(ax, shp_gdf, color="#1F2933", linewidth=0.50, zorder=7)


def plot_event_environment_grid(
    args: argparse.Namespace,
    base: Path,
    eval_df: pd.DataFrame,
    event_row: pd.Series,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    shp_gdf=None,
) -> None:
    if str(getattr(args, "event_env_source", "grid")).lower() == "none":
        return

    setup_journal_style()
    center_time = _event_timestamp(event_row["peak_time"])
    offsets = list(range(-int(args.event_window_hours), int(args.event_window_hours) + 1))
    event_times = [center_time + pd.Timedelta(hours=h) for h in offsets]

    rh_fields, rh_sources = load_tianji_event_grid_fields(args, base, event_times)
    pm10_fields, pm10_sources = load_pm10_event_grid_fields(args, base, event_times)

    df = eval_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.floor("h")

    nrows = len(event_times)
    fig_h = max(7.2, 1.18 * nrows + 1.25)
    fig, axes = plt.subplots(nrows, 5, figsize=(12.4, fig_h), squeeze=False)
    col_titles = ["Observed", "Model", "IFS diagnostic", "Tianji RH2m", "PM10"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8.5, fontweight="bold", pad=4)

    rh_norm = Normalize(vmin=float(args.event_env_rh2m_vmin), vmax=float(args.event_env_rh2m_vmax))
    pm10_norm = Normalize(vmin=float(args.event_env_pm10_vmin), vmax=float(args.event_env_pm10_vmax))
    for row_idx, (offset, valid_time) in enumerate(zip(offsets, event_times)):
        sub = df[df["time"] == valid_time]
        axes[row_idx, 0].set_ylabel(
            f"{valid_time:%m-%d}\n{valid_time:%H:00}",
            rotation=0,
            labelpad=18,
            fontsize=7.2,
            va="center",
            ha="right",
        )
        _draw_visibility_class_panel(axes[row_idx, 0], sub, "vis_raw_m", shp_gdf)
        _draw_visibility_class_panel(axes[row_idx, 1], sub, "pmst_pred", shp_gdf)
        _draw_visibility_class_panel(axes[row_idx, 2], sub, "ifs_diagnostic_vis_m", shp_gdf, valid_col="ifs_diagnostic_valid")
        _draw_grid_panel(axes[row_idx, 3], rh_fields.get(valid_time), shp_gdf, "YlGnBu", rh_norm, "RH2m missing")
        _draw_grid_panel(axes[row_idx, 4], pm10_fields.get(valid_time), shp_gdf, "YlOrRd", pm10_norm, "PM10 missing")
        axes[row_idx, 0].text(
            -0.31,
            0.5,
            f"{offset:+d} h",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.0,
            color="#4B5563",
        )

    class_sm = plt.cm.ScalarMappable(norm=CLASS_NORM, cmap=CLASS_CMAP)
    class_sm.set_array([])
    rh_sm = plt.cm.ScalarMappable(norm=rh_norm, cmap="YlGnBu")
    rh_sm.set_array([])
    pm10_sm = plt.cm.ScalarMappable(norm=pm10_norm, cmap="YlOrRd")
    pm10_sm.set_array([])
    cb1 = fig.colorbar(class_sm, ax=axes[:, :3].ravel().tolist(), orientation="horizontal", fraction=0.035, pad=0.025)
    cb1.set_ticks([0, 1, 2])
    cb1.set_ticklabels(["Fog\n<500", "Mist\n500-1000", "Clear\n>=1000"])
    cb1.set_label("Visibility category (m)", fontsize=7.2)
    cb2 = fig.colorbar(rh_sm, ax=axes[:, 3].ravel().tolist(), orientation="horizontal", fraction=0.035, pad=0.025)
    cb2.set_label("RH2m (%)", fontsize=7.2)
    cb3 = fig.colorbar(pm10_sm, ax=axes[:, 4].ravel().tolist(), orientation="horizontal", fraction=0.035, pad=0.025)
    cb3.set_label(r"PM10 ($\mu$g m$^{-3}$)", fontsize=7.2)

    rank = int(event_row.get("event_rank", 1))
    actual_peak = event_row.get("actual_peak_time", "")
    title = f"Event {rank}: {center_time:%Y-%m-%d %H:00 UTC}"
    if "actual_peak_time" in event_row and pd.notna(actual_peak):
        actual_ts = _event_timestamp(actual_peak)
        if actual_ts != center_time:
            title += f" window center; true peak {actual_ts:%Y-%m-%d %H:00 UTC}"
    fig.suptitle(title, x=0.5, y=0.988, fontsize=9.8, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.992, top=0.94, bottom=0.10, wspace=0.035, hspace=0.035)

    all_sources = list(sources) + rh_sources + pm10_sources
    save_fig_pair(
        fig,
        out_dir,
        f"fig9_event_{rank}_environment_grid",
        manifest,
        all_sources,
        notes=(
            "Rows are UTC hours around the selected widespread low-visibility event. "
            "The first three columns use shared visibility categories; the model panel is categorical. "
            "RH2m and PM10 are raw gridded forecast fields read only for the displayed valid times."
        ),
        n=int(len(eval_df)),
    )


def plot_event_environment_grids(
    args: argparse.Namespace,
    base: Path,
    eval_df: pd.DataFrame,
    event_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    shp_gdf=None,
) -> None:
    if event_df is None or event_df.empty:
        return
    max_events = max(0, int(getattr(args, "event_env_max_events", 3) or 0))
    if max_events <= 0:
        return
    for _, event_row in event_df.head(max_events).iterrows():
        try:
            plot_event_environment_grid(args, base, eval_df, event_row, out_dir, manifest, sources, shp_gdf=shp_gdf)
        except Exception as exc:
            rank = event_row.get("event_rank", "?")
            print(f"[events] environment grid failed for event {rank}: {exc}", flush=True)


def run_event_plots(
    args: argparse.Namespace,
    base: Path,
    out_dir: Path,
    manifest: Manifest,
    meta: pd.DataFrame,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    pred: np.ndarray,
    eval_df: pd.DataFrame,
    ifs_nc: Path,
    ifs_valid: Optional[np.ndarray],
    shp_gdf,
) -> None:
    event_df = pd.DataFrame()
    event_eval_completed = False
    if run_widespread_event_evaluation is not None:
        try:
            event_df = run_widespread_event_evaluation(
                meta=meta,
                y_true=y_cls,
                y_true_raw=y_raw,
                pmst_pred=pred,
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
    if event_df is not None and not event_df.empty:
        event_sources = [str(event_path), str(out_dir / "per_sample_eval.csv")]
        if plot_three_events_footprint_row is not None:
            three_footprint_path = out_dir / "fig_three_events_footprint_row.png"
            if plot_three_events_footprint_row(
                meta,
                y_raw,
                pred,
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
                pred,
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
        plot_event_environment_grids(
            args,
            base,
            eval_df,
            event_df,
            out_dir,
            manifest,
            event_sources,
            shp_gdf=shp_gdf,
        )
    plot_event_peak_grid(
        eval_df,
        event_df,
        out_dir,
        manifest,
        [str(event_path), str(out_dir / "per_sample_eval.csv")],
        shp_gdf=shp_gdf,
    )
    hourly_paths = [out_dir / f"fig9_event_{k}_hourly_metrics.csv" for k in (1, 2, 3)]
    plot_event_footprint(hourly_paths, out_dir, manifest, [str(p) for p in hourly_paths])


def evaluate_target(
    args: argparse.Namespace,
    target: EvalTarget,
    base: Path,
    train_dir: Path,
    data_dir: Path,
    out_root: Path,
    mod,
    device,
) -> Dict[str, object]:
    import joblib

    print(f"\n=== Evaluating {target.label}: {target.run_id} ===", flush=True)
    x_path, y_cls, y_raw, meta = load_main_data(
        data_dir, args.limit_samples, getattr(args, "meta_time_shift_hours", 0.0)
    )
    dyn, fe = infer_layout_from_x(x_path, args.window_size)
    layout = mod.Layout(window_size=args.window_size, dyn_vars=dyn, fe_dim=fe)
    ckpt_dir = as_abs_under(base, args.ckpt_dir)
    scaler_path = resolve_scaler_path(base, ckpt_dir, target.run_id, layout, target.variant.use_pm, target.scaler_path)
    scaler = joblib.load(scaler_path)

    state, ckpt_meta = load_checkpoint_payload(target.checkpoint, device)
    model = make_model(mod, layout, target.variant, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing keys: {len(missing)} first={missing[:5]}", flush=True)
    if unexpected:
        print(f"  [ckpt] unexpected keys: {len(unexpected)} first={unexpected[:5]}", flush=True)

    probs = run_static_inference(
        x_path,
        scaler,
        model,
        device,
        args.batch_size,
        layout,
        mod,
        target.variant,
        args.limit_samples,
    )
    pred, decision_meta = predict_from_probs(args, probs, ckpt_meta)
    metrics = classification_metrics(y_cls, pred, probs=probs)

    out_dir = unique_dir(out_root / target.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir)
    sources = [str(x_path), str(data_dir / "y_test.npy"), str(data_dir / "meta_test.csv"), str(target.checkpoint), str(scaler_path)]

    ifs_pred = ifs_vis = ifs_valid = None
    ifs_metrics = None
    ifs_nc = as_abs_under(base, args.ifs_vis_nc)
    overall_rows = [{"source": "pmst", "model_label": target.label, "sample_scope": "test", **decision_meta, **metrics}]
    if ifs_nc.exists():
        try:
            ifs_pred, ifs_vis, ifs_valid = load_ifs_diagnostic(meta, ifs_nc, args.ifs_vis_var)
            if int(np.sum(ifs_valid)) > 0:
                ifs_metrics = classification_metrics(y_cls[ifs_valid], ifs_pred[ifs_valid])
                matched_metrics = classification_metrics(y_cls[ifs_valid], pred[ifs_valid], probs=probs[ifs_valid])
                pd.DataFrame(
                    [
                        {"source": "pmst", "model_label": target.label, "sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid)), **decision_meta, **matched_metrics},
                        {"source": "ifs_diagnostic", "sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid)), "ifs_forecast_nc": str(ifs_nc), **ifs_metrics},
                    ]
                ).to_csv(out_dir / "ifs_diagnostic_matched_metrics.csv", index=False)
                overall_rows.extend(
                    [
                        {"source": "pmst", "model_label": target.label, "sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid)), **decision_meta, **matched_metrics},
                        {"source": "ifs_diagnostic", "sample_scope": "ifs_diagnostic_matched_test", "matched_rows": int(np.sum(ifs_valid)), "ifs_forecast_nc": str(ifs_nc), **ifs_metrics},
                    ]
                )
                sources.append(str(ifs_nc))
        except Exception as exc:
            print(f"[IFS] diagnostic baseline skipped: {exc}", flush=True)
            ifs_valid = np.zeros(len(y_cls), dtype=bool)
    elif args.plots == "all":
        print(f"[IFS] diagnostic NetCDF not found: {ifs_nc}; skip IFS comparison plots.", flush=True)
    pd.DataFrame(overall_rows).to_csv(out_dir / "overall_metrics.csv", index=False)
    np.save(out_dir / "probs.npy", probs.astype(np.float32))
    eval_df = export_per_sample(
        out_dir / "per_sample_eval.csv",
        meta,
        y_cls,
        y_raw,
        pred,
        probs,
        ifs_pred=ifs_pred,
        ifs_vis=ifs_vis,
        ifs_valid=ifs_valid,
    )
    write_report(out_dir / "rare_event_report.txt", y_cls, pred, metrics, ifs_metrics)

    scenario_df = build_scenario_metrics(eval_df)
    scenario_df.to_csv(out_dir / "scenario_metrics.csv", index=False)
    station_df = aggregate_station_metrics(eval_df, "pmst_pred")
    station_df.to_csv(out_dir / "station_metrics.csv", index=False)
    station_delta_df = aggregate_station_model_vs_ifs_metrics(eval_df)
    if station_delta_df is not None and not station_delta_df.empty:
        station_delta_df.to_csv(out_dir / "station_model_vs_ifs_metrics.csv", index=False, float_format="%.6f")
        print(f"[table] {out_dir / 'station_model_vs_ifs_metrics.csv'}", flush=True)

    if bool(getattr(args, "run_feature_importance", False)) and args.plots == "all":
        run_static_feature_importance(
            args,
            target,
            data_dir,
            out_dir,
            manifest,
            x_path,
            y_cls,
            scaler,
            model,
            device,
            layout,
            mod,
            decision_meta,
        )

    if args.plots != "none":
        matched_for_plot = None
        metrics_for_plot = metrics
        if ifs_metrics is not None and ifs_valid is not None and int(np.sum(ifs_valid)) > 0:
            matched_for_plot = int(np.sum(ifs_valid))
            metrics_for_plot = classification_metrics(y_cls[ifs_valid], pred[ifs_valid], probs=probs[ifs_valid])
        plot_confusion_pmst_vs_ifs(y_cls, pred, ifs_pred, ifs_valid, out_dir, manifest, sources)
        plot_csi_recall_pmst_vs_ifs(metrics_for_plot, ifs_metrics, out_dir, manifest, sources, n=len(y_cls), matched_ifs=matched_for_plot)
        if args.plots == "all" and ifs_vis is not None and ifs_valid is not None:
            plot_ifs_visibility_bias(y_cls, y_raw, ifs_vis, ifs_valid, out_dir, manifest, sources)
        for split, order in (
            ("time_of_day", TIME_OF_DAY_LOCAL_ORDER),
            ("season", ["DJF", "MAM", "JJA", "SON"]),
            ("region", [r[0] for r in REGION_DEFS] + ["Other"]),
        ):
            plot_scenario_split(scenario_df, split, order, out_dir, manifest, [str(out_dir / "scenario_metrics.csv")])
        plot_time_of_day_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")], offset_hours=args.local_time_offset_hours)
        plot_region_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")])
        run_static_48h_optional(
            args,
            target,
            base,
            data_dir,
            out_dir,
            manifest,
            scaler,
            model,
            device,
            layout,
            mod,
            decision_meta,
        )
        if args.plots == "all":
            plot_diurnal_time_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")], offset_hours=args.local_time_offset_hours)
            shp = read_shapefile(args.shp_path) if args.shp_path else None
            if station_delta_df is not None and not station_delta_df.empty:
                plot_station_recall_delta_map(
                    station_delta_df,
                    out_dir,
                    manifest,
                    [str(out_dir / "station_model_vs_ifs_metrics.csv")],
                    shp_gdf=shp,
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
                shp_gdf=shp,
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
                shp_gdf=shp,
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
                shp_gdf=shp,
                cmap="cividis",
                vmin=0,
                vmax=1,
            )
            run_event_plots(
                args,
                base,
                out_dir,
                manifest,
                meta,
                y_cls,
                y_raw,
                pred,
                eval_df,
                ifs_nc,
                ifs_valid,
                shp,
            )
            if target.label == "main":
                run_key_variable_quality_subprocess(args, base, out_dir, manifest)
                if bool(getattr(args, "run_overlap_source_comparison", False)):
                    run_overlap_subprocess(args, base, out_dir, manifest)
    manifest.write()

    run_config = {
        "target": target.label,
        "run_id": target.run_id,
        "checkpoint": str(target.checkpoint),
        "scaler": str(scaler_path),
        "variant": asdict(target.variant),
        "layout": asdict(layout),
        "decision": decision_meta,
        "meta_time_shift_hours": float(getattr(args, "meta_time_shift_hours", 0.0) or 0.0),
        "run_feature_importance": bool(getattr(args, "run_feature_importance", False)),
        "run_variable_quality": bool(getattr(args, "run_variable_quality", False)),
        "event_environment": {
            "source": str(getattr(args, "event_env_source", "grid")),
            "max_events": int(getattr(args, "event_env_max_events", 3) or 0),
            "tianji_template": str(getattr(args, "event_env_tianji_template", "")),
            "rh2m_var": str(getattr(args, "event_env_rh2m_var", "rh2m")),
            "pm10_dir": str(getattr(args, "event_env_pm10_dir", "pm10_data")),
            "pm10_var": str(getattr(args, "event_env_pm10_var", "pm10")),
        },
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "checkpoint_metadata": ckpt_meta,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    row = {"label": target.label, "run_id": target.run_id, "checkpoint": str(target.checkpoint), "out_dir": str(out_dir), **decision_meta, **metrics}
    return row


def main() -> None:
    args = parse_args()
    args = apply_paper_eval_config(args, "static_rnn_eval", default_dir=VIS_EVAL_DIR)

    base = Path(args.base).expanduser()
    train_dir = Path(args.train_dir).expanduser()
    data_dir = as_abs_under(base, args.data_dir)
    ckpt_dir = as_abs_under(base, args.ckpt_dir)
    out_root = as_abs_under(base, args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = build_targets(args, base, ckpt_dir)
    mod = load_static_rnn_module(train_dir)
    device = resolve_device(args.device)

    print("Static-RNN journal evaluation", flush=True)
    print(f"mode      : {args.mode}", flush=True)
    print(f"base      : {base}", flush=True)
    print(f"train_dir : {train_dir}", flush=True)
    print(f"data_dir  : {data_dir}", flush=True)
    print(f"out_root  : {out_root}", flush=True)
    print(f"device    : {device}", flush=True)
    print(f"targets   : {[t.run_id for t in targets]}", flush=True)

    rows = []
    for target in targets:
        rows.append(evaluate_target(args, target, base, train_dir, data_dir, out_root, mod, device))
    summary = pd.DataFrame(rows)
    summary_path = out_root / "static_rnn_eval_summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}", flush=True)
    if args.plots != "none" and len(summary) > 1:
        manifest = Manifest(out_root)
        plot_summary_bar(summary, out_root, manifest, [str(summary_path)])
        manifest.write()


if __name__ == "__main__":
    main()
