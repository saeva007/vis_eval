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


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent

for _p in (str(VIS_EVAL_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from paper_eval_config import DEFAULT_CONFIG_NAME, apply_paper_eval_config
    from metrics_core import pred_from_thresholds_mutual
    from run_paper_eval_pm10_pm25_journal import (
        Manifest,
        TIME_OF_DAY_LOCAL_ORDER,
        REGION_DEFS,
        add_scenario_columns,
        aggregate_station_metrics,
        build_scenario_metrics,
        classification_metrics,
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
        plot_fig11_lead_init,
        plot_ifs_visibility_bias,
        plot_region_detail,
        plot_scenario_split,
        plot_station_metric_map,
        plot_three_events_footprint_row,
        plot_three_events_peak_row,
        plot_time_of_day_detail,
        read_shapefile,
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
    p.add_argument("--event_top_k", type=int, default=3)
    p.add_argument("--event_window_hours", type=int, default=3)
    p.add_argument("--event_min_fog_stations", type=int, default=80)
    p.add_argument("--event_min_regions", type=int, default=3)
    p.add_argument("--event_min_lon_span", type=float, default=10.0)
    p.add_argument("--event_min_lat_span", type=float, default=4.0)
    p.add_argument("--event_gap_hours", type=int, default=24)
    p.add_argument("--plots", choices=["none", "core", "all"], default="core")
    p.add_argument("--shp_path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp", help="Optional China boundary shapefile for station/event maps when --plots all.")
    p.add_argument("--allow_missing", action="store_true", help="Skip missing matrix checkpoints instead of failing.")
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
        x_path, y_cls, _, meta = load_main_data(data_48h, args.limit_samples)
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
        plot_fig11_lead_init(
            pooled,
            lead00,
            lead12,
            out_dir,
            manifest,
            [str(x_path), str(data_48h / "meta_test.csv"), str(target.checkpoint)],
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
            plot_fig11_48h_model_vs_ifs(
                cmp_df,
                out_dir,
                manifest,
                [str(x_path), str(data_48h / "meta_test.csv"), str(ifs_48h_nc), str(cmp_path)],
            )
        except Exception as exc:
            print(f"[48h IFS] skipped after error: {exc}", flush=True)
    except Exception as exc:
        print(f"[48h] skipped after error: {exc}", flush=True)


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
    x_path, y_cls, y_raw, meta = load_main_data(data_dir, args.limit_samples)
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
    manifest.write()

    run_config = {
        "target": target.label,
        "run_id": target.run_id,
        "checkpoint": str(target.checkpoint),
        "scaler": str(scaler_path),
        "variant": asdict(target.variant),
        "layout": asdict(layout),
        "decision": decision_meta,
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
