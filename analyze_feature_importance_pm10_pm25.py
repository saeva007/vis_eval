#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permutation feature importance for the PM10+PM2.5 S2 PMST model.

The script evaluates a trained checkpoint on a stratified subset of the S2 test
set, then shuffles one input group at a time. Dynamic variables are shuffled as
whole 12 h sequences, while static and feature-engineering columns are shuffled
as single columns. Positive importance means the metric gets worse after the
feature is broken.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
LOCAL_ROOT = VIS_EVAL_DIR.parent

for _p in (str(LOCAL_ROOT), str(VIS_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_catalog_pm10_pm25 import catalog_rows, permutation_groups, write_catalog


DEFAULT_CKPT = "checkpoints/exp_1778563813_pm10_more_temp_search_utc_S2_PhaseB_best_score.pt"
DEFAULT_SCALER = "checkpoints/robust_scaler_exp_1778563813_pm10_more_temp_search_utc_w12_dyn27_s2_48h_pm10.pkl"
DEFAULT_SEASON_TH = "checkpoints/exp_1778563813_pm10_more_temp_search_utc_season_thresholds.pt"
LOWER_IS_BETTER = {"false_positive_rate", "Fog_FAR", "Mist_FAR", "Clear_FAR", "ECE", "Brier_Fog", "Brier_Mist"}
METRIC_KEYS = [
    "low_vis_f2",
    "low_vis_csi",
    "low_vis_recall",
    "low_vis_precision",
    "Fog_CSI",
    "Fog_R",
    "Mist_CSI",
    "Mist_R",
    "false_positive_rate",
    "accuracy",
]


def abs_under_base(base: Path, path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return base / p


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


def eval_helpers():
    import run_paper_eval_pm10_pm25_journal as ev

    return ev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run grouped permutation feature importance for the PMST S2 model.")
    p.add_argument("--base", default=os.environ.get("VIS_MLP_ROOT", "/public/home/putianshu/vis_mlp"))
    p.add_argument("--data_dir", default="ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2")
    p.add_argument("--out_dir", default="paper_eval_results_pm10_pm25_journal_utc/feature_importance")
    p.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    p.add_argument("--scaler_path", default=DEFAULT_SCALER)
    p.add_argument("--season_th_path", default=DEFAULT_SEASON_TH)
    p.add_argument("--model_py", default="")
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--dyn_vars_count", type=int, default=0)
    p.add_argument("--extra_feat_dim", type=int, default=0)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--fog_th", type=float, default=0.10)
    p.add_argument("--mist_th", type=float, default=0.42)
    p.add_argument("--threshold_rule", choices=["default", "mutual", "joint"], default="mutual")
    p.add_argument("--use_calibration", action="store_true", help="Load temperature from --season_th_path if present.")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max_fog", type=int, default=8000)
    p.add_argument("--max_mist", type=int, default=8000)
    p.add_argument("--max_clear", type=int, default=20000)
    p.add_argument("--limit_samples", type=int, default=0, help="Only consider the first N test rows before sampling.")
    p.add_argument("--max_groups", type=int, default=0, help="Debug option: evaluate only first N feature groups.")
    p.add_argument("--sort_metric", default="low_vis_f2", choices=METRIC_KEYS)
    p.add_argument("--catalog_only", action="store_true", help="Write feature catalog and exit without loading model.")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def load_test_labels(data_dir: Path, limit_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    y_path = data_dir / "y_test.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing required input: {y_path}")
    y_raw_all = np.load(y_path)
    n = len(y_raw_all) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(y_raw_all))
    return eval_helpers().visibility_to_class(y_raw_all[:n])


def sample_indices(y_cls: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(int(args.seed))
    parts = []
    for cls, max_n in ((0, args.max_fog), (1, args.max_mist), (2, args.max_clear)):
        idx = np.flatnonzero(y_cls == cls)
        if max_n and max_n > 0 and len(idx) > max_n:
            idx = rng.choice(idx, size=int(max_n), replace=False)
        parts.append(idx)
    idx = np.concatenate(parts) if parts else np.arange(len(y_cls))
    idx = np.unique(idx)
    idx.sort()
    if len(idx) == 0:
        raise ValueError("No sampled rows available for feature importance.")
    return idx


def load_sample_rows(x_path: Path, idx: np.ndarray) -> np.ndarray:
    X = np.load(x_path, mmap_mode="r")
    return np.asarray(X[idx], dtype=np.float32)


def predict_rows(
    rows: np.ndarray,
    scaler,
    model,
    device,
    batch_size: int,
    window_size: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    temperature: Optional[float] = None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    temp = 1.0 if temperature is None else max(float(temperature), 1e-6)
    out = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        end = min(start + int(batch_size), len(rows))
        final = eval_helpers().prepare_batch_rows(rows[start:end], scaler, window_size, dyn_vars_count, extra_feat_dim)
        x = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits = model(x)[0]
            probs = F.softmax(logits / temp, dim=1)
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def load_model_and_scaler(args: argparse.Namespace, base: Path, dyn: int, fe: int):
    import joblib
    import torch

    ckpt_path = abs_under_base(base, args.ckpt_path)
    scaler_path = abs_under_base(base, args.scaler_path)
    season_th_path = abs_under_base(base, args.season_th_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    ev = eval_helpers()
    device = ev.resolve_device(args.device)
    scaler = joblib.load(scaler_path)
    model_cls = ev.import_model_class(ev.resolve_model_py(base, args.model_py))
    model = model_cls(
        dyn_vars_count=dyn,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        extra_feat_dim=fe,
    ).to(device)
    ev.load_checkpoint_into_model(model, ckpt_path, device)
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
    return model, scaler, device, temperature


def score_metrics(y_true: np.ndarray, probs: np.ndarray, args: argparse.Namespace) -> Dict[str, float]:
    ev = eval_helpers()
    pred = ev.pred_from_probs_rule(probs, args.fog_th, args.mist_th, args.threshold_rule)
    metrics = ev.classification_metrics(y_true, pred, probs=probs)
    return {k: float(metrics.get(k, np.nan)) for k in METRIC_KEYS}


def importance_delta(metric: str, baseline: float, permuted: float) -> float:
    if metric in LOWER_IS_BETTER:
        return float(permuted - baseline)
    return float(baseline - permuted)


def run_permutation_importance(
    rows: np.ndarray,
    y_true: np.ndarray,
    groups: Sequence[Dict[str, object]],
    baseline_metrics: Dict[str, float],
    model,
    scaler,
    device,
    temperature: Optional[float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(args.seed))
    n = len(rows)
    records = []
    groups_eval = list(groups)
    if args.max_groups and args.max_groups > 0:
        groups_eval = groups_eval[: int(args.max_groups)]

    for group_i, group in enumerate(groups_eval, start=1):
        cols = np.asarray(group["columns"], dtype=np.int64)
        repeat_metrics: List[Dict[str, float]] = []
        print(
            f"[importance] {group_i}/{len(groups_eval)} {group['block']}::{group['feature']} ({len(cols)} cols)",
            flush=True,
        )
        for r in range(int(args.repeats)):
            perm = rng.permutation(n)
            x_perm = rows.copy()
            x_perm[:, cols] = rows[perm][:, cols]
            probs = predict_rows(
                x_perm,
                scaler,
                model,
                device,
                args.batch_size,
                args.window_size,
                args.dyn_vars_count,
                args.extra_feat_dim,
                temperature=temperature,
            )
            repeat_metrics.append(score_metrics(y_true, probs, args))
            del x_perm, probs
        row: Dict[str, object] = {
            "feature": group["feature"],
            "block": group["block"],
            "n_columns": int(group["n_columns"]),
            "repeats": int(args.repeats),
        }
        for key in METRIC_KEYS:
            values = np.asarray([m[key] for m in repeat_metrics], dtype=float)
            mean_val = float(np.nanmean(values))
            std_val = float(np.nanstd(values))
            row[f"baseline_{key}"] = float(baseline_metrics[key])
            row[f"permuted_{key}_mean"] = mean_val
            row[f"permuted_{key}_std"] = std_val
            row[f"importance_{key}"] = importance_delta(key, baseline_metrics[key], mean_val)
        records.append(row)
    out = pd.DataFrame(records)
    sort_col = f"importance_{args.sort_metric}"
    if sort_col in out:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


def plot_importance(df: pd.DataFrame, out_dir: Path, sort_metric: str) -> None:
    col = f"importance_{sort_metric}"
    if col not in df or df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    top = df.sort_values(col, ascending=False).head(25).iloc[::-1]
    colors = top["block"].map(
        {
            "dynamic_12h": "#2E5A87",
            "static": "#6E91B5",
            "static_category": "#6E91B5",
            "feature_engineering": "#E69F00",
            "feature_engineering_vera_optional": "#2A9D8F",
        }
    ).fillna("#7F7F7F")
    fig, ax = plt.subplots(figsize=(8.8, max(5.0, 0.28 * len(top) + 1.2)))
    ax.barh(top["feature"].astype(str), top[col].astype(float), color=colors)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel(f"Permutation importance ({sort_metric}; positive = worse after shuffling)")
    ax.set_ylabel("")
    ax.set_title("Grouped permutation feature importance")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"fig_feature_importance_{sort_metric}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.base).resolve()
    out_dir = abs_under_base(base, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = abs_under_base(base, args.data_dir)
    x_path = data_dir / "X_test.npy"
    if x_path.exists():
        dyn_inferred, fe_inferred = infer_layout_from_x(x_path, args.window_size)
        args.dyn_vars_count = int(args.dyn_vars_count or dyn_inferred)
        args.extra_feat_dim = int(args.extra_feat_dim or fe_inferred)
    elif args.catalog_only and args.dyn_vars_count and args.extra_feat_dim:
        args.dyn_vars_count = int(args.dyn_vars_count)
        args.extra_feat_dim = int(args.extra_feat_dim)
    else:
        raise FileNotFoundError(f"Missing required input: {x_path}")
    print(f"[layout] dyn_vars_count={args.dyn_vars_count}, extra_feat_dim={args.extra_feat_dim}", flush=True)

    rows_catalog = catalog_rows(args.dyn_vars_count, args.extra_feat_dim)
    catalog_csv = out_dir / "feature_catalog_pm10_pm25.csv"
    catalog_md = out_dir / "feature_catalog_pm10_pm25.md"
    write_catalog(rows_catalog, catalog_csv, catalog_md)
    print(f"[catalog] {catalog_csv}", flush=True)
    print(f"[catalog] {catalog_md}", flush=True)
    if args.catalog_only:
        return

    y_cls, y_raw = load_test_labels(data_dir, args.limit_samples)
    idx = sample_indices(y_cls, args)
    y_sample = y_cls[idx]
    rows = load_sample_rows(x_path, idx)
    class_counts = {str(k): int(np.sum(y_sample == k)) for k in (0, 1, 2)}
    print(f"[sample] rows={len(idx)} class_counts={class_counts}", flush=True)

    model, scaler, device, temperature = load_model_and_scaler(args, base, args.dyn_vars_count, args.extra_feat_dim)
    base_probs = predict_rows(
        rows,
        scaler,
        model,
        device,
        args.batch_size,
        args.window_size,
        args.dyn_vars_count,
        args.extra_feat_dim,
        temperature=temperature,
    )
    baseline_metrics = score_metrics(y_sample, base_probs, args)
    with open(out_dir / "feature_importance_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_metrics": baseline_metrics,
                "sample_size": int(len(idx)),
                "class_counts": class_counts,
                "data_dir": str(data_dir),
                "x_path": str(x_path),
                "ckpt_path": str(abs_under_base(base, args.ckpt_path)),
                "scaler_path": str(abs_under_base(base, args.scaler_path)),
                "threshold_rule": args.threshold_rule,
                "fog_th": float(args.fog_th),
                "mist_th": float(args.mist_th),
                "dyn_vars_count": int(args.dyn_vars_count),
                "extra_feat_dim": int(args.extra_feat_dim),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[baseline] {baseline_metrics}", flush=True)

    groups = permutation_groups(args.window_size, args.dyn_vars_count, args.extra_feat_dim)
    imp_df = run_permutation_importance(
        rows,
        y_sample,
        groups,
        baseline_metrics,
        model,
        scaler,
        device,
        temperature,
        args,
    )
    imp_path = out_dir / "feature_importance_permutation.csv"
    imp_df.to_csv(imp_path, index=False, float_format="%.8f")
    print(f"[table] {imp_path}", flush=True)
    if not args.no_plot:
        plot_importance(imp_df, out_dir, args.sort_metric)

    print(f"[OK] feature importance outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
