#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate and plot feature-engineering ablations in the journal figure style.

Two workflows are supported:

1. Plot an existing ablation metrics table:
     python vis_eval/run_feature_engineering_ablation_journal.py \
       --mode plot --metrics_csv paper_eval_results_pm10_pm25_journal/feature_engineering_ablation_metrics.csv

2. Evaluate a variant specification table and plot the resulting metrics:
     python vis_eval/run_feature_engineering_ablation_journal.py \
       --mode all --spec_csv feature_engineering_ablation_spec.csv

The spec CSV should contain at least:
  variant, checkpoint

Optional columns:
  label, ablation_mode, custom_indices, fog_th, mist_th, threshold_rule,
  temperature, season_th_path

The ablation_mode names match PMST_net_test_11_s2_pm10_fe_ablation.py.  The
same FE mask is applied at inference time so that the test-time feature stream
matches the training-time intervention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
LOCAL_ROOT = VIS_EVAL_DIR.parent
for _p in (str(LOCAL_ROOT), str(VIS_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_paper_eval_pm10_pm25_journal import (  # noqa: E402
    CLEAR_COLOR,
    FOG_COLOR,
    IFS_DIAG_COLOR,
    MIST_COLOR,
    PMST_COLOR,
    Manifest,
    abs_under_base,
    add_panel_label,
    classification_metrics,
    import_model_class,
    infer_layout_from_x,
    load_checkpoint_into_model,
    load_main_data,
    pred_from_probs_rule,
    prepare_batch_rows,
    resolve_device,
    resolve_model_py,
    save_fig_pair,
    setup_journal_style,
)


FE_FEATURE_NAMES: List[str] = [
    "saturation_dpd_proxy",
    "wind_favorability",
    "inversion_weak_wind_stability",
    "night_clear_sky_radiative_cooling",
    "rh2m_minus_rh925",
    "composite_fog_potential",
    "rh2m_delta_3h",
    "rh2m_delta_6h",
    "rh2m_std_12h",
    "rh2m_range_12h",
    "t2m_delta_3h",
    "t2m_delta_6h",
    "t2m_std_12h",
    "t2m_range_12h",
    "wspd10_delta_3h",
    "wspd10_delta_6h",
    "wspd10_std_12h",
    "wspd10_range_12h",
    "rh2m_acceleration",
    "moist_cold_proxy",
    "night_low_cloud_proxy",
    "cold_humid_weak_wind_indicator",
    "rh_low_cloud_ratio",
    "rh_squared_proxy",
    "low_level_shear_magnitude",
    "low_level_direction_turning",
    "convective_wet_proxy",
    "daytime_mixing_proxy",
    "ventilation_proxy",
    "moisture_stratification",
    "vertical_velocity_contrast",
    "warm_instability_proxy",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]


FE_GROUPS: Dict[str, Tuple[int, ...]] = {
    "core_physics": tuple(range(0, 6)),
    "temporal_stats": tuple(range(6, 19)),
    "empirical_flags": tuple(range(19, 24)),
    "boundary_layer": tuple(range(24, 32)),
    "time_cyc": tuple(range(32, 36)),
}


MODE_LABELS: Dict[str, str] = {
    "full": "Full FE",
    "no_fe_all": "No FE values",
    "no_core_physics": "No core physics",
    "no_temporal_stats": "No temporal stats",
    "no_empirical_flags": "No empirical flags",
    "no_boundary_layer": "No boundary layer",
    "no_time_cyc": "No time cycle",
    "only_core_physics": "Only core physics",
    "only_temporal_stats": "Only temporal stats",
    "only_time_cyc": "Only time cycle",
}


VARIANT_COLORS: List[str] = [
    PMST_COLOR,
    "#8DA0CB",
    MIST_COLOR,
    "#66A61E",
    "#A6761D",
    "#E78AC3",
    IFS_DIAG_COLOR,
    "#4C78A8",
    "#F58518",
    "#54A24B",
]


def _bounded(indices: Iterable[int], fe_dim: int) -> List[int]:
    return sorted({int(i) for i in indices if 0 <= int(i) < int(fe_dim)})


def parse_index_spec(spec: str, fe_dim: int) -> List[int]:
    if not str(spec).strip():
        return []
    out: List[int] = []
    for raw in str(spec).split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
            start = int(left) if left.strip() else 0
            stop = int(right) if right.strip() else fe_dim
            out.extend(range(start, stop))
        else:
            out.append(int(token))
    return _bounded(out, fe_dim)


def mask_for_mode(mode: str, fe_dim: int, custom_indices: str = "") -> np.ndarray:
    mode = str(mode or "full")
    all_indices = set(range(fe_dim))
    if mode == "full":
        dropped: List[int] = []
    elif mode == "no_fe_all":
        dropped = list(range(fe_dim))
    elif mode == "custom_drop":
        dropped = parse_index_spec(custom_indices, fe_dim)
    elif mode == "custom_keep":
        keep = set(parse_index_spec(custom_indices, fe_dim))
        dropped = sorted(all_indices - keep)
    elif mode.startswith("no_"):
        dropped = _bounded(FE_GROUPS[mode[3:]], fe_dim)
    elif mode.startswith("only_"):
        keep = set(_bounded(FE_GROUPS[mode[5:]], fe_dim))
        dropped = sorted(all_indices - keep)
    else:
        raise ValueError(f"Unknown ablation_mode={mode}")
    mask = np.ones(fe_dim, dtype=np.float32)
    if dropped:
        mask[np.asarray(dropped, dtype=np.int64)] = 0.0
    return mask


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate and plot PMST FE ablations.")
    ap.add_argument("--mode", choices=["eval", "plot", "all"], default="all")
    ap.add_argument("--base", default="/public/home/putianshu/vis_mlp")
    ap.add_argument("--data_dir", default="ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2")
    ap.add_argument("--model_py", default="")
    ap.add_argument("--scaler_path", default="")
    ap.add_argument("--spec_csv", default="")
    ap.add_argument("--metrics_csv", default="")
    ap.add_argument("--out_dir", default="paper_eval_results_pm10_pm25_journal/feature_engineering_ablation")
    ap.add_argument("--write_template", default="", help="Write a spec CSV template and exit.")

    ap.add_argument("--window_size", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=0)
    ap.add_argument("--extra_feat_dim", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit_samples", type=int, default=0)

    ap.add_argument("--fog_th", type=float, default=0.10)
    ap.add_argument("--mist_th", type=float, default=0.42)
    ap.add_argument("--threshold_rule", choices=["default", "mutual", "joint"], default="mutual")
    ap.add_argument("--use_calibration", action="store_true")
    ap.add_argument("--baseline_variant", default="full")
    ap.add_argument("--variant_order", default="", help="Comma-separated order; defaults to CSV order.")
    ap.add_argument("--figure_stem", default="fig12_feature_engineering_ablation")
    return ap


def write_template(path: Path) -> None:
    rows = [
        {
            "variant": "full",
            "label": "Full FE",
            "ablation_mode": "full",
            "checkpoint": "checkpoints/exp_1776227576_pm10_more_temp_search_S2_PhaseB_best_score.pt",
            "season_th_path": "checkpoints/exp_1776227576_pm10_more_temp_search_season_thresholds.pt",
            "fog_th": 0.10,
            "mist_th": 0.42,
            "threshold_rule": "mutual",
            "temperature": "",
            "custom_indices": "",
        },
        {
            "variant": "no_fe_all",
            "label": "No FE values",
            "ablation_mode": "no_fe_all",
            "checkpoint": "checkpoints/exp_1776227576_pm10_more_temp_search_feabl_no_fe_all_S2_PhaseB_best_score.pt",
            "season_th_path": "checkpoints/exp_1776227576_pm10_more_temp_search_feabl_no_fe_all_season_thresholds.pt",
            "fog_th": 0.10,
            "mist_th": 0.42,
            "threshold_rule": "mutual",
            "temperature": "",
            "custom_indices": "",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[template] wrote {path}", flush=True)


def first_present(row: pd.Series, names: Sequence[str], default=None):
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]) != "":
            return row[name]
    return default


def resolve_optional_path(base: Path, value) -> Optional[Path]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return abs_under_base(base, str(value).strip())


def load_temperature(base: Path, row: pd.Series, use_calibration: bool) -> Optional[float]:
    value = first_present(row, ["temperature"], default=None)
    if value is not None:
        try:
            if math.isfinite(float(value)) and float(value) > 0:
                return float(value)
        except Exception:
            pass
    if not use_calibration:
        return None
    season_path = resolve_optional_path(base, first_present(row, ["season_th_path", "season_thresholds"], None))
    if season_path is None or not season_path.exists():
        return None
    import torch

    try:
        try:
            cal = torch.load(season_path, map_location="cpu", weights_only=True)
        except TypeError:
            cal = torch.load(season_path, map_location="cpu")
        temp = cal.get("temperature") if isinstance(cal, dict) else None
        if temp is not None and float(temp) > 0:
            return float(temp)
    except Exception as exc:
        print(f"  [WARN] could not load temperature from {season_path}: {exc}", flush=True)
    return None


def run_model_inference_masked(
    x_path: Path,
    scaler,
    model,
    device,
    batch_size: int,
    window_size: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    fe_mask: np.ndarray,
    limit_samples: int = 0,
    temperature: Optional[float] = None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    X = np.load(x_path, mmap_mode="r")
    n = len(X) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(X))
    temp = 1.0 if temperature is None else max(float(temperature), 1e-6)
    split_dyn = window_size * dyn_vars_count
    fe_start = split_dyn + 6
    mask = np.ones(extra_feat_dim, dtype=np.float32)
    if fe_mask.size:
        mask[: min(extra_feat_dim, fe_mask.size)] = fe_mask[: min(extra_feat_dim, fe_mask.size)]

    out = []
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = prepare_batch_rows(rows, scaler, window_size, dyn_vars_count, extra_feat_dim)
        if extra_feat_dim > 0 and np.any(mask == 0):
            final[:, fe_start : fe_start + extra_feat_dim] *= mask[None, :]
        x = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits = model(x)[0]
            probs = F.softmax(logits / temp, dim=1)
        out.append(probs.detach().cpu().numpy())
        if start == 0 or end == n or (start // max(batch_size, 1)) % 20 == 0:
            print(f"  [inference] {end}/{n}", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def evaluate_specs(args: argparse.Namespace, base: Path, out_dir: Path, manifest: Manifest) -> pd.DataFrame:
    import joblib

    if not args.spec_csv:
        raise ValueError("--spec_csv is required for --mode eval/all")
    spec_path = abs_under_base(base, args.spec_csv)
    specs = pd.read_csv(spec_path)
    if "variant" not in specs or "checkpoint" not in specs:
        raise ValueError("spec CSV must contain at least variant and checkpoint columns")

    data_dir = abs_under_base(base, args.data_dir)
    x_path, y_cls, _, _ = load_main_data(data_dir, args.limit_samples)
    dyn_inferred, fe_inferred = infer_layout_from_x(x_path, args.window_size)
    dyn = int(args.dyn_vars_count or dyn_inferred)
    fe = int(args.extra_feat_dim or fe_inferred)
    print(f"[layout] dyn_vars_count={dyn}, extra_feat_dim={fe}", flush=True)

    scaler_path = abs_under_base(
        base,
        args.scaler_path or f"checkpoints/robust_scaler_w{args.window_size}_dyn{dyn}_s2_48h_pm10.pkl",
    )
    scaler = joblib.load(scaler_path)
    model_cls = import_model_class(resolve_model_py(base, args.model_py))
    device = resolve_device(args.device)

    import torch

    rows = []
    for _, row in specs.iterrows():
        variant = str(row["variant"])
        label = str(first_present(row, ["label"], MODE_LABELS.get(variant, variant)))
        mode = str(first_present(row, ["ablation_mode"], variant))
        custom_indices = str(first_present(row, ["custom_indices"], ""))
        ckpt = abs_under_base(base, str(row["checkpoint"]))
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint for variant={variant} not found: {ckpt}")

        fog_th = float(first_present(row, ["fog_th"], args.fog_th))
        mist_th = float(first_present(row, ["mist_th"], args.mist_th))
        threshold_rule = str(first_present(row, ["threshold_rule"], args.threshold_rule))
        temperature = load_temperature(base, row, args.use_calibration)
        fe_mask = mask_for_mode(mode, fe, custom_indices)

        print(
            f"\n[variant] {variant} label={label} mode={mode} "
            f"fog_th={fog_th:.3f} mist_th={mist_th:.3f} T={temperature}",
            flush=True,
        )
        model = model_cls(
            dyn_vars_count=dyn,
            window_size=args.window_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            extra_feat_dim=fe,
        ).to(device)
        load_checkpoint_into_model(model, ckpt, device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"[model] DataParallel on {torch.cuda.device_count()} devices", flush=True)

        probs = run_model_inference_masked(
            x_path,
            scaler,
            model,
            device,
            args.batch_size,
            args.window_size,
            dyn,
            fe,
            fe_mask,
            limit_samples=args.limit_samples,
            temperature=temperature,
        )
        pred = pred_from_probs_rule(probs, fog_th, mist_th, threshold_rule)
        metrics = classification_metrics(y_cls[: len(pred)], pred, probs=probs)
        out_row = {
            "source": "pmst",
            "variant": variant,
            "label": label,
            "ablation_mode": mode,
            "sample_scope": "test",
            "checkpoint": str(ckpt),
            "data_dir": str(data_dir),
            "fog_threshold": fog_th,
            "mist_threshold": mist_th,
            "threshold_rule": threshold_rule,
            "temperature": "" if temperature is None else temperature,
            "fe_kept": int(np.sum(fe_mask > 0.5)),
            "fe_dropped": int(np.sum(fe_mask <= 0.5)),
        }
        out_row.update(metrics)
        rows.append(out_row)

        del model, probs, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    metrics_path = out_dir / "feature_engineering_ablation_metrics.csv"
    df.to_csv(metrics_path, index=False, float_format="%.6f")
    print(f"[table] {metrics_path}", flush=True)
    manifest.add(
        "feature_engineering_ablation_metrics.csv",
        [str(spec_path), str(x_path), str(scaler_path)],
        notes="Metrics from FE ablation checkpoints evaluated on the same test set.",
        n=int(df["n"].max()) if "n" in df else None,
    )
    return df


def load_metrics_table(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else out_dir / "feature_engineering_ablation_metrics.csv"
    if not metrics_path.is_absolute():
        metrics_path = Path(args.base) / metrics_path
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if "variant" not in df:
        raise ValueError("metrics CSV must contain a variant column")
    if "label" not in df:
        df["label"] = df["variant"].map(lambda v: MODE_LABELS.get(str(v), str(v)))
    return df


def order_metrics(df: pd.DataFrame, order_spec: str) -> pd.DataFrame:
    out = df.copy()
    if order_spec.strip():
        order = [x.strip() for x in order_spec.split(",") if x.strip()]
    else:
        order = list(dict.fromkeys(out["variant"].astype(str).tolist()))
    rank = {v: i for i, v in enumerate(order)}
    out["_rank"] = out["variant"].astype(str).map(lambda v: rank.get(v, len(rank)))
    return out.sort_values(["_rank", "variant"]).drop(columns=["_rank"])


def ensure_f2_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _fbeta(p_col: str, r_col: str, out_col: str) -> None:
        if out_col in out or p_col not in out or r_col not in out:
            return
        p = pd.to_numeric(out[p_col], errors="coerce")
        r = pd.to_numeric(out[r_col], errors="coerce")
        den = 4.0 * p + r
        out[out_col] = np.where(den > 0, 5.0 * p * r / den, 0.0)

    _fbeta("Fog_P", "Fog_R", "Fog_F2")
    _fbeta("Mist_P", "Mist_R", "Mist_F2")
    _fbeta("low_vis_precision", "low_vis_recall", "low_vis_f2")
    return out


def adaptive_ylim(values: Sequence[float], lower_is_better: bool = False) -> Tuple[float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 0.0, 1.0
    high = float(np.nanmax(vals))
    padded = high * 1.15 + 0.015
    step = 0.05 if padded <= 0.5 else 0.10
    top = min(1.0, max(step * 2, math.ceil(padded / step) * step))
    return 0.0, top


def plot_ablation_bars(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    manifest: Manifest,
    sources: Sequence[str],
    figure_stem: str,
    variant_order: str,
) -> None:
    setup_journal_style()
    df = order_metrics(ensure_f2_metrics(metrics_df), variant_order)
    panels = [
        ("Fog", [("Fog_CSI", "CSI"), ("Fog_R", "Recall"), ("Fog_P", "Precision"), ("Fog_F2", "F2 score")]),
        ("Mist", [("Mist_CSI", "CSI"), ("Mist_R", "Recall"), ("Mist_P", "Precision"), ("Mist_F2", "F2 score")]),
        ("Low visibility", [("low_vis_csi", "CSI"), ("low_vis_recall", "Recall"), ("low_vis_f2", "F2 score"), ("false_positive_rate", "Clear FPR")]),
    ]

    variants = df["variant"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    colors = [VARIANT_COLORS[i % len(VARIANT_COLORS)] for i in range(len(variants))]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), sharey=False)
    for ax_idx, (ax, (title, specs)) in enumerate(zip(axes, panels)):
        x = np.arange(len(specs))
        width = min(0.16, 0.82 / max(len(variants), 1))
        panel_values: List[float] = []
        for vi, (_, row) in enumerate(df.iterrows()):
            vals = [float(row.get(metric, np.nan)) for metric, _ in specs]
            panel_values.extend(vals)
            vals_plot = [0.0 if not np.isfinite(v) else v for v in vals]
            ax.bar(
                x + (vi - (len(variants) - 1) / 2) * width,
                vals_plot,
                width * 0.92,
                color=colors[vi],
                edgecolor="white",
                linewidth=0.4,
                label=labels[vi] if ax_idx == 0 else None,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in specs], rotation=25, ha="right")
        ax.set_ylim(*adaptive_ylim(panel_values))
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        add_panel_label(ax, chr(ord("a") + ax_idx))

    ncol = min(4, max(1, len(variants)))
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    save_fig_pair(
        fig,
        out_dir,
        figure_stem,
        manifest,
        sources,
        notes="Feature-engineering ablation comparison. Clear FPR is lower-is-better.",
        n=int(df["n"].max()) if "n" in df else None,
    )


def write_delta_table(metrics_df: pd.DataFrame, out_dir: Path, baseline_variant: str) -> Optional[Path]:
    metrics_df = ensure_f2_metrics(metrics_df)
    if "variant" not in metrics_df or baseline_variant not in set(metrics_df["variant"].astype(str)):
        print(f"[delta] baseline variant not found: {baseline_variant}; skip delta table.", flush=True)
        return None
    metrics = [
        "Fog_CSI",
        "Fog_R",
        "Fog_P",
        "Fog_F2",
        "Mist_CSI",
        "Mist_R",
        "Mist_P",
        "Mist_F2",
        "low_vis_csi",
        "low_vis_recall",
        "low_vis_precision",
        "low_vis_f2",
        "false_positive_rate",
        "accuracy",
    ]
    base_row = metrics_df[metrics_df["variant"].astype(str) == baseline_variant].iloc[0]
    rows = []
    for _, row in metrics_df.iterrows():
        item = {"variant": row["variant"], "label": row.get("label", row["variant"])}
        for metric in metrics:
            if metric in metrics_df and pd.notna(row.get(metric, np.nan)) and pd.notna(base_row.get(metric, np.nan)):
                item[f"delta_{metric}"] = float(row[metric]) - float(base_row[metric])
        rows.append(item)
    path = out_dir / "feature_engineering_ablation_delta_vs_baseline.csv"
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    print(f"[table] {path}", flush=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.write_template:
        write_template(Path(args.write_template))
        return

    base = Path(args.base).resolve()
    out_dir = abs_under_base(base, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir)

    if args.mode in ("eval", "all"):
        metrics_df = evaluate_specs(args, base, out_dir, manifest)
        metrics_sources = [str(out_dir / "feature_engineering_ablation_metrics.csv")]
    else:
        metrics_df = load_metrics_table(args, out_dir)
        metrics_sources = [str(Path(args.metrics_csv) if args.metrics_csv else out_dir / "feature_engineering_ablation_metrics.csv")]

    if args.mode in ("plot", "all"):
        plot_ablation_bars(
            metrics_df,
            out_dir,
            manifest,
            metrics_sources,
            args.figure_stem,
            args.variant_order,
        )
        delta_path = write_delta_table(metrics_df, out_dir, args.baseline_variant)
        if delta_path is not None:
            manifest.add(
                delta_path.name,
                metrics_sources,
                notes=f"Metric deltas relative to baseline variant {args.baseline_variant}.",
            )

    metadata = {
        "script": str(SCRIPT_PATH),
        "mode": args.mode,
        "data_dir": args.data_dir,
        "metrics_csv": args.metrics_csv,
        "spec_csv": args.spec_csv,
        "baseline_variant": args.baseline_variant,
        "feature_groups": {k: list(v) for k, v in FE_GROUPS.items()},
        "feature_names_0_35": FE_FEATURE_NAMES,
    }
    (out_dir / "feature_engineering_ablation_plot_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest.write()


if __name__ == "__main__":
    main(sys.argv[1:])
