#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Static-RNN sampling-method ablations on the S2 test set."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import run_static_rnn_lowvis_eval_journal as journal


SAMPLING_EXPERIMENTS: Dict[int, str] = {
    0: "natural_shuffle",
    1: "current_stratified",
    2: "light_lowvis_oversample",
    3: "heavy_lowvis_oversample",
    4: "mild_lowvis_oversample",
}

SAMPLING_LABELS: Dict[str, str] = {
    "natural_shuffle": "No Low-vis event oversampling",
    "current_stratified": "With Low-vis event oversampling",
    "light_lowvis_oversample": "Light Low-vis event oversampling",
    "heavy_lowvis_oversample": "Heavy Low-vis event oversampling",
    "mild_lowvis_oversample": "Mild Low-vis event oversampling",
}

CLASS_NAMES = ("Fog", "Mist", "Clear")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Static-RNN sampling-method ablation checkpoints.")
    p.add_argument(
        "--config_json",
        default=os.environ.get("PAPER_EVAL_CONFIG", str(journal.VIS_EVAL_DIR / journal.DEFAULT_CONFIG_NAME)),
    )
    p.add_argument("--base", default=str(journal.DEFAULT_BASE))
    p.add_argument("--train_dir", default=str(journal.DEFAULT_TRAIN_DIR))
    p.add_argument("--data_dir", default=journal.DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="static_rnn_sampling_eval_results")
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--stage_tag", default=journal.DEFAULT_STAGE_TAG)
    p.add_argument("--meta_time_shift_hours", type=float, default=0.0)

    p.add_argument("--sampling_run_prefix", "--matrix_run_prefix", dest="sampling_run_prefix", default=os.environ.get("SAMPLING_RUN_PREFIX", os.environ.get("LOWVIS_RNN_RUN_PREFIX", "")))
    p.add_argument(
        "--current_main_run_id",
        default=os.environ.get("CURRENT_MAIN_RUN_ID", os.environ.get("MAIN_RUN_ID", "")),
        help="Existing mainline run id to use as experiment 1/current_stratified instead of retraining it.",
    )
    p.add_argument(
        "--current_main_ckpt",
        default=os.environ.get("CURRENT_MAIN_CKPT", os.environ.get("MAIN_CKPT", "")),
        help="Existing mainline checkpoint path to use as experiment 1/current_stratified.",
    )
    p.add_argument(
        "--experiments",
        "--matrix_experiments",
        dest="experiments",
        default=os.environ.get("LOWVIS_RNN_SAMPLING_EXPERIMENTS", "0:1"),
        help="Sampling experiment ids separated by colon/comma/space.",
    )
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--no_auto_latest", action="store_true")

    p.add_argument("--threshold_source", choices=["checkpoint", "cli", "argmax"], default="argmax")
    p.add_argument("--fog_th", type=float, default=0.5)
    p.add_argument("--mist_th", type=float, default=0.5)
    p.add_argument("--ifs_vis_nc", default="VIS_IDW_KDTree_20250101_20251231.nc")
    p.add_argument("--ifs_vis_var", default="VIS")
    p.add_argument("--shp_path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    p.add_argument("--run_event_eval", action="store_true", help="Run main-eval style widespread event detection and peak-case figures for each sampling experiment.")
    p.add_argument("--event_top_k", type=int, default=3)
    p.add_argument("--event_window_hours", type=int, default=3)
    p.add_argument("--event_min_fog_stations", type=int, default=80)
    p.add_argument("--event_min_regions", type=int, default=3)
    p.add_argument("--event_min_lon_span", type=float, default=10.0)
    p.add_argument("--event_min_lat_span", type=float, default=4.0)
    p.add_argument("--event_gap_hours", type=int, default=24)
    p.add_argument("--event_preferred_times", default="10-30 22:00")
    p.add_argument("--event_env_source", choices=["grid", "none"], default="none")
    p.add_argument("--event_env_max_events", type=int, default=3)
    p.add_argument("--event_env_rh2m_var", default="rh2m")
    p.add_argument("--event_env_rh2m_vmin", type=float, default=40.0)
    p.add_argument("--event_env_rh2m_vmax", type=float, default=100.0)
    p.add_argument(
        "--event_env_tianji_template",
        default="/tj01/sd3op/userpp/pp_data/{init_yyyymmddhh}/stage26Q/multi_model_sources/{init_yyyymmddhh}/{variable}.nc",
    )
    p.add_argument("--event_env_pm10_dir", default="pm10_data")
    p.add_argument("--event_env_pm10_var", default="pm10")
    p.add_argument("--event_env_pm10_vmin", type=float, default=0.0)
    p.add_argument("--event_env_pm10_vmax", type=float, default=240.0)
    p.add_argument("--rank_metric", default="low_vis_csi")
    p.add_argument("--save_outputs", action="store_true", help="Save per-experiment probability arrays for debugging.")
    return p.parse_args()


def parse_id_list(value: str) -> List[int]:
    out: List[int] = []
    for token in (value or "").replace(",", ":").replace(" ", ":").split(":"):
        token = token.strip()
        if not token:
            continue
        exp_id = int(token)
        if exp_id not in SAMPLING_EXPERIMENTS:
            raise ValueError(f"Unknown sampling experiment id={exp_id}; valid ids={sorted(SAMPLING_EXPERIMENTS)}")
        out.append(exp_id)
    return out or sorted(SAMPLING_EXPERIMENTS)


def checkpoint_suffix(exp_id: int, name: str, stage_tag: str) -> str:
    return f"_{exp_id}_{name}_{stage_tag}_best_score.pt"


def discover_latest_prefix(ckpt_dir: Path, stage_tag: str, exp_ids: Sequence[int]) -> Optional[str]:
    runs: Dict[str, Dict[str, object]] = {}
    for exp_id in exp_ids:
        name = SAMPLING_EXPERIMENTS[exp_id]
        suffix = checkpoint_suffix(exp_id, name, stage_tag)
        for path in ckpt_dir.glob(f"*{suffix}"):
            if not path.name.endswith(suffix):
                continue
            prefix = path.name[: -len(suffix)]
            rec = runs.setdefault(prefix, {"ids": set(), "mtime": 0.0})
            rec["ids"].add(exp_id)
            rec["mtime"] = max(float(rec["mtime"]), path.stat().st_mtime)
    if not runs:
        return None
    ranked = sorted(runs.items(), key=lambda item: (len(item[1]["ids"]), float(item[1]["mtime"])), reverse=True)
    return ranked[0][0]


def build_targets(
    sampling_run_prefix: str,
    base: Path,
    ckpt_dir: Path,
    stage_tag: str,
    exp_ids: Sequence[int],
    allow_missing: bool,
    current_main_run_id: str = "",
    current_main_ckpt: str = "",
) -> List[journal.EvalTarget]:
    targets: List[journal.EvalTarget] = []
    spec = journal.VARIANTS[0]
    for exp_id in exp_ids:
        name = SAMPLING_EXPERIMENTS[exp_id]
        if exp_id == 1 and (str(current_main_run_id).strip() or str(current_main_ckpt).strip()):
            if str(current_main_ckpt).strip():
                checkpoint = journal.as_abs_under(base, str(current_main_ckpt).strip())
                suffix = f"_{stage_tag}_best_score.pt"
                run_id = str(current_main_run_id).strip() or checkpoint.name.removesuffix(suffix)
            else:
                run_id = str(current_main_run_id).strip()
                checkpoint = ckpt_dir / f"{run_id}_{stage_tag}_best_score.pt"
        else:
            run_id = f"{sampling_run_prefix}_{exp_id}_{name}"
            checkpoint = ckpt_dir / f"{run_id}_{stage_tag}_best_score.pt"
        if not checkpoint.exists():
            if allow_missing:
                print(f"[skip] missing checkpoint: {checkpoint}", flush=True)
                continue
            raise FileNotFoundError(f"Missing checkpoint for sampling experiment {exp_id} {name}: {checkpoint}")
        targets.append(journal.EvalTarget(label=name, run_id=run_id, checkpoint=checkpoint, variant=spec))
    return targets


def confusion_counts(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    valid = (y_true >= 0) & (y_true <= 2) & (pred >= 0) & (pred <= 2)
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y_true[valid], pred[valid]), 1)
    return cm


def sampling_fields(ckpt_meta: Dict[str, object]) -> Dict[str, object]:
    sampling = ckpt_meta.get("sampling", {}) if isinstance(ckpt_meta, dict) else {}
    if not isinstance(sampling, dict):
        sampling = {}
    batch_counts = sampling.get("batch_class_counts", {})
    train_counts = sampling.get("train_class_counts", {})
    if not isinstance(batch_counts, dict):
        batch_counts = {}
    if not isinstance(train_counts, dict):
        train_counts = {}
    return {
        "sampler_mode": sampling.get("sampler_mode", ""),
        "sample_fog_ratio": sampling.get("fog_ratio", np.nan),
        "sample_mist_ratio": sampling.get("mist_ratio", np.nan),
        "batch_fog": batch_counts.get("fog", np.nan),
        "batch_mist": batch_counts.get("mist", np.nan),
        "batch_clear": batch_counts.get("clear", np.nan),
        "train_fog": train_counts.get("fog", np.nan),
        "train_mist": train_counts.get("mist", np.nan),
        "train_clear": train_counts.get("clear", np.nan),
    }


def class_metric_rows(
    label: str,
    run_id: str,
    experiment_id: int,
    cm: np.ndarray,
    decision_meta: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cls_id, class_name in enumerate(CLASS_NAMES):
        tp = int(cm[cls_id, cls_id])
        fp = int(cm[:, cls_id].sum() - cm[cls_id, cls_id])
        fn = int(cm[cls_id, :].sum() - cm[cls_id, cls_id])
        support = int(cm[cls_id, :].sum())
        predicted = int(cm[:, cls_id].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        far = fp / (tp + fp) if (tp + fp) else 0.0
        rows.append(
            {
                "experiment_id": experiment_id,
                "label": label,
                "display_label": SAMPLING_LABELS.get(label, label),
                "run_id": run_id,
                "class_id": cls_id,
                "class_name": class_name,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "csi": csi,
                "far": far,
                **decision_meta,
            }
        )
    return rows


def confusion_rows(label: str, run_id: str, experiment_id: int, cm: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for true_id, true_name in enumerate(CLASS_NAMES):
        row_sum = int(cm[true_id, :].sum())
        for pred_id, pred_name in enumerate(CLASS_NAMES):
            count = int(cm[true_id, pred_id])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "label": label,
                    "display_label": SAMPLING_LABELS.get(label, label),
                    "run_id": run_id,
                    "true_class_id": true_id,
                    "true_class_name": true_name,
                    "pred_class_id": pred_id,
                    "pred_class_name": pred_name,
                    "count": count,
                    "row_fraction": count / row_sum if row_sum else 0.0,
                }
            )
    return rows


def simple_markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return ""
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def evaluate_target(
    args: argparse.Namespace,
    target: journal.EvalTarget,
    experiment_id: int,
    x_path: Path,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    meta: pd.DataFrame,
    base: Path,
    ckpt_dir: Path,
    out_dir: Path,
    mod,
    layout,
    device,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    import joblib

    print(f"\n=== Evaluating sampling experiment {experiment_id}: {target.label} ===", flush=True)
    print(f"checkpoint: {target.checkpoint}", flush=True)
    scaler_path = journal.resolve_scaler_path(base, ckpt_dir, target.run_id, layout, target.variant.use_pm, None)
    scaler = joblib.load(scaler_path)

    state, ckpt_meta = journal.load_checkpoint_payload(target.checkpoint, device)
    model = journal.make_model(mod, layout, target.variant, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing keys: {len(missing)} first={missing[:5]}", flush=True)
    if unexpected:
        print(f"  [ckpt] unexpected keys: {len(unexpected)} first={unexpected[:5]}", flush=True)

    probs = journal.run_static_inference(
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
    pred, decision_meta = journal.predict_from_probs(args, probs, ckpt_meta)
    metrics = journal.classification_metrics(y_cls, pred, probs=probs)
    cm = confusion_counts(y_cls, pred)

    if args.save_outputs:
        np.save(out_dir / f"{experiment_id}_{target.label}_probs.npy", probs.astype(np.float32))

    overall = {
        "experiment_id": experiment_id,
        "label": target.label,
        "display_label": SAMPLING_LABELS.get(target.label, target.label),
        "run_id": target.run_id,
        "checkpoint": str(target.checkpoint),
        "scaler": str(scaler_path),
        **sampling_fields(ckpt_meta),
        **decision_meta,
        **metrics,
        "ckpt_score": ckpt_meta.get("score") if isinstance(ckpt_meta, dict) else None,
        "ckpt_step": ckpt_meta.get("step") if isinstance(ckpt_meta, dict) else None,
        "ckpt_threshold_mode": ckpt_meta.get("threshold_mode") if isinstance(ckpt_meta, dict) else None,
    }
    event_rows = []
    if bool(getattr(args, "run_event_eval", False)):
        event_rows = run_event_eval_for_target(
            args=args,
            target=target,
            experiment_id=experiment_id,
            base=base,
            out_dir=out_dir,
            x_path=x_path,
            y_cls=y_cls,
            y_raw=y_raw,
            meta=meta,
            pred=pred,
            probs=probs,
            scaler_path=scaler_path,
        )
    return (
        overall,
        class_metric_rows(target.label, target.run_id, experiment_id, cm, decision_meta),
        confusion_rows(target.label, target.run_id, experiment_id, cm),
        event_rows,
    )


def run_event_eval_for_target(
    args: argparse.Namespace,
    target: journal.EvalTarget,
    experiment_id: int,
    base: Path,
    out_dir: Path,
    x_path: Path,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    meta: pd.DataFrame,
    pred: np.ndarray,
    probs: np.ndarray,
    scaler_path: Path,
) -> List[Dict[str, object]]:
    event_dir = out_dir / f"{experiment_id}_{target.label}_event_eval"
    event_dir.mkdir(parents=True, exist_ok=True)
    ifs_nc = journal.as_abs_under(base, args.ifs_vis_nc)
    manifest = journal.Manifest(event_dir)
    sources = [
        str(x_path),
        str(journal.as_abs_under(base, args.data_dir) / f"y_{getattr(args, 'eval_split', 'test')}.npy"),
        str(journal.as_abs_under(base, args.data_dir) / f"meta_{getattr(args, 'eval_split', 'test')}.csv"),
        str(target.checkpoint),
        str(scaler_path),
    ]

    ifs_pred = ifs_vis = ifs_valid = None
    if ifs_nc.exists():
        try:
            ifs_pred, ifs_vis, ifs_valid = journal.load_ifs_diagnostic(meta, ifs_nc, args.ifs_vis_var)
            sources.append(str(ifs_nc))
        except Exception as exc:
            print(f"[events] IFS diagnostic baseline skipped for {target.label}: {exc}", flush=True)
            ifs_valid = np.zeros(len(y_cls), dtype=bool)
    else:
        print(f"[events] IFS diagnostic NetCDF not found: {ifs_nc}; event metrics will be skipped.", flush=True)
        ifs_valid = np.zeros(len(y_cls), dtype=bool)

    np.save(event_dir / "probs.npy", probs.astype(np.float32))
    eval_df = journal.export_per_sample(
        event_dir / "per_sample_eval.csv",
        meta,
        y_cls,
        y_raw,
        pred,
        probs,
        ifs_pred=ifs_pred,
        ifs_vis=ifs_vis,
        ifs_valid=ifs_valid,
    )

    if ifs_pred is not None and ifs_vis is not None and ifs_valid is not None and int(np.sum(ifs_valid)) > 0:
        shp = journal.read_shapefile(args.shp_path) if str(getattr(args, "shp_path", "") or "") else None
        journal.run_event_plots(
            args,
            base,
            event_dir,
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
    manifest.add(
        "per_sample_eval.csv",
        sources,
        notes="Per-sample sampling-ablation predictions used for optional widespread-event case evaluation.",
        n=int(len(y_cls)),
        matched_ifs=int(np.sum(ifs_valid)) if ifs_valid is not None else None,
    )
    manifest.write()

    run_config = {
        "experiment_id": int(experiment_id),
        "label": target.label,
        "display_label": SAMPLING_LABELS.get(target.label, target.label),
        "run_id": target.run_id,
        "event_dir": str(event_dir),
        "ifs_vis_nc": str(ifs_nc),
        "event_top_k": int(args.event_top_k),
        "event_window_hours": int(args.event_window_hours),
        "event_env_source": str(args.event_env_source),
    }
    (event_dir / "event_eval_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = event_dir / "fig9_event_summary_metrics.csv"
    if not summary_path.exists():
        return []
    event_df = pd.read_csv(summary_path)
    event_df.insert(0, "event_eval_dir", str(event_dir))
    event_df.insert(0, "run_id", target.run_id)
    event_df.insert(0, "display_label", SAMPLING_LABELS.get(target.label, target.label))
    event_df.insert(0, "label", target.label)
    event_df.insert(0, "experiment_id", int(experiment_id))
    return event_df.to_dict("records")


def write_markdown_summary(out_dir: Path, overall: pd.DataFrame, per_class: pd.DataFrame, rank_metric: str) -> None:
    lines = ["# Static-RNN Sampling-Method Ablation Metrics", ""]
    if rank_metric in overall.columns:
        ranked = overall.sort_values(rank_metric, ascending=False)
        best = ranked.iloc[0]
        lines.append(f"- Best by `{rank_metric}`: `{best['display_label']}` ({float(best[rank_metric]):.6f})")
        lines.append("")
    show_cols = [
        "experiment_id",
        "display_label",
        "sampler_mode",
        "sample_fog_ratio",
        "sample_mist_ratio",
        "Fog_CSI",
        "Fog_R",
        "Mist_CSI",
        "Mist_R",
        "low_vis_precision",
        "low_vis_recall",
        "low_vis_csi",
        "false_positive_rate",
        "accuracy",
    ]
    lines.append("## Overall")
    lines.append("")
    lines.append(simple_markdown_table(overall, show_cols))
    lines.append("")
    lines.append("## Per Class")
    lines.append("")
    cls_cols = ["experiment_id", "display_label", "class_name", "support", "predicted", "precision", "recall", "csi", "far"]
    lines.append(simple_markdown_table(per_class, cls_cols))
    lines.append("")
    (out_dir / "sampling_ablation_metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args = journal.apply_paper_eval_config(args, "static_rnn_sampling_eval", default_dir=journal.VIS_EVAL_DIR)

    base = Path(args.base).expanduser()
    train_dir = Path(args.train_dir).expanduser()
    data_dir = journal.as_abs_under(base, args.data_dir)
    ckpt_dir = journal.as_abs_under(base, args.ckpt_dir)
    out_dir = journal.unique_dir(journal.as_abs_under(base, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_ids = parse_id_list(args.experiments)
    if not args.sampling_run_prefix and not args.no_auto_latest:
        args.sampling_run_prefix = discover_latest_prefix(ckpt_dir, args.stage_tag, exp_ids) or ""
        if args.sampling_run_prefix:
            print(f"[auto] sampling_run_prefix={args.sampling_run_prefix}", flush=True)
    needs_sampling_prefix = any(exp_id != 1 for exp_id in exp_ids) or not (
        str(getattr(args, "current_main_run_id", "") or "").strip()
        or str(getattr(args, "current_main_ckpt", "") or "").strip()
    )
    if needs_sampling_prefix and not args.sampling_run_prefix:
        raise ValueError("Pass --sampling_run_prefix, or leave auto-detect enabled with matching checkpoints in --ckpt_dir.")

    x_path, y_cls, y_raw, meta = journal.load_main_data(
        data_dir,
        args.limit_samples,
        getattr(args, "meta_time_shift_hours", 0.0),
    )
    dyn, fe = journal.infer_layout_from_x(x_path, args.window_size)
    mod = journal.load_static_rnn_module(train_dir)
    layout = mod.Layout(window_size=args.window_size, dyn_vars=dyn, fe_dim=fe)
    device = journal.resolve_device(args.device)
    targets = build_targets(
        args.sampling_run_prefix,
        base,
        ckpt_dir,
        args.stage_tag,
        exp_ids,
        args.allow_missing,
        current_main_run_id=getattr(args, "current_main_run_id", ""),
        current_main_ckpt=getattr(args, "current_main_ckpt", ""),
    )
    if not targets:
        raise FileNotFoundError("No sampling-ablation checkpoints were selected for evaluation.")

    print("Static-RNN sampling-method ablation evaluation", flush=True)
    print(f"base      : {base}", flush=True)
    print(f"train_dir : {train_dir}", flush=True)
    print(f"data_dir  : {data_dir}", flush=True)
    print(f"ckpt_dir  : {ckpt_dir}", flush=True)
    print(f"out_dir   : {out_dir}", flush=True)
    print(f"layout    : {layout}", flush=True)
    print(f"device    : {device}", flush=True)
    print(f"targets   : {[t.run_id for t in targets]}", flush=True)

    overall_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    id_by_label = {name: exp_id for exp_id, name in SAMPLING_EXPERIMENTS.items()}
    for target in targets:
        exp_id = id_by_label[target.label]
        overall, per_class, confusion, events = evaluate_target(
            args,
            target,
            exp_id,
            x_path,
            y_cls,
            y_raw,
            meta,
            base,
            ckpt_dir,
            out_dir,
            mod,
            layout,
            device,
        )
        overall_rows.append(overall)
        class_rows.extend(per_class)
        cm_rows.extend(confusion)
        event_rows.extend(events)

    overall_df = pd.DataFrame(overall_rows).sort_values("experiment_id")
    per_class_df = pd.DataFrame(class_rows).sort_values(["experiment_id", "class_id"])
    cm_df = pd.DataFrame(cm_rows).sort_values(["experiment_id", "true_class_id", "pred_class_id"])

    overall_path = out_dir / "sampling_ablation_overall_metrics.csv"
    per_class_path = out_dir / "sampling_ablation_per_class_metrics.csv"
    cm_path = out_dir / "sampling_ablation_confusion_counts.csv"
    overall_df.to_csv(overall_path, index=False, float_format="%.8f")
    per_class_df.to_csv(per_class_path, index=False, float_format="%.8f")
    cm_df.to_csv(cm_path, index=False, float_format="%.8f")
    event_summary_path = out_dir / "sampling_ablation_event_summary_metrics.csv"
    if event_rows:
        pd.DataFrame(event_rows).sort_values(["experiment_id", "event_rank"]).to_csv(event_summary_path, index=False, float_format="%.8f")
    write_markdown_summary(out_dir, overall_df, per_class_df, args.rank_metric)

    run_config = {
        "sampling_run_prefix": args.sampling_run_prefix,
        "current_main_run_id": str(getattr(args, "current_main_run_id", "") or ""),
        "current_main_ckpt": str(getattr(args, "current_main_ckpt", "") or ""),
        "experiments": {str(k): SAMPLING_EXPERIMENTS[k] for k in exp_ids},
        "labels": SAMPLING_LABELS,
        "base": str(base),
        "train_dir": str(train_dir),
        "data_dir": str(data_dir),
        "ckpt_dir": str(ckpt_dir),
        "out_dir": str(out_dir),
        "layout": asdict(layout),
        "device": str(device),
        "threshold_source": args.threshold_source,
        "fog_th": float(args.fog_th),
        "mist_th": float(args.mist_th),
        "run_event_eval": bool(args.run_event_eval),
        "ifs_vis_nc": str(journal.as_abs_under(base, args.ifs_vis_nc)),
        "event_env_source": str(args.event_env_source),
        "limit_samples": int(args.limit_samples),
        "meta_time_shift_hours": float(getattr(args, "meta_time_shift_hours", 0.0) or 0.0),
        "outputs": {
            "overall_metrics": str(overall_path),
            "per_class_metrics": str(per_class_path),
            "confusion_counts": str(cm_path),
            "summary_md": str(out_dir / "sampling_ablation_metrics_summary.md"),
            "event_summary_metrics": str(event_summary_path) if event_rows else "",
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[table] {overall_path}", flush=True)
    print(f"[table] {per_class_path}", flush=True)
    print(f"[table] {cm_path}", flush=True)
    if event_rows:
        print(f"[table] {event_summary_path}", flush=True)


if __name__ == "__main__":
    main()
