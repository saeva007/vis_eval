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
}

SAMPLING_LABELS: Dict[str, str] = {
    "natural_shuffle": "Natural shuffle",
    "current_stratified": "Current stratified",
    "light_lowvis_oversample": "Light low-vis oversampling",
    "heavy_lowvis_oversample": "Heavy low-vis oversampling",
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
        "--experiments",
        "--matrix_experiments",
        dest="experiments",
        default=os.environ.get("LOWVIS_RNN_SAMPLING_EXPERIMENTS", "0:1:2:3"),
        help="Sampling experiment ids separated by colon/comma/space.",
    )
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--no_auto_latest", action="store_true")

    p.add_argument("--threshold_source", choices=["checkpoint", "cli", "argmax"], default="argmax")
    p.add_argument("--fog_th", type=float, default=0.5)
    p.add_argument("--mist_th", type=float, default=0.5)
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
    ckpt_dir: Path,
    stage_tag: str,
    exp_ids: Sequence[int],
    allow_missing: bool,
) -> List[journal.EvalTarget]:
    targets: List[journal.EvalTarget] = []
    spec = journal.VARIANTS[0]
    for exp_id in exp_ids:
        name = SAMPLING_EXPERIMENTS[exp_id]
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
    base: Path,
    ckpt_dir: Path,
    out_dir: Path,
    mod,
    layout,
    device,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
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
    return (
        overall,
        class_metric_rows(target.label, target.run_id, experiment_id, cm, decision_meta),
        confusion_rows(target.label, target.run_id, experiment_id, cm),
    )


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
    if not args.sampling_run_prefix:
        raise ValueError("Pass --sampling_run_prefix, or leave auto-detect enabled with matching checkpoints in --ckpt_dir.")

    x_path, y_cls, _y_raw, _meta = journal.load_main_data(
        data_dir,
        args.limit_samples,
        getattr(args, "meta_time_shift_hours", 0.0),
    )
    dyn, fe = journal.infer_layout_from_x(x_path, args.window_size)
    mod = journal.load_static_rnn_module(train_dir)
    layout = mod.Layout(window_size=args.window_size, dyn_vars=dyn, fe_dim=fe)
    device = journal.resolve_device(args.device)
    targets = build_targets(args.sampling_run_prefix, ckpt_dir, args.stage_tag, exp_ids, args.allow_missing)
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
    id_by_label = {name: exp_id for exp_id, name in SAMPLING_EXPERIMENTS.items()}
    for target in targets:
        exp_id = id_by_label[target.label]
        overall, per_class, confusion = evaluate_target(
            args,
            target,
            exp_id,
            x_path,
            y_cls,
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

    overall_df = pd.DataFrame(overall_rows).sort_values("experiment_id")
    per_class_df = pd.DataFrame(class_rows).sort_values(["experiment_id", "class_id"])
    cm_df = pd.DataFrame(cm_rows).sort_values(["experiment_id", "true_class_id", "pred_class_id"])

    overall_path = out_dir / "sampling_ablation_overall_metrics.csv"
    per_class_path = out_dir / "sampling_ablation_per_class_metrics.csv"
    cm_path = out_dir / "sampling_ablation_confusion_counts.csv"
    overall_df.to_csv(overall_path, index=False, float_format="%.8f")
    per_class_df.to_csv(per_class_path, index=False, float_format="%.8f")
    cm_df.to_csv(cm_path, index=False, float_format="%.8f")
    write_markdown_summary(out_dir, overall_df, per_class_df, args.rank_metric)

    run_config = {
        "sampling_run_prefix": args.sampling_run_prefix,
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
        "limit_samples": int(args.limit_samples),
        "meta_time_shift_hours": float(getattr(args, "meta_time_shift_hours", 0.0) or 0.0),
        "outputs": {
            "overall_metrics": str(overall_path),
            "per_class_metrics": str(per_class_path),
            "confusion_counts": str(cm_path),
            "summary_md": str(out_dir / "sampling_ablation_metrics_summary.md"),
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[table] {overall_path}", flush=True)
    print(f"[table] {per_class_path}", flush=True)
    print(f"[table] {cm_path}", flush=True)


if __name__ == "__main__":
    main()
