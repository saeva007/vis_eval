#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Static-RNN loss-function ablations on the S2 test set.

The script keeps the journal evaluator's data loading, scaling, model loading,
and threshold logic, but writes compact source-data tables for a paper figure
that compares:

0. plain hard-label cross entropy,
1. plain log-visibility regression,
2. the proposed rare-event focal objective,
3. a plain hard-label focal-loss objective without additional rare-event tricks.
"""

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


LOSS_EXPERIMENTS: Dict[int, str] = {
    0: "simple_ce_classification",
    1: "simple_logvis_regression",
    2: "proposed_rare_event_focal",
    3: "plain_focal_loss",
}

LOSS_LABELS: Dict[str, str] = {
    "simple_ce_classification": "CE classification",
    "simple_logvis_regression": "MSE regression",
    "proposed_rare_event_focal": "Proposed focal",
    "plain_focal_loss": "Plain focal",
}

CLASS_NAMES = ("Fog", "Mist", "Clear")
BOUNDARY_BANDS: Tuple[Tuple[str, str, float, float], ...] = (
    ("fog_mist_400_600m", "Fog-Mist transition (400-600 m)", 400.0, 600.0),
    ("mist_clear_800_1200m", "Mist-Clear transition (800-1200 m)", 800.0, 1200.0),
    ("low_visibility_0_1000m", "All observed low visibility (<1000 m)", 0.0, 1000.0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Static-RNN loss-function ablation checkpoints.")
    p.add_argument(
        "--config_json",
        default=os.environ.get("PAPER_EVAL_CONFIG", str(journal.VIS_EVAL_DIR / journal.DEFAULT_CONFIG_NAME)),
    )
    p.add_argument("--base", default=str(journal.DEFAULT_BASE))
    p.add_argument("--train_dir", default=str(journal.DEFAULT_TRAIN_DIR))
    p.add_argument("--data_dir", default=journal.DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="static_rnn_loss_eval_results")
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--stage_tag", default=journal.DEFAULT_STAGE_TAG)
    p.add_argument("--meta_time_shift_hours", type=float, default=0.0)

    p.add_argument("--loss_run_prefix", "--matrix_run_prefix", dest="loss_run_prefix", default=os.environ.get("LOWVIS_RNN_RUN_PREFIX", ""))
    p.add_argument(
        "--loss_run_prefix_overrides",
        "--matrix_run_prefix_overrides",
        dest="loss_run_prefix_overrides",
        default=os.environ.get("LOSS_RUN_PREFIX_OVERRIDES", ""),
        help=(
            "Optional per-experiment prefix overrides, e.g. "
            "'3=exp_20260609_plain_focal_mist_alpha'. "
            "Use this when one loss checkpoint was trained under a different run prefix."
        ),
    )
    p.add_argument(
        "--experiments",
        "--matrix_experiments",
        dest="experiments",
        default=os.environ.get("LOWVIS_RNN_EXPERIMENTS", "0:1:2"),
        help="Loss experiment ids separated by colon/comma/space.",
    )
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--no_auto_latest", action="store_true")

    p.add_argument("--threshold_source", choices=["checkpoint", "cli", "argmax"], default="checkpoint")
    p.add_argument("--fog_th", type=float, default=0.5)
    p.add_argument("--mist_th", type=float, default=0.5)
    p.add_argument("--rank_metric", default="low_vis_csi")
    p.add_argument("--save_outputs", action="store_true", help="Save per-experiment probs/regression arrays for debugging.")
    return p.parse_args()


def parse_id_list(value: str) -> List[int]:
    out: List[int] = []
    for token in (value or "").replace(",", ":").replace(" ", ":").split(":"):
        token = token.strip()
        if token:
            exp_id = int(token)
            if exp_id not in LOSS_EXPERIMENTS:
                raise ValueError(f"Unknown loss experiment id={exp_id}; valid ids={sorted(LOSS_EXPERIMENTS)}")
            out.append(exp_id)
    return out or sorted(LOSS_EXPERIMENTS)


def parse_prefix_overrides(value: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for token in (value or "").replace(",", " ").replace(";", " ").split():
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Bad --loss_run_prefix_overrides item {token!r}; expected EXP_ID=PREFIX."
            )
        exp_raw, prefix = token.split("=", 1)
        exp_id = int(exp_raw.strip())
        prefix = prefix.strip()
        if exp_id not in LOSS_EXPERIMENTS:
            raise ValueError(f"Unknown loss experiment id={exp_id}; valid ids={sorted(LOSS_EXPERIMENTS)}")
        if not prefix:
            raise ValueError(f"Empty prefix override for loss experiment id={exp_id}.")
        out[exp_id] = prefix
    return out


def checkpoint_suffix(exp_id: int, name: str, stage_tag: str) -> str:
    return f"_{exp_id}_{name}_{stage_tag}_best_score.pt"


def discover_latest_prefix(ckpt_dir: Path, stage_tag: str, exp_ids: Sequence[int]) -> Optional[str]:
    runs: Dict[str, Dict[str, object]] = {}
    for exp_id in exp_ids:
        name = LOSS_EXPERIMENTS[exp_id]
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
    loss_run_prefix: str,
    prefix_overrides: Dict[int, str],
    ckpt_dir: Path,
    stage_tag: str,
    exp_ids: Sequence[int],
    allow_missing: bool,
) -> List[journal.EvalTarget]:
    targets: List[journal.EvalTarget] = []
    spec = journal.VARIANTS[0]
    for exp_id in exp_ids:
        name = LOSS_EXPERIMENTS[exp_id]
        prefix = prefix_overrides.get(exp_id, loss_run_prefix)
        if not prefix:
            raise ValueError(f"No loss_run_prefix is available for experiment {exp_id} {name}.")
        run_id = f"{prefix}_{exp_id}_{name}"
        checkpoint = ckpt_dir / f"{run_id}_{stage_tag}_best_score.pt"
        if not checkpoint.exists():
            if allow_missing:
                print(f"[skip] missing checkpoint: {checkpoint}", flush=True)
                continue
            raise FileNotFoundError(f"Missing checkpoint for loss experiment {exp_id} {name}: {checkpoint}")
        targets.append(journal.EvalTarget(label=name, run_id=run_id, checkpoint=checkpoint, variant=spec))
    return targets


def pred_from_regression_logvis(logvis_pred: np.ndarray) -> np.ndarray:
    logvis = np.asarray(logvis_pred, dtype=np.float64)
    vis = np.expm1(np.clip(logvis, 0.0, np.log1p(80000.0)))
    pred = np.full(len(vis), 2, dtype=np.int64)
    pred[vis < 1000.0] = 1
    pred[vis < 500.0] = 0
    return pred


def run_static_outputs(
    x_path: Path,
    scaler,
    model,
    device,
    batch_size: int,
    layout,
    mod,
    spec: journal.VariantSpec,
    limit_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    X = np.load(x_path, mmap_mode="r")
    n = len(X) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(X))
    probs_out: List[np.ndarray] = []
    reg_out: List[np.ndarray] = []
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = journal.prepare_static_rows(rows, scaler, layout, mod, spec)
        bx = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits, reg = model(bx)
            probs = F.softmax(logits, dim=1)
        probs_out.append(probs.detach().cpu().numpy())
        reg_out.append(reg.detach().cpu().numpy().astype(np.float32))
        if start == 0 or end == n or (start // max(batch_size, 1)) % 20 == 0:
            print(f"  [inference:{spec.name}] {end}/{n}", flush=True)
    probs_arr = np.concatenate(probs_out, axis=0) if probs_out else np.zeros((0, 3), dtype=np.float32)
    reg_arr = np.concatenate(reg_out, axis=0) if reg_out else np.zeros((0,), dtype=np.float32)
    return probs_arr, reg_arr


def objective_from_checkpoint(label: str, ckpt_meta: Dict[str, object]) -> str:
    mode = str(ckpt_meta.get("loss_mode", "") if isinstance(ckpt_meta, dict) else "").strip()
    if mode:
        return mode
    if "regression" in label:
        return "regression"
    if "ce" in label:
        return "ce"
    return "designed_focal"


def confusion_counts(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    valid = (y_true >= 0) & (y_true <= 2) & (pred >= 0) & (pred <= 2)
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y_true[valid], pred[valid]), 1)
    return cm


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
                "display_label": LOSS_LABELS.get(label, label),
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
                    "display_label": LOSS_LABELS.get(label, label),
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


def boundary_metric_rows(
    label: str,
    run_id: str,
    experiment_id: int,
    y_cls: np.ndarray,
    y_raw: np.ndarray,
    pred: np.ndarray,
    probs: Optional[np.ndarray],
    decision_meta: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    y_raw = np.asarray(y_raw, dtype=np.float64)
    for band_id, band_label, lo, hi in BOUNDARY_BANDS:
        mask = np.isfinite(y_raw) & (y_raw >= lo) & (y_raw < hi)
        band_probs = probs[mask] if probs is not None and len(probs) == len(y_cls) else None
        metrics = journal.classification_metrics(y_cls[mask], pred[mask], probs=band_probs)
        rows.append(
            {
                "experiment_id": experiment_id,
                "label": label,
                "display_label": LOSS_LABELS.get(label, label),
                "run_id": run_id,
                "band_id": band_id,
                "band_label": band_label,
                "vis_min_m": lo,
                "vis_max_m": hi,
                "support": int(mask.sum()),
                **decision_meta,
                **metrics,
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
    base: Path,
    ckpt_dir: Path,
    out_dir: Path,
    mod,
    layout,
    device,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    import joblib

    print(f"\n=== Evaluating loss experiment {experiment_id}: {target.label} ===", flush=True)
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

    probs, logvis_pred = run_static_outputs(
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
    objective = objective_from_checkpoint(target.label, ckpt_meta)
    if objective == "regression":
        pred = pred_from_regression_logvis(logvis_pred)
        decision_meta = {
            "threshold_source": "fixed_visibility",
            "decision_type": "regression_threshold",
            "fog_vis_m": 500.0,
            "mist_vis_m": 1000.0,
        }
        metric_probs = None
    else:
        pred, decision_meta = journal.predict_from_probs(args, probs, ckpt_meta)
        decision_meta = {**decision_meta, "decision_type": "probability_threshold"}
        metric_probs = probs

    metrics = journal.classification_metrics(y_cls, pred, probs=metric_probs)
    vis_pred = np.expm1(np.clip(logvis_pred.astype(np.float64), 0.0, np.log1p(80000.0)))
    reg_err = vis_pred - np.maximum(np.asarray(y_raw, dtype=np.float64), 0.0)
    metrics["regression_mae_m"] = float(np.mean(np.abs(reg_err)))
    metrics["regression_rmse_m"] = float(np.sqrt(np.mean(reg_err ** 2)))
    cm = confusion_counts(y_cls, pred)

    if args.save_outputs:
        np.save(out_dir / f"{experiment_id}_{target.label}_probs.npy", probs.astype(np.float32))
        np.save(out_dir / f"{experiment_id}_{target.label}_logvis_pred.npy", logvis_pred.astype(np.float32))

    overall = {
        "experiment_id": experiment_id,
        "label": target.label,
        "display_label": LOSS_LABELS.get(target.label, target.label),
        "objective": objective,
        "run_id": target.run_id,
        "checkpoint": str(target.checkpoint),
        "scaler": str(scaler_path),
        **decision_meta,
        **metrics,
        "ckpt_score": ckpt_meta.get("score") if isinstance(ckpt_meta, dict) else None,
        "ckpt_step": ckpt_meta.get("step") if isinstance(ckpt_meta, dict) else None,
    }
    return (
        overall,
        class_metric_rows(target.label, target.run_id, experiment_id, cm, decision_meta),
        confusion_rows(target.label, target.run_id, experiment_id, cm),
        boundary_metric_rows(target.label, target.run_id, experiment_id, y_cls, y_raw, pred, metric_probs, decision_meta),
    )


def write_markdown_summary(out_dir: Path, overall: pd.DataFrame, per_class: pd.DataFrame, boundary: pd.DataFrame, rank_metric: str) -> None:
    lines = ["# Static-RNN Loss Function Ablation Metrics", ""]
    if rank_metric in overall.columns:
        ranked = overall.sort_values(rank_metric, ascending=False)
        best = ranked.iloc[0]
        lines.append(f"- Best by `{rank_metric}`: `{best['display_label']}` ({float(best[rank_metric]):.6f})")
        lines.append("")
    show_cols = [
        "experiment_id",
        "display_label",
        "objective",
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
    lines.append("## Boundary Bands")
    lines.append("")
    band_cols = ["experiment_id", "display_label", "band_label", "support", "accuracy", "low_vis_recall", "low_vis_precision", "low_vis_csi", "false_positive_rate"]
    lines.append(simple_markdown_table(boundary, band_cols))
    lines.append("")
    (out_dir / "loss_ablation_metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args = journal.apply_paper_eval_config(args, "static_rnn_loss_eval", default_dir=journal.VIS_EVAL_DIR)

    base = Path(args.base).expanduser()
    train_dir = Path(args.train_dir).expanduser()
    data_dir = journal.as_abs_under(base, args.data_dir)
    ckpt_dir = journal.as_abs_under(base, args.ckpt_dir)
    out_dir = journal.unique_dir(journal.as_abs_under(base, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_ids = parse_id_list(args.experiments)
    prefix_overrides = parse_prefix_overrides(args.loss_run_prefix_overrides)
    exp_ids_needing_base_prefix = [exp_id for exp_id in exp_ids if exp_id not in prefix_overrides]
    if exp_ids_needing_base_prefix and not args.loss_run_prefix and not args.no_auto_latest:
        args.loss_run_prefix = discover_latest_prefix(ckpt_dir, args.stage_tag, exp_ids_needing_base_prefix) or ""
        if args.loss_run_prefix:
            print(f"[auto] loss_run_prefix={args.loss_run_prefix}", flush=True)
    if exp_ids_needing_base_prefix and not args.loss_run_prefix:
        raise ValueError("Pass --loss_run_prefix, or leave auto-detect enabled with matching checkpoints in --ckpt_dir.")

    x_path, y_cls, y_raw, _meta = journal.load_main_data(
        data_dir,
        args.limit_samples,
        getattr(args, "meta_time_shift_hours", 0.0),
    )
    dyn, fe = journal.infer_layout_from_x(x_path, args.window_size)
    mod = journal.load_static_rnn_module(train_dir)
    layout = mod.Layout(window_size=args.window_size, dyn_vars=dyn, fe_dim=fe)
    device = journal.resolve_device(args.device)
    targets = build_targets(args.loss_run_prefix, prefix_overrides, ckpt_dir, args.stage_tag, exp_ids, args.allow_missing)
    if not targets:
        raise FileNotFoundError("No loss-ablation checkpoints were selected for evaluation.")

    print("Static-RNN loss-function ablation evaluation", flush=True)
    print(f"base      : {base}", flush=True)
    print(f"train_dir : {train_dir}", flush=True)
    print(f"data_dir  : {data_dir}", flush=True)
    print(f"ckpt_dir  : {ckpt_dir}", flush=True)
    print(f"out_dir   : {out_dir}", flush=True)
    print(f"layout    : {layout}", flush=True)
    print(f"device    : {device}", flush=True)
    if prefix_overrides:
        print(f"prefix overrides: {prefix_overrides}", flush=True)
    print(f"targets   : {[t.run_id for t in targets]}", flush=True)

    overall_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    boundary_rows: List[Dict[str, object]] = []
    id_by_label = {name: exp_id for exp_id, name in LOSS_EXPERIMENTS.items()}
    for target in targets:
        exp_id = id_by_label[target.label]
        overall, per_class, confusion, boundary = evaluate_target(
            args,
            target,
            exp_id,
            x_path,
            y_cls,
            y_raw,
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
        boundary_rows.extend(boundary)

    overall_df = pd.DataFrame(overall_rows).sort_values("experiment_id")
    per_class_df = pd.DataFrame(class_rows).sort_values(["experiment_id", "class_id"])
    cm_df = pd.DataFrame(cm_rows).sort_values(["experiment_id", "true_class_id", "pred_class_id"])
    boundary_df = pd.DataFrame(boundary_rows).sort_values(["experiment_id", "band_id"])

    overall_path = out_dir / "loss_ablation_overall_metrics.csv"
    per_class_path = out_dir / "loss_ablation_per_class_metrics.csv"
    cm_path = out_dir / "loss_ablation_confusion_counts.csv"
    boundary_path = out_dir / "loss_ablation_boundary_metrics.csv"
    overall_df.to_csv(overall_path, index=False, float_format="%.8f")
    per_class_df.to_csv(per_class_path, index=False, float_format="%.8f")
    cm_df.to_csv(cm_path, index=False, float_format="%.8f")
    boundary_df.to_csv(boundary_path, index=False, float_format="%.8f")
    write_markdown_summary(out_dir, overall_df, per_class_df, boundary_df, args.rank_metric)

    run_config = {
        "loss_run_prefix": args.loss_run_prefix,
        "loss_run_prefix_overrides": {str(k): v for k, v in sorted(prefix_overrides.items())},
        "experiments": {str(k): LOSS_EXPERIMENTS[k] for k in exp_ids},
        "labels": LOSS_LABELS,
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
            "boundary_metrics": str(boundary_path),
            "summary_md": str(out_dir / "loss_ablation_metrics_summary.md"),
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[table] {overall_path}", flush=True)
    print(f"[table] {per_class_path}", flush=True)
    print(f"[table] {cm_path}", flush=True)
    print(f"[table] {boundary_path}", flush=True)


if __name__ == "__main__":
    main()
