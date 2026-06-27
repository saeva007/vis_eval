#!/usr/bin/env python3
"""Fit temperature and a validation-only hierarchical Low-vis event gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probs-npy", required=True)
    p.add_argument("--labels-npy", required=True)
    p.add_argument("--sample-meta-csv", required=True)
    p.add_argument("--event-summary-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-version", required=True)
    p.add_argument("--time-column", default="time")
    p.add_argument("--event-window-hours", type=int, default=3)
    p.add_argument("--min-event-recall", type=float, default=0.60)
    p.add_argument("--min-ultra-recall", type=float, default=0.55)
    p.add_argument("--min-moderate-recall", type=float, default=0.30)
    p.add_argument("--min-moderate-csi", type=float, default=0.09)
    p.add_argument("--target-max-fpr", type=float, default=0.035)
    p.add_argument("--gate-low", type=float, default=0.05)
    p.add_argument("--gate-high", type=float, default=0.95)
    p.add_argument("--gate-step", type=float, default=0.0025)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-curve-csv", default="")
    return p.parse_args()


def visibility_to_labels(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if np.isfinite(arr).all() and np.all((arr >= 0) & (arr <= 2)) and np.allclose(arr, np.round(arr)):
        return arr.astype(np.int64)
    out = np.full(len(arr), 2, dtype=np.int64)
    out[arr < 1000.0] = 1
    out[arr < 500.0] = 0
    return out


def softmax(values: np.ndarray) -> np.ndarray:
    z = values - np.max(values, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def calibrate_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-9, 1.0)) / max(float(temperature), 1e-6)
    return softmax(logits)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1.0))))


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    before = nll(probs, labels)
    temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    losses = np.asarray([nll(calibrate_probs(probs, t), labels) for t in temperatures])
    best = int(np.nanargmin(losses))
    return float(temperatures[best]), before, float(losses[best])


def predictions(probs: np.ndarray, gate: float) -> np.ndarray:
    p_low = probs[:, 0] + probs[:, 1]
    pred = np.full(len(probs), 2, dtype=np.int64)
    passed = p_low >= float(gate)
    pred[passed] = np.argmax(probs[passed, :2], axis=1)
    return pred


def metrics(labels: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    low_true = labels <= 1
    low_pred = pred <= 1
    tp = int(np.sum(low_true & low_pred))
    fp = int(np.sum(~low_true & low_pred))
    fn = int(np.sum(low_true & ~low_pred))
    tn = int(np.sum(~low_true & ~low_pred))

    def class_metrics(cls: int) -> Tuple[float, float, float]:
        ctp = int(np.sum((labels == cls) & (pred == cls)))
        cfp = int(np.sum((labels != cls) & (pred == cls)))
        cfn = int(np.sum((labels == cls) & (pred != cls)))
        return (
            ctp / max(ctp + cfp, 1),
            ctp / max(ctp + cfn, 1),
            ctp / max(ctp + cfp + cfn, 1),
        )

    fog_p, fog_r, fog_csi = class_metrics(0)
    mist_p, mist_r, mist_csi = class_metrics(1)
    return {
        "low_vis_precision": tp / max(tp + fp, 1),
        "low_vis_far": fp / max(tp + fp, 1),
        "low_vis_recall": tp / max(tp + fn, 1),
        "low_vis_csi": tp / max(tp + fp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "low_vis_area_ratio": int(np.sum(low_pred)) / max(int(np.sum(low_true)), 1),
        "Fog_P": fog_p,
        "Fog_R": fog_r,
        "Fog_CSI": fog_csi,
        "Mist_P": mist_p,
        "Mist_R": mist_r,
        "Mist_CSI": mist_csi,
        "clear_to_ultra_fp": int(np.sum((labels == 2) & (pred == 0))),
        "clear_to_moderate_fp": int(np.sum((labels == 2) & (pred == 1))),
    }


def event_masks(meta: pd.DataFrame, events: pd.DataFrame, time_col: str, window_hours: int) -> List[Tuple[str, np.ndarray]]:
    if time_col not in meta:
        for fallback in ("time_utc", "time_analysis"):
            if fallback in meta:
                time_col = fallback
                break
    if time_col not in meta:
        raise KeyError(f"No usable time column in sample metadata: requested {time_col!r}")
    times = pd.to_datetime(meta[time_col], errors="coerce")
    out: List[Tuple[str, np.ndarray]] = []
    for idx, row in events.iterrows():
        event_id = str(row.get("event_rank", row.get("event_id", idx + 1)))
        peak = pd.Timestamp(row["peak_time"])
        start = peak - pd.Timedelta(hours=window_hours)
        end = peak + pd.Timedelta(hours=window_hours)
        mask = ((times >= start) & (times <= end)).to_numpy()
        if int(mask.sum()) == 0:
            raise ValueError(f"Validation event {event_id} has no matched samples in {start}..{end}")
        out.append((event_id, mask))
    if not out:
        raise ValueError("No validation events were supplied")
    return out


def main() -> None:
    args = parse_args()
    probs = np.asarray(np.load(args.probs_npy), dtype=np.float64)
    labels = visibility_to_labels(np.load(args.labels_npy))
    meta = pd.read_csv(args.sample_meta_csv)
    events = pd.read_csv(args.event_summary_csv)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected probability array [N,3], got {probs.shape}")
    if not (len(probs) == len(labels) == len(meta)):
        raise ValueError(f"Length mismatch: probs={len(probs)} labels={len(labels)} meta={len(meta)}")

    temperature, nll_before, nll_after = fit_temperature(probs, labels)
    calibrated = calibrate_probs(probs, temperature)
    masks = event_masks(meta, events, args.time_column, args.event_window_hours)
    gates = np.arange(args.gate_low, args.gate_high + 1e-12, args.gate_step)
    rows: List[Dict[str, object]] = []
    for gate in gates:
        pred = predictions(calibrated, float(gate))
        row: Dict[str, object] = {"low_vis_gate": float(gate), **metrics(labels, pred)}
        recalls = []
        for event_id, mask in masks:
            event_metrics = metrics(labels[mask], pred[mask])
            recall = float(event_metrics["low_vis_recall"])
            row[f"event_{event_id}_low_vis_recall"] = recall
            recalls.append(recall)
        row["min_event_low_vis_recall"] = float(np.min(recalls))
        row["passes_constraints"] = bool(
            row["min_event_low_vis_recall"] >= args.min_event_recall
            and row["Fog_R"] >= args.min_ultra_recall
            and row["Mist_R"] >= args.min_moderate_recall
            and row["Mist_CSI"] >= args.min_moderate_csi
        )
        row["meets_target_fpr"] = bool(row["false_positive_rate"] <= args.target_max_fpr)
        rows.append(row)

    curve = pd.DataFrame(rows)
    feasible = curve[curve["passes_constraints"]].copy()
    if feasible.empty:
        raise RuntimeError("No gate satisfies all recall/Moderate-low constraints on validation data")
    feasible = feasible.sort_values(
        ["false_positive_rate", "low_vis_csi", "Mist_CSI", "low_vis_gate"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    best = feasible.iloc[0].to_dict()

    out_json = Path(args.out_json)
    out_curve = Path(args.out_curve_csv) if args.out_curve_csv else out_json.with_name(out_json.stem + "_curve.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_curve, index=False, float_format="%.8f")
    payload = {
        "schema_version": 1,
        "experiment_status": "offline_candidate_gate",
        "deployment_approved": False,
        "replaces_mainline": False,
        "selection_source": "validation_only",
        "decision_rule": "temperature_scaled_hierarchical_lowvis_gate",
        "checkpoint": str(args.checkpoint),
        "data_version": str(args.data_version),
        "temperature": temperature,
        "low_vis_gate": float(best["low_vis_gate"]),
        "severity_rule": "argmax_over_ultra_and_moderate_after_gate",
        "constraints": {
            "target_max_fpr": args.target_max_fpr,
            "min_event_recall": args.min_event_recall,
            "min_ultra_recall": args.min_ultra_recall,
            "min_moderate_recall": args.min_moderate_recall,
            "min_moderate_csi": args.min_moderate_csi,
        },
        "validation_metrics": {k: v for k, v in best.items() if k != "passes_constraints"},
        "target_max_fpr_met": bool(best["false_positive_rate"] <= args.target_max_fpr),
        "calibration": {"nll_before": nll_before, "nll_after": nll_after},
        "event_summary_csv": str(args.event_summary_csv),
        "curve_csv": str(out_curve),
        "test_reselection_forbidden": True,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not payload["target_max_fpr_met"]:
        print(
            f"[warn] Best recall-feasible gate FPR={best['false_positive_rate']:.6f} "
            f"does not reach target {args.target_max_fpr:.6f}; threshold was not relaxed using test data."
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[table] {out_curve}")
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
