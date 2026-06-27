#!/usr/bin/env python3
"""Offline evaluation of a frozen candidate gate; never changes deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_static_rnn_lowvis_gate import calibrate_probs, event_masks, metrics, predictions, visibility_to_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--probs-npy", required=True)
    p.add_argument("--labels-npy", required=True)
    p.add_argument("--sample-meta-csv", required=True)
    p.add_argument("--event-summary-csv", required=True)
    p.add_argument("--time-column", default="time")
    p.add_argument("--event-window-hours", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gate_path = Path(args.gate_json)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("selection_source") != "validation_only":
        raise ValueError("Gate artifact is not marked validation_only")
    temperature = float(gate["temperature"])
    threshold = float(gate["low_vis_gate"])
    probs = np.asarray(np.load(args.probs_npy), dtype=np.float64)
    labels = visibility_to_labels(np.load(args.labels_npy))
    meta = pd.read_csv(args.sample_meta_csv)
    events = pd.read_csv(args.event_summary_csv)
    if not (len(probs) == len(labels) == len(meta)):
        raise ValueError("Probability, label, and metadata lengths differ")

    calibrated = calibrate_probs(probs, temperature)
    pred = predictions(calibrated, threshold)
    overall = metrics(labels, pred)
    rows = []
    for event_id, mask in event_masks(meta, events, args.time_column, args.event_window_hours):
        rows.append({"event_id": event_id, **metrics(labels[mask], pred[mask]), "n": int(mask.sum())})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([overall]).to_csv(out_dir / "gate_overall_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame(rows).to_csv(out_dir / "gate_event_metrics.csv", index=False, float_format="%.8f")
    out_sample = meta.copy()
    out_sample["y_true"] = labels
    out_sample["gate_pred"] = pred
    out_sample["calibrated_p_ultra"] = calibrated[:, 0]
    out_sample["calibrated_p_moderate"] = calibrated[:, 1]
    out_sample["calibrated_p_clear"] = calibrated[:, 2]
    out_sample["calibrated_p_lowvis"] = calibrated[:, 0] + calibrated[:, 1]
    out_sample.to_csv(out_dir / "gate_per_sample_eval.csv", index=False, float_format="%.8f")
    run = {
        "experiment_status": "offline_candidate_evaluation",
        "deployment_changed": False,
        "replaces_mainline": False,
        "gate_json": str(gate_path),
        "gate_selection_source": gate["selection_source"],
        "temperature": temperature,
        "low_vis_gate": threshold,
        "test_reselection_performed": False,
        "overall_metrics": overall,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
