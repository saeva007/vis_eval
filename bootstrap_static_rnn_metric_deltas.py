#!/usr/bin/env python3
"""Paired date-by-station block bootstrap for Static-RNN metric deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


METRIC_NAMES = (
    "false_positive_rate",
    "low_vis_precision",
    "low_vis_recall",
    "low_vis_csi",
    "Fog_R",
    "Mist_R",
    "Mist_CSI",
    "clear_to_ultra_fp_rate",
    "clear_to_moderate_fp_rate",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--match-columns", default="station_id,time")
    p.add_argument("--station-column", default="station_id")
    p.add_argument("--time-column", default="time")
    p.add_argument("--event-summary-csv", default="")
    p.add_argument("--event-window-hours", type=int, default=3)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    low_true = y <= 1
    low_pred = pred <= 1
    tp = int(np.sum(low_true & low_pred))
    fp = int(np.sum(~low_true & low_pred))
    fn = int(np.sum(low_true & ~low_pred))
    tn = int(np.sum(~low_true & ~low_pred))

    def recall(cls: int) -> float:
        support = int(np.sum(y == cls))
        return int(np.sum((y == cls) & (pred == cls))) / max(support, 1)

    def csi(cls: int) -> float:
        ctp = int(np.sum((y == cls) & (pred == cls)))
        cfp = int(np.sum((y != cls) & (pred == cls)))
        cfn = int(np.sum((y == cls) & (pred != cls)))
        return ctp / max(ctp + cfp + cfn, 1)

    clear_n = int(np.sum(y == 2))
    return {
        "false_positive_rate": fp / max(fp + tn, 1),
        "low_vis_precision": tp / max(tp + fp, 1),
        "low_vis_recall": tp / max(tp + fn, 1),
        "low_vis_csi": tp / max(tp + fp + fn, 1),
        "Fog_R": recall(0),
        "Mist_R": recall(1),
        "Mist_CSI": csi(1),
        "clear_to_ultra_fp_rate": int(np.sum((y == 2) & (pred == 0))) / max(clear_n, 1),
        "clear_to_moderate_fp_rate": int(np.sum((y == 2) & (pred == 1))) / max(clear_n, 1),
    }


def paired_data(args: argparse.Namespace) -> pd.DataFrame:
    baseline = pd.read_csv(args.baseline_csv)
    candidate = pd.read_csv(args.candidate_csv)
    keys = [x.strip() for x in args.match_columns.split(",") if x.strip()]
    for key in keys:
        if key not in baseline or key not in candidate:
            raise KeyError(f"Match column {key!r} is missing")
    required = {"y_true", "pmst_pred"}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        raise KeyError("Both inputs need y_true and pmst_pred")
    left = baseline[keys + ["y_true", "pmst_pred"]].rename(columns={"pmst_pred": "baseline_pred"})
    right = candidate[keys + ["y_true", "pmst_pred"]].rename(columns={"y_true": "candidate_y", "pmst_pred": "candidate_pred"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.array_equal(merged["y_true"].to_numpy(), merged["candidate_y"].to_numpy()):
        raise ValueError("Matched baseline and candidate labels differ")
    merged[args.time_column] = pd.to_datetime(merged[args.time_column], errors="coerce")
    merged["__block"] = (
        merged[args.time_column].dt.strftime("%Y-%m-%d").fillna("unknown")
        + "|"
        + merged[args.station_column].astype(str)
    )
    return merged


def contexts(df: pd.DataFrame, args: argparse.Namespace) -> List[Tuple[str, pd.DataFrame]]:
    out = [("overall", df)]
    if not args.event_summary_csv:
        return out
    events = pd.read_csv(args.event_summary_csv)
    for idx, row in events.iterrows():
        event_id = str(row.get("event_rank", row.get("event_id", idx + 1)))
        peak = pd.Timestamp(row["peak_time"])
        start = peak - pd.Timedelta(hours=args.event_window_hours)
        end = peak + pd.Timedelta(hours=args.event_window_hours)
        out.append((f"event_{event_id}", df[(df[args.time_column] >= start) & (df[args.time_column] <= end)]))
    return out


def bootstrap_context(name: str, df: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> List[Dict[str, object]]:
    if df.empty:
        return []
    df = df.reset_index(drop=True)
    groups = {key: idx.to_numpy() for key, idx in df.groupby("__block").groups.items()}
    block_keys = np.asarray(list(groups), dtype=object)
    y = df["y_true"].to_numpy(dtype=np.int64)
    base = df["baseline_pred"].to_numpy(dtype=np.int64)
    cand = df["candidate_pred"].to_numpy(dtype=np.int64)
    point_base = metrics(y, base)
    point_cand = metrics(y, cand)
    samples = {metric: [] for metric in METRIC_NAMES}
    for _ in range(args.n_bootstrap):
        chosen = rng.choice(block_keys, size=len(block_keys), replace=True)
        indices = np.concatenate([groups[key] for key in chosen])
        mb = metrics(y[indices], base[indices])
        mc = metrics(y[indices], cand[indices])
        for metric in METRIC_NAMES:
            samples[metric].append(mc[metric] - mb[metric])
    rows: List[Dict[str, object]] = []
    for metric in METRIC_NAMES:
        delta = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "context": name,
                "metric": metric,
                "baseline": point_base[metric],
                "candidate": point_cand[metric],
                "delta_candidate_minus_baseline": point_cand[metric] - point_base[metric],
                "ci95_low": float(np.nanpercentile(delta, 2.5)),
                "ci95_high": float(np.nanpercentile(delta, 97.5)),
                "n_rows": int(len(df)),
                "n_blocks": int(len(block_keys)),
                "n_bootstrap": int(args.n_bootstrap),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    df = paired_data(args)
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    for name, subset in contexts(df, args):
        rows.extend(bootstrap_context(name, subset, args, rng))
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.8f")
    print(f"[table] {out}")


if __name__ == "__main__":
    main()
