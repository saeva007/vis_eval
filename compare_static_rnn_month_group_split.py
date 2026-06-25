#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the default Static-RNN model with a month-group split retraining.

This script compares *official test-set evaluations* from two protocols.  It
does not present the old default checkpoint on the new month-held-out test as a
clean head-to-head result, because the default month-tail training set contains
samples from every calendar month and would contaminate a whole-month holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SKILL_METRICS = [
    ("Fog_CSI", "Ultra-low CSI", "higher"),
    ("Mist_CSI", "Moderate-low CSI", "higher"),
    ("low_vis_csi", "Low-vis event CSI", "higher"),
    ("low_vis_recall", "Low-vis event recall", "higher"),
    ("low_vis_precision", "Low-vis event precision", "higher"),
]

FPR_METRIC = ("false_positive_rate", "Clear FPR", "lower")
KEY_METRICS = [*SKILL_METRICS, FPR_METRIC]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Static-RNN default and month-group split evaluation outputs.")
    p.add_argument("--current_eval_dir", required=True, help="Default/current main-model evaluation directory.")
    p.add_argument("--month_group_eval_dir", required=True, help="Month-group retrained model evaluation directory.")
    p.add_argument("--out_dir", default="", help="Output directory; default is the month-group eval directory.")
    p.add_argument("--current_label", default="Current month-tail")
    p.add_argument("--month_group_label", default="Month-group retrain")
    p.add_argument("--figure_stem", default="fig_static_rnn_month_group_split_comparison")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_model_row(overall_path: Path) -> pd.Series:
    df = pd.read_csv(overall_path)
    if "source" in df.columns:
        df = df[df["source"].astype(str) == "pmst"]
    if "sample_scope" in df.columns:
        exact = df[df["sample_scope"].astype(str) == "test"]
        if not exact.empty:
            df = exact
    if df.empty:
        raise ValueError(f"No PMST test row found in {overall_path}")
    return df.iloc[0]


def dataset_split_metadata(eval_config: Dict[str, object]) -> Dict[str, object]:
    data_dir = eval_config.get("data_dir")
    if not data_dir:
        return {}
    data_path = Path(str(data_dir))
    cfg = load_json(data_path / "dataset_split_config.json")
    return cfg


def row_from_eval(eval_dir: Path, label: str) -> Dict[str, object]:
    overall_path = eval_dir / "overall_metrics.csv"
    run_config_path = eval_dir / "run_config.json"
    if not overall_path.exists():
        raise FileNotFoundError(f"Missing overall metrics: {overall_path}")
    row = select_model_row(overall_path)
    run_config = load_json(run_config_path)
    split_config = dataset_split_metadata(run_config)
    out = {
        "protocol_label": label,
        "eval_dir": str(eval_dir),
        "run_id": run_config.get("run_id", row.get("run_id", "")),
        "checkpoint": run_config.get("checkpoint", ""),
        "data_dir": run_config.get("data_dir", ""),
        "split_policy": split_config.get("split_policy", "unknown"),
        "split_name": split_config.get("split_name", ""),
        "n": row.get("n", np.nan),
    }
    for metric, _, _ in KEY_METRICS:
        out[metric] = row.get(metric, np.nan)
    return out


def build_delta_rows(summary: pd.DataFrame, current_label: str, month_group_label: str) -> pd.DataFrame:
    cur = summary[summary["protocol_label"] == current_label].iloc[0]
    alt = summary[summary["protocol_label"] == month_group_label].iloc[0]
    rows: List[Dict[str, object]] = []
    same_data_dir = str(cur.get("data_dir", "")) == str(alt.get("data_dir", ""))
    for metric, display, direction in KEY_METRICS:
        cur_v = float(cur.get(metric, np.nan))
        alt_v = float(alt.get(metric, np.nan))
        delta = alt_v - cur_v
        alt_better = delta < 0 if direction == "lower" else delta > 0
        rows.append(
            {
                "metric": metric,
                "display_metric": display,
                current_label: cur_v,
                month_group_label: alt_v,
                "delta_month_group_minus_current": delta,
                "preferred_direction": direction,
                "month_group_better": bool(alt_better) if np.isfinite(delta) else False,
                "same_test_dataset": bool(same_data_dir),
                "comparison_scope": "same-test" if same_data_dir else "protocol-level official-test comparison",
            }
        )
    return pd.DataFrame(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )


def plot_comparison(delta: pd.DataFrame, out_dir: Path, stem: str, current_label: str, month_group_label: str) -> None:
    setup_style()
    skill_names = {metric for metric, _, _ in SKILL_METRICS}
    skill_delta = delta[delta["metric"].astype(str).isin(skill_names)].copy()
    metrics = skill_delta["display_metric"].tolist()
    cur_vals = skill_delta[current_label].astype(float).to_numpy()
    alt_vals = skill_delta[month_group_label].astype(float).to_numpy()
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - width / 2, cur_vals, width=width, color="#8A8F98", label=current_label)
    ax.bar(x + width / 2, alt_vals, width=width, color="#2A9D8F", label=month_group_label)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, max(1.0, float(np.nanmax([cur_vals, alt_vals])) * 1.12))
    ax.set_title("Static-RNN split-protocol skill metrics")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.legend(loc="upper left", frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    plt.close(fig)

    fpr_delta = delta[delta["metric"].astype(str) == "false_positive_rate"]
    if not fpr_delta.empty:
        cur_fpr = float(fpr_delta.iloc[0][current_label])
        alt_fpr = float(fpr_delta.iloc[0][month_group_label])
        values = np.asarray([cur_fpr, alt_fpr], dtype=float)
        fig, ax = plt.subplots(figsize=(3.7, 3.05))
        ax.bar(
            np.arange(2),
            values,
            width=0.55,
            color=["#8A8F98", "#2A9D8F"],
            edgecolor="white",
            linewidth=0.45,
        )
        ymax = max(0.04, float(np.nanmax(values)) * 1.35) if np.isfinite(values).any() else 0.1
        for xi, value in enumerate(values):
            if np.isfinite(value):
                ax.text(xi, value + max(0.002, ymax * 0.025), f"{value:.3f}", ha="center", va="bottom", fontsize=8.6)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels([current_label, month_group_label], rotation=20, ha="right")
        ax.set_ylim(0, min(1.0, ymax))
        ax.set_ylabel("False-positive rate")
        ax.set_title("Clear-condition false-positive rate")
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.text(0.98, 0.96, "lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=8.4, color="#555555")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            path = out_dir / f"{stem}_clear_fpr.{ext}"
            fig.savefig(path, bbox_inches="tight")
            print(f"[figure] {path}", flush=True)
        plt.close(fig)


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, summary: pd.DataFrame, delta: pd.DataFrame, current_label: str, month_group_label: str) -> None:
    same_test = bool(delta["same_test_dataset"].iloc[0]) if not delta.empty else False
    lines = ["# Static-RNN Month-Group Split Comparison", ""]
    if same_test:
        lines.append("- Comparison scope: same test dataset.")
    else:
        lines.append(
            "- Comparison scope: protocol-level official-test comparison. "
            "This is leakage-safe because each model is evaluated on its own official test split, "
            "but it is not a same-test head-to-head score."
        )
        lines.append(
            "- Do not use the old month-tail checkpoint on the month-held-out test as the headline result: "
            "the month-tail training split includes samples from every calendar month."
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    show_cols = ["protocol_label", "run_id", "data_dir", "split_policy", "split_name", "n"] + [m for m, _, _ in KEY_METRICS]
    lines.append(simple_markdown_table(summary[[c for c in show_cols if c in summary.columns]]))
    lines.append("")
    lines.append("## Metric Deltas")
    lines.append("")
    delta_cols = [
        "display_metric",
        current_label,
        month_group_label,
        "delta_month_group_minus_current",
        "preferred_direction",
        "month_group_better",
        "comparison_scope",
    ]
    lines.append(simple_markdown_table(delta[[c for c in delta_cols if c in delta.columns]]))
    lines.append("")
    (out_dir / "static_rnn_month_group_split_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    current_eval_dir = Path(args.current_eval_dir).expanduser()
    month_eval_dir = Path(args.month_group_eval_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else month_eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            row_from_eval(current_eval_dir, args.current_label),
            row_from_eval(month_eval_dir, args.month_group_label),
        ]
    )
    delta = build_delta_rows(summary, args.current_label, args.month_group_label)
    summary_path = out_dir / "static_rnn_month_group_split_comparison_summary.csv"
    delta_path = out_dir / "static_rnn_month_group_split_metric_deltas.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    delta.to_csv(delta_path, index=False, float_format="%.8f")
    write_report(out_dir, summary, delta, args.current_label, args.month_group_label)
    if not args.no_plot:
        plot_comparison(delta, out_dir, args.figure_stem, args.current_label, args.month_group_label)
    print(f"[table] {summary_path}", flush=True)
    print(f"[table] {delta_path}", flush=True)


if __name__ == "__main__":
    main()
