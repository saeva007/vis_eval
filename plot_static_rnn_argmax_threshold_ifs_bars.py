#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot PMST argmax, PMST threshold, and IFS diagnostic bar comparison.

Inputs are the two ``overall_metrics.csv`` files written by
``run_static_rnn_lowvis_eval_journal.py``:

* an argmax run with ``--threshold_source argmax``;
* the default checkpoint-threshold run.

The script extracts only the IFS-matched test rows so the three bars are
computed on the same station-time sample set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "PMST argmax",
    "PMST threshold",
    "IFS diagnostic",
]

METHOD_COLORS = {
    "PMST argmax": "#4C78A8",
    "PMST threshold": "#2A9D8F",
    "IFS diagnostic": "#8A8F98",
}

PANELS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
    ("Fog", (("Fog_P", "Precision"), ("Fog_R", "Recall"), ("Fog_CSI", "CSI"))),
    ("Mist", (("Mist_P", "Precision"), ("Mist_R", "Recall"), ("Mist_CSI", "CSI"))),
    (
        "Low visibility",
        (("low_vis_precision", "Precision"), ("low_vis_recall", "Recall"), ("low_vis_csi", "CSI")),
    ),
    ("Overall / false alarms", (("accuracy", "Accuracy"), ("false_positive_rate", "Clear FPR"))),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Draw grouped bars for PMST argmax, PMST threshold, and IFS diagnostic metrics."
    )
    p.add_argument("--argmax_csv", required=True, help="overall_metrics.csv from the argmax evaluation run.")
    p.add_argument("--threshold_csv", required=True, help="overall_metrics.csv from the checkpoint-threshold run.")
    p.add_argument("--out_dir", default="", help="Output directory; default is beside --threshold_csv.")
    p.add_argument("--figure_stem", default="fig_static_rnn_argmax_threshold_ifs_bars")
    p.add_argument("--title", default="Decision-rule comparison on IFS-matched test samples")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def _pick_row(df: pd.DataFrame, source: str, sample_scope: str = "ifs_diagnostic_matched_test") -> pd.Series:
    sub = df.copy()
    if "source" in sub.columns:
        sub = sub[sub["source"].astype(str) == source]
    if "sample_scope" in sub.columns:
        scoped = sub[sub["sample_scope"].astype(str) == sample_scope]
        if not scoped.empty:
            sub = scoped
    if sub.empty:
        raise ValueError(f"No row found for source={source!r}, sample_scope={sample_scope!r}")
    return sub.iloc[0]


def load_comparison(argmax_csv: Path, threshold_csv: Path) -> pd.DataFrame:
    argmax_df = pd.read_csv(argmax_csv)
    threshold_df = pd.read_csv(threshold_csv)
    rows = []
    specs = [
        ("PMST argmax", _pick_row(argmax_df, "pmst")),
        ("PMST threshold", _pick_row(threshold_df, "pmst")),
        ("IFS diagnostic", _pick_row(threshold_df, "ifs_diagnostic")),
    ]
    for method, row in specs:
        rec: Dict[str, object] = {
            "method": method,
            "source_csv": str(argmax_csv if method == "PMST argmax" else threshold_csv),
            "source": row.get("source", ""),
            "sample_scope": row.get("sample_scope", ""),
            "threshold_source": row.get("threshold_source", ""),
            "threshold_rule": row.get("threshold_rule", ""),
            "fog_th": row.get("fog_th", np.nan),
            "mist_th": row.get("mist_th", np.nan),
            "n": row.get("n", np.nan),
            "matched_rows": row.get("matched_rows", np.nan),
        }
        for _, metrics in PANELS:
            for metric, _label in metrics:
                rec[metric] = row.get(metric, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def plot_panel(ax, data: pd.DataFrame, metrics: Sequence[Tuple[str, str]], title: str) -> None:
    x = np.arange(len(metrics), dtype=float)
    methods = METHOD_ORDER
    width = 0.23
    offsets = np.linspace(-width, width, len(methods))
    for offset, method in zip(offsets, methods):
        row = data[data["method"] == method]
        values = [float(row.iloc[0][metric]) if not row.empty and metric in row else np.nan for metric, _ in metrics]
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=method,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.016,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.1,
                    rotation=0,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.set_title(title, pad=8)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.65)
    if title == "Overall / false alarms":
        ax.text(
            1.0,
            0.98,
            "Clear FPR: lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#555555",
        )


def save_outputs(fig, out_dir: Path, stem: str, dpi: int, source_csvs: Sequence[Path], source_data: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = out_dir / f"{stem}_source_data.csv"
    source_data.to_csv(source_path, index=False, float_format="%.8f")
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    manifest = pd.DataFrame(
        [
            {
                "figure": f"{stem}.png/pdf/svg",
                "source_data": str(source_path),
                "source_csvs": ";".join(str(p) for p in source_csvs),
                "notes": "IFS-matched test-set grouped bars for PMST argmax, PMST checkpoint-threshold, and IFS diagnostic.",
            }
        ]
    )
    manifest.to_csv(out_dir / f"{stem}_source_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    setup_style()
    argmax_csv = Path(args.argmax_csv).expanduser()
    threshold_csv = Path(args.threshold_csv).expanduser()
    for path in (argmax_csv, threshold_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else threshold_csv.parent
    data = load_comparison(argmax_csv, threshold_csv)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    for ax, panel_label, (title, metrics) in zip(axes.ravel(), ("a", "b", "c", "d"), PANELS):
        plot_panel(ax, data, metrics, title)
        add_panel_label(ax, panel_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.025))
    if args.title:
        fig.suptitle(args.title, y=1.06, fontsize=12.5)

    save_outputs(fig, out_dir, args.figure_stem, args.dpi, [argmax_csv, threshold_csv], data)


if __name__ == "__main__":
    main()
