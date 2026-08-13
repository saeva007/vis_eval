#!/usr/bin/env python3
"""Redraw the three-panel long-tail training figure from metric tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figure_geometry import GROUPED_BAR_FILL, GROUPED_BAR_TOTAL_WIDTH

import compare_static_rnn_month_group_split as split_plot
import plot_static_rnn_loss_comparison as loss_plot
import plot_static_rnn_sampling_ablation as sampling_plot


FIGURE_WIDTH = 8.20
INK = "#17191B"
GRID = "#E5E7EB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s1-csv", type=Path, required=True, help="Month-group split metric-delta or comparison-summary CSV.")
    p.add_argument("--sampling-csv", type=Path, required=True, help="Complete 0--50% sampling_ablation_overall_metrics.csv.")
    p.add_argument("--loss-csv", type=Path, required=True, help="loss_ablation_overall_metrics.csv.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--figure-stem", default="fig_long_tail_training_composite")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def require_columns(df: pd.DataFrame, names: Iterable[str], path: Path) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def load_s1(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    metrics = [metric for metric, _, _ in split_plot.SKILL_METRICS]
    if "protocol_label" in df.columns:
        require_columns(df, ["protocol_label", *metrics], path)
        labels = df["protocol_label"].astype(str).tolist()
        rows = []
        for metric, display, _ in split_plot.SKILL_METRICS:
            row = {"metric": metric, "display_metric": display}
            for record in df.itertuples(index=False):
                row[str(record.protocol_label)] = float(getattr(record, metric))
            rows.append(row)
        return pd.DataFrame(rows), labels
    require_columns(df, ["metric", "display_metric"], path)
    fixed = {
        "metric",
        "display_metric",
        "delta_month_group_minus_current",
        "preferred_direction",
        "month_group_better",
        "same_test_dataset",
        "comparison_scope",
    }
    labels = [column for column in df.columns if column not in fixed and pd.api.types.is_numeric_dtype(df[column])]
    if len(labels) != 2:
        raise ValueError(f"{path}: expected exactly two protocol value columns, found {labels}")
    return df[df["metric"].astype(str).isin(metrics)].copy(), labels


def draw_s1(ax, source: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    order = [metric for metric, _, _ in split_plot.SKILL_METRICS]
    lookup = source.set_index("metric").reindex(order)
    display = [label for _, label, _ in split_plot.SKILL_METRICS]
    x = np.arange(len(order))
    slot = GROUPED_BAR_TOTAL_WIDTH / 2.0
    width = slot * GROUPED_BAR_FILL
    offsets = [-slot / 2, slot / 2]
    for offset, label in zip(offsets, labels):
        values = pd.to_numeric(lookup[label], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offset, values, width=width, color=split_plot.label_color(label), label=label)
    tick_labels = [
        text.replace("Ultra-low ", "Ultra-\nlow ")
        .replace("Moderate-low ", "Moderate-\nlow ")
        .replace("Low-vis event ", "Low-vis\n")
        for text in display
    ]
    ax.set_xticks(x, tick_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=6.8, pad=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Pretraining improves skill", loc="left", fontweight="bold", pad=5)
    ax.grid(False)
    ax.legend(loc="upper left", handlelength=1.3)
    return source


def draw_sampling(ax, overall: pd.DataFrame) -> pd.DataFrame:
    curve = sampling_plot.sampling_curve_table(overall)
    if curve["target_lowvis_pct"].nunique() != 11:
        present = sorted(curve["target_lowvis_pct"].unique().tolist())
        raise ValueError(f"Sampling figure requires all 11 points from 0% to 50% at 5% steps; found {present}")
    x = curve["target_lowvis_pct"].to_numpy(dtype=float)
    for metric, label in (("low_vis_csi", "CSI"), ("low_vis_recall", "Recall"), ("low_vis_precision", "Precision")):
        ax.plot(
            x,
            curve[metric].to_numpy(dtype=float),
            color=sampling_plot.CURVE_COLORS[metric],
            linewidth=1.8,
            marker="o",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=label,
        )
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlim(-2, 52)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Low-vis target share (%)")
    ax.set_ylabel("Score")
    ax.set_title("Sampling response", loc="left", fontweight="bold", pad=5)
    ax.grid(False)
    ax.legend(loc="best", handlelength=1.5)
    return curve


def prepare_loss(overall: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    source = overall.copy()
    source["display_label"] = source.apply(loss_plot.display_label, axis=1)
    labels = loss_plot.ordered_labels(source)
    return source, labels


def draw_loss(ax, overall: pd.DataFrame) -> pd.DataFrame:
    source, labels = prepare_loss(overall)
    loss_plot.panel_lowvis_event_skill(ax, source, labels)
    # The helper writes a centred title. Clear it before adding the composite's
    # left-aligned title so both Matplotlib title slots cannot overlap.
    ax.set_title("", loc="center")
    ax.set_title("Loss-function comparison", loc="left", fontweight="bold", pad=5)
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), handlelength=1.3, fontsize=6.8)
    return source


def panel_label(ax, letter: str) -> None:
    ax.text(-0.16, 1.07, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)


def export(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"dpi": dpi} if ext in {"png", "tiff"} else {}
        fig.savefig(out_dir / f"{stem}.{ext}", facecolor="white", **kwargs)


def main() -> None:
    args = parse_args()
    setup_style()
    s1_path = args.s1_csv.expanduser().resolve()
    sampling_path = args.sampling_csv.expanduser().resolve()
    loss_path = args.loss_csv.expanduser().resolve()
    for path in (s1_path, sampling_path, loss_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    s1, labels = load_s1(s1_path)
    sampling = pd.read_csv(sampling_path)
    loss = pd.read_csv(loss_path)
    require_columns(sampling, ["low_vis_csi", "low_vis_recall", "low_vis_precision"], sampling_path)
    require_columns(loss, ["label", "low_vis_csi", "low_vis_recall", "low_vis_precision"], loss_path)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.85))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.18, 1.04, 1.05],
        left=0.06,
        right=0.992,
        top=0.88,
        bottom=0.22,
        wspace=0.34,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    source_a = draw_s1(axes[0], s1, labels)
    source_b = draw_sampling(axes[1], sampling)
    source_c = draw_loss(axes[2], loss)
    for letter, ax in zip("abc", axes):
        panel_label(ax, letter)

    out_dir = args.out_dir.expanduser().resolve()
    export(fig, out_dir, args.figure_stem, args.dpi)
    plt.close(fig)
    source_a.assign(panel="a_pretraining").to_csv(out_dir / f"{args.figure_stem}_pretraining_source.csv", index=False, float_format="%.8f")
    source_b.assign(panel="b_sampling").to_csv(out_dir / f"{args.figure_stem}_sampling_source.csv", index=False, float_format="%.8f")
    source_c.assign(panel="c_loss").to_csv(out_dir / f"{args.figure_stem}_loss_source.csv", index=False, float_format="%.8f")
    manifest = {
        "s1_csv": str(s1_path),
        "sampling_csv": str(sampling_path),
        "loss_csv": str(loss_path),
        "figure": str(out_dir / f"{args.figure_stem}.pdf"),
        "rendering": "all panels redrawn from metric tables on one canvas",
    }
    (out_dir / f"{args.figure_stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_dir / f"{args.figure_stem}.png")


if __name__ == "__main__":
    main()
