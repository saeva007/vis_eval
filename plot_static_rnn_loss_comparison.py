#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a journal-ready figure for the Static-RNN loss-function ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS: Dict[str, str] = {
    "simple_ce_classification": "CE classification",
    "simple_logvis_regression": "MSE regression",
    "proposed_rare_event_focal": "Proposed focal",
    "plain_focal_loss": "Plain focal",
}

METHOD_COLORS: Dict[str, str] = {
    "CE classification": "#8A8F98",
    "MSE regression": "#D09A3A",
    "Proposed focal": "#2A9D8F",
    "Plain focal": "#6C5CE7",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Static-RNN loss-function ablation figure.")
    p.add_argument("--eval_dir", required=True, help="Directory produced by run_static_rnn_lowvis_loss_eval.py.")
    p.add_argument("--out_dir", default="", help="Output directory; default is --eval_dir.")
    p.add_argument("--figure_stem", default="fig_static_rnn_loss_function_ablation")
    p.add_argument("--title", default="")
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )


def display_label(row: pd.Series) -> str:
    label = str(row.get("label", ""))
    return str(row.get("display_label") or METHOD_LABELS.get(label, label))


def ordered_labels(overall: pd.DataFrame) -> List[str]:
    df = overall.copy()
    if "display_label" not in df.columns:
        df["display_label"] = df.apply(display_label, axis=1)
    if "experiment_id" in df.columns:
        df = df.sort_values("experiment_id")
    return [str(v) for v in df["display_label"].tolist()]


def metric_values(overall: pd.DataFrame, labels: Sequence[str], metric: str) -> List[float]:
    lookup = overall.set_index("display_label")
    vals = []
    for label in labels:
        value = lookup.loc[label, metric] if metric in lookup.columns and label in lookup.index else np.nan
        vals.append(float(value) if pd.notna(value) else np.nan)
    return vals


def proposed_soft_targets(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fog = np.zeros_like(x, dtype=float)
    mist = np.zeros_like(x, dtype=float)
    clear = np.zeros_like(x, dtype=float)

    fog[x < 400.0] = 1.0
    m = (x >= 400.0) & (x < 600.0)
    fog[m] = 1.0 - (x[m] - 400.0) / 200.0
    mist[m] = (x[m] - 400.0) / 200.0

    m = (x >= 600.0) & (x < 800.0)
    mist[m] = 1.0
    m = (x >= 800.0) & (x < 1200.0)
    mist[m] = 1.0 - (x[m] - 800.0) / 400.0
    clear[m] = (x[m] - 800.0) / 400.0
    clear[x >= 1200.0] = 1.0
    return fog, mist, clear


def panel_soft_targets(ax) -> None:
    x = np.linspace(0.0, 1600.0, 500)
    fog, mist, clear = proposed_soft_targets(x)
    ax.axvspan(400, 600, color="#D7DEE8", alpha=0.45, lw=0)
    ax.axvspan(800, 1200, color="#D7DEE8", alpha=0.30, lw=0)
    ax.plot(x, fog, color="#2E5A87", lw=1.8, label="Fog target")
    ax.plot(x, mist, color="#D09A3A", lw=1.8, label="Mist target")
    ax.plot(x, clear, color="#777777", lw=1.8, label="Clear target")
    for val, txt in [(500, "500 m"), (1000, "1000 m")]:
        ax.axvline(val, color="#333333", lw=0.7, ls="--", alpha=0.65)
        ax.text(val + 18, 1.03, txt, ha="left", va="bottom", fontsize=6.5)
    ax.set_xlim(0, 1600)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("Observed visibility (m)")
    ax.set_ylabel("Training target mass")
    ax.set_title("Visibility-aware targets")
    ax.legend(loc="upper right", ncol=1, handlelength=1.4, borderaxespad=0.2)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)


def panel_overall(ax, overall: pd.DataFrame, labels: Sequence[str]) -> None:
    metrics = [
        ("low_vis_csi", "CSI"),
        ("low_vis_recall", "Recall"),
        ("low_vis_precision", "Precision"),
        ("false_positive_rate", "FPR"),
    ]
    x = np.arange(len(metrics))
    width = 0.22
    offsets = np.linspace(-width, width, len(labels))
    for offset, label in zip(offsets, labels):
        vals = [metric_values(overall, labels, m)[labels.index(label)] for m, _ in metrics]
        ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=METHOD_COLORS.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Operational low-visibility skill")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.text(3, 0.98, "lower is better", ha="center", va="top", fontsize=6.4, color="#555555")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1, frameon=False)


def panel_fog_mist_metrics(ax, per_class: pd.DataFrame, labels: Sequence[str]) -> None:
    df = per_class.copy()
    if "display_label" not in df.columns:
        df["display_label"] = df.apply(display_label, axis=1)

    groups = [
        ("Fog", "csi", "Fog\nCSI"),
        ("Fog", "recall", "Fog\nRecall"),
        ("Fog", "precision", "Fog\nPrecision"),
        ("Fog", "far", "Fog\nFAR"),
        ("Mist", "csi", "Mist\nCSI"),
        ("Mist", "recall", "Mist\nRecall"),
        ("Mist", "precision", "Mist\nPrecision"),
        ("Mist", "far", "Mist\nFAR"),
    ]
    x = np.arange(len(groups))
    width = min(0.22, 0.72 / max(1, len(labels)))
    offsets = np.linspace(-width, width, len(labels))

    for offset, label in zip(offsets, labels):
        vals = []
        for class_name, metric, _ in groups:
            sub = df[(df["class_name"] == class_name) & (df["display_label"] == label)]
            value = sub.iloc[0][metric] if not sub.empty and metric in sub.columns else np.nan
            vals.append(float(value) if pd.notna(value) else np.nan)
        ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=METHOD_COLORS.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )

    ax.axvline(3.5, color="#D1D5DB", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, _, name in groups])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fog and Mist skill by class")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.text(3, 0.98, "FAR: lower is better", ha="center", va="top", fontsize=6.4, color="#555555")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1, frameon=False)


def panel_class_heatmap(ax, per_class: pd.DataFrame, labels: Sequence[str]) -> None:
    df = per_class.copy()
    if "display_label" not in df.columns:
        df["display_label"] = df.apply(display_label, axis=1)
    classes = ["Fog", "Mist", "Clear"]
    mat = np.full((len(classes), len(labels)), np.nan)
    for i, cls in enumerate(classes):
        for j, label in enumerate(labels):
            sub = df[(df["class_name"] == cls) & (df["display_label"] == label)]
            if not sub.empty and "csi" in sub.columns:
                mat[i, j] = float(sub.iloc[0]["csi"])
    vmax = max(0.25, float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=min(1.0, vmax * 1.15), aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_title("Class-wise CSI")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isfinite(value):
                color = "white" if value > vmax * 0.55 else "#111111"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5, length=2)


def panel_boundary(ax, boundary: pd.DataFrame, labels: Sequence[str]) -> None:
    df = boundary.copy()
    if "display_label" not in df.columns:
        df["display_label"] = df.apply(display_label, axis=1)
    bands = [
        ("fog_mist_400_600m", "400-600 m"),
        ("mist_clear_800_1200m", "800-1200 m"),
    ]
    x = np.arange(len(bands))
    width = 0.22
    offsets = np.linspace(-width, width, len(labels))
    for offset, label in zip(offsets, labels):
        vals = []
        for band_id, _ in bands:
            sub = df[(df["band_id"] == band_id) & (df["display_label"] == label)]
            vals.append(float(sub.iloc[0]["accuracy"]) if not sub.empty and "accuracy" in sub.columns else np.nan)
        ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=METHOD_COLORS.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in bands])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Observed visibility band")
    ax.set_title("Threshold-neighbour robustness")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def save_figure(fig, out_dir: Path, stem: str, sources: Sequence[Path], notes: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    manifest = pd.DataFrame(
        [
            {
                "figure": f"{stem}.png/pdf/svg",
                "sources": ";".join(str(p) for p in sources),
                "notes": notes
                or "Static-RNN loss-function ablation: objective schematic, test-set skill, class CSI, and transition-band robustness.",
            }
        ]
    )
    manifest.to_csv(out_dir / f"{stem}_source_manifest.csv", index=False)


def save_split_figures(
    out_dir: Path,
    figure_stem: str,
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    boundary: pd.DataFrame,
    labels: Sequence[str],
    sources: Sequence[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(4.1, 3.0), constrained_layout=True)
    panel_soft_targets(ax)
    save_figure(
        fig,
        out_dir,
        f"{figure_stem}_panel_a_soft_targets",
        sources,
        "Standalone panel a: visibility-aware targets used by the proposed focal objective.",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.7, 3.15), constrained_layout=True)
    panel_overall(ax, overall, labels)
    save_figure(
        fig,
        out_dir,
        f"{figure_stem}_panel_b_lowvis_metrics",
        sources,
        "Standalone panel b: aggregate low-visibility CSI, recall, precision and false-positive rate.",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    panel_fog_mist_metrics(ax, per_class, labels)
    save_figure(
        fig,
        out_dir,
        f"{figure_stem}_fog_mist_metrics",
        sources,
        "Fog and Mist per-class CSI, recall, precision and false-alarm ratio.",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.05), constrained_layout=True)
    panel_class_heatmap(ax, per_class, labels)
    save_figure(
        fig,
        out_dir,
        f"{figure_stem}_panel_c_class_csi",
        sources,
        "Standalone panel c: class-wise CSI heatmap.",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.45, 3.1), constrained_layout=True)
    panel_boundary(ax, boundary, labels)
    save_figure(
        fig,
        out_dir,
        f"{figure_stem}_panel_d_boundary_robustness",
        sources,
        "Standalone panel d: accuracy inside the 500 m and 1000 m threshold-neighbour bands.",
    )
    plt.close(fig)


def write_caption(out_dir: Path, stem: str) -> None:
    text = (
        "Figure caption draft\n"
        "a, Visibility-aware soft targets used by the proposed rare-event focal objective around the 500 m and 1000 m operational thresholds. "
        "b, Test-set low-visibility CSI, recall, precision and clear-sky false-positive rate. "
        "An additional standalone panel reports Fog and Mist CSI, recall, precision and false-alarm ratio separately, avoiding reliance on the aggregate low-visibility score alone. "
        "c, Class-wise critical success index for Fog, Mist and Clear. "
        "d, Accuracy inside the two threshold-neighbour visibility bands where hard classification and direct regression are expected to be most fragile.\n"
    )
    (out_dir / f"{stem}_caption.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    eval_dir = Path(args.eval_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else eval_dir
    overall_path = eval_dir / "loss_ablation_overall_metrics.csv"
    per_class_path = eval_dir / "loss_ablation_per_class_metrics.csv"
    boundary_path = eval_dir / "loss_ablation_boundary_metrics.csv"
    for path in (overall_path, per_class_path, boundary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing source table: {path}")

    overall = pd.read_csv(overall_path)
    per_class = pd.read_csv(per_class_path)
    boundary = pd.read_csv(boundary_path)
    if "display_label" not in overall.columns:
        overall["display_label"] = overall.apply(display_label, axis=1)
    labels = ordered_labels(overall)

    fig = plt.figure(figsize=(7.25, 5.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.1], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_soft_targets(ax_a)
    panel_overall(ax_b, overall, labels)
    panel_class_heatmap(ax_c, per_class, labels)
    panel_boundary(ax_d, boundary, labels)

    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), ("a", "b", "c", "d")):
        add_panel_label(ax, label)
    if args.title:
        fig.suptitle(args.title, y=1.02, fontsize=9.5)

    sources = [overall_path, per_class_path, boundary_path]
    save_figure(fig, out_dir, args.figure_stem, sources)
    save_split_figures(out_dir, args.figure_stem, overall, per_class, boundary, labels, sources)
    write_caption(out_dir, args.figure_stem)
    plt.close(fig)


if __name__ == "__main__":
    main()
