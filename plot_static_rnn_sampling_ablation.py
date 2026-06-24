#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Static-RNN sampling-method ablation in the journal figure style."""

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
    "natural_shuffle": "No Low-vis event oversampling",
    "current_stratified": "With Low-vis event oversampling",
    "light_lowvis_oversample": "Light Low-vis event oversampling",
    "heavy_lowvis_oversample": "Heavy Low-vis event oversampling",
}

SHORT_LABELS: Dict[str, str] = {
    "No Low-vis event oversampling": "No\noversampling",
    "With Low-vis event oversampling": "With\noversampling",
    "Light Low-vis event oversampling": "Light\noversampling",
    "Heavy Low-vis event oversampling": "Heavy\noversampling",
}

METHOD_COLORS: Dict[str, str] = {
    "No Low-vis event oversampling": "#6B7280",
    "With Low-vis event oversampling": "#2E5A87",
    "Light Low-vis event oversampling": "#7AA6A1",
    "Heavy Low-vis event oversampling": "#C77C3D",
}

CLASS_COLORS: Dict[str, str] = {
    "Fog": "#2E5A87",
    "Mist": "#D09A3A",
    "Clear": "#8A8F98",
}

CLASS_DISPLAY: Dict[str, str] = {
    "Fog": "Ultra-low",
    "Mist": "Moderate-low",
    "Clear": "Clear",
}

GRID_COLOR = "#E5E7EB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Static-RNN sampling ablation summary figures.")
    p.add_argument("--eval_dir", required=True, help="Directory containing sampling_ablation_*.csv.")
    p.add_argument("--out_dir", default="", help="Output directory; default is --eval_dir.")
    p.add_argument("--figure_stem", default="fig_static_rnn_sampling_method_ablation")
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
            "axes.linewidth": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )


def display_label(row: pd.Series) -> str:
    label = str(row.get("label", ""))
    if label in METHOD_LABELS:
        return METHOD_LABELS[label]
    return str(row.get("display_label") or label)


def ordered_labels(overall: pd.DataFrame) -> List[str]:
    df = overall.copy()
    if "display_label" not in df.columns:
        df["display_label"] = df.apply(display_label, axis=1)
    sort_cols = ["experiment_id"] if "experiment_id" in df.columns else ["display_label"]
    return [str(v) for v in df.sort_values(sort_cols)["display_label"].tolist()]


def short_label(label: str) -> str:
    return SHORT_LABELS.get(label, label.replace(" ", "\n"))


def method_color(label: str) -> str:
    return METHOD_COLORS.get(label, "#9CA3AF")


def class_display(class_name: str) -> str:
    return CLASS_DISPLAY.get(str(class_name), str(class_name))


def grouped_bar_geometry(n_labels: int, total_width: float = 0.78) -> Tuple[np.ndarray, float]:
    n = max(int(n_labels), 1)
    bar_slot = float(total_width) / n
    offsets = (np.arange(n, dtype=float) - (n - 1) / 2.0) * bar_slot
    return offsets, bar_slot * 0.88


def metric_lookup(overall: pd.DataFrame, labels: Sequence[str], metric: str) -> List[float]:
    lookup = overall.set_index("display_label")
    vals: List[float] = []
    for label in labels:
        value = lookup.loc[label, metric] if label in lookup.index and metric in lookup.columns else np.nan
        vals.append(float(value) if pd.notna(value) else np.nan)
    return vals


def read_inputs(eval_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Path]]:
    overall_path = eval_dir / "sampling_ablation_overall_metrics.csv"
    per_class_path = eval_dir / "sampling_ablation_per_class_metrics.csv"
    confusion_path = eval_dir / "sampling_ablation_confusion_counts.csv"
    for path in (overall_path, per_class_path, confusion_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing source table: {path}")

    overall = pd.read_csv(overall_path)
    per_class = pd.read_csv(per_class_path)
    confusion = pd.read_csv(confusion_path)
    overall["display_label"] = overall.apply(display_label, axis=1)
    per_class["display_label"] = per_class.apply(display_label, axis=1)
    confusion["display_label"] = confusion.apply(display_label, axis=1)
    return overall, per_class, confusion, [overall_path, per_class_path, confusion_path]


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def sampling_share_table(overall: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    lookup = overall.set_index("display_label")
    for label in labels:
        row = lookup.loc[label] if label in lookup.index else pd.Series(dtype=object)
        batch_vals = [row.get("batch_fog", np.nan), row.get("batch_mist", np.nan), row.get("batch_clear", np.nan)]
        train_vals = [row.get("train_fog", np.nan), row.get("train_mist", np.nan), row.get("train_clear", np.nan)]
        sample_fog = row.get("sample_fog_ratio", np.nan)
        sample_mist = row.get("sample_mist_ratio", np.nan)

        if all(_finite(v) for v in batch_vals) and sum(float(v) for v in batch_vals) > 0:
            total = sum(float(v) for v in batch_vals)
            fog = float(batch_vals[0]) / total
            mist = float(batch_vals[1]) / total
            source = "batch_class_counts"
        elif _finite(sample_fog) and _finite(sample_mist):
            fog = max(float(sample_fog), 0.0)
            mist = max(float(sample_mist), 0.0)
            source = "configured_ratios"
        elif all(_finite(v) for v in train_vals) and sum(float(v) for v in train_vals) > 0:
            total = sum(float(v) for v in train_vals)
            fog = float(train_vals[0]) / total
            mist = float(train_vals[1]) / total
            source = "natural_train_distribution"
        else:
            fog = np.nan
            mist = np.nan
            source = "missing"
        rows.append(
            {
                "display_label": label,
                "fog_share": fog,
                "mist_share": mist,
                "ultra_low_share": fog,
                "moderate_low_share": mist,
                "low_vis_share": fog + mist if np.isfinite(fog) and np.isfinite(mist) else np.nan,
                "share_source": source,
                "sampler_mode": row.get("sampler_mode", ""),
            }
        )
    return pd.DataFrame(rows)


def panel_sampling_design(ax, overall: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    shares = sampling_share_table(overall, labels)
    x = np.arange(len(labels), dtype=float)
    fog = shares["fog_share"].astype(float).to_numpy()
    mist = shares["mist_share"].astype(float).to_numpy()
    ax.bar(x, fog, width=0.64, color=CLASS_COLORS["Fog"], edgecolor="white", linewidth=0.45, label="Ultra-low target")
    ax.bar(
        x,
        mist,
        width=0.64,
        bottom=np.nan_to_num(fog, nan=0.0),
        color=CLASS_COLORS["Mist"],
        edgecolor="white",
        linewidth=0.45,
        label="Moderate-low target",
    )
    if np.isfinite(fog).any():
        natural = shares[shares["share_source"].astype(str) == "natural_train_distribution"]
        if not natural.empty:
            y = float(natural["low_vis_share"].iloc[0])
            ax.axhline(y, color="#4B5563", lw=0.7, ls="--", alpha=0.65)
            ax.text(0.02, y + 0.012, "natural Low-vis event share", fontsize=8.5, color="#4B5563")
    for i, row in shares.iterrows():
        if str(row["share_source"]) == "missing":
            ax.text(i, 0.03, "natural", ha="center", va="bottom", fontsize=8.5, color="#555555")
    ymax = max(0.32, float(np.nanmax(shares["low_vis_share"].to_numpy(dtype=float))) * 1.28)
    ax.set_ylim(0, min(1.0, ymax))
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(label) for label in labels])
    ax.set_ylabel("Batch share")
    ax.set_title("Sampling target for Low-vis event classes")
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1, frameon=False)
    return shares


def panel_overall(ax, overall: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    metrics = [
        ("low_vis_csi", "CSI"),
        ("low_vis_recall", "Recall"),
        ("low_vis_precision", "Precision"),
    ]
    x = np.arange(len(metrics), dtype=float)
    offsets, bar_width = grouped_bar_geometry(len(labels))
    rows: List[Dict[str, object]] = []
    for offset, label in zip(offsets, labels):
        vals = metric_lookup(overall, labels, metrics[0][0])
        values = [metric_lookup(overall, labels, m)[labels.index(label)] for m, _ in metrics]
        del vals
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=method_color(label),
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        for metric, metric_label in metrics:
            rows.append(
                {
                    "display_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "value": metric_lookup(overall, labels, metric)[labels.index(label)],
                }
            )
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Low-vis event skill")
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1, frameon=False)
    return pd.DataFrame(rows)


def panel_fpr(ax, overall: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    x = np.arange(len(labels), dtype=float)
    values = metric_lookup(overall, labels, "false_positive_rate")
    ax.bar(
        x,
        values,
        width=0.62,
        color=[method_color(label) for label in labels],
        edgecolor="white",
        linewidth=0.45,
    )
    ymax = max(0.04, float(np.nanmax(values)) * 1.35) if np.isfinite(values).any() else 0.1
    for xi, value in zip(x, values):
        if np.isfinite(value):
            ax.text(xi, value + max(0.002, ymax * 0.025), f"{value:.3f}", ha="center", va="bottom", fontsize=8.6)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(label) for label in labels])
    ax.set_ylim(0, min(1.0, ymax))
    ax.set_ylabel("False-positive rate")
    ax.set_title("Clear-condition false positives")
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    ax.text(0.98, 0.96, "lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=8.4, color="#555555")
    return pd.DataFrame(
        [
            {
                "display_label": label,
                "metric": "false_positive_rate",
                "metric_label": "Clear-condition FPR",
                "value": value,
            }
            for label, value in zip(labels, values)
        ]
    )


def panel_ultra_moderate(ax, per_class: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    df = per_class.copy()
    class_aliases = {
        "Fog": {"Fog", "Ultra-low"},
        "Mist": {"Mist", "Moderate-low"},
    }
    groups = [
        ("Fog", "csi", "Ultra-low\nCSI"),
        ("Fog", "recall", "Ultra-low\nRecall"),
        ("Fog", "precision", "Ultra-low\nPrecision"),
        ("Mist", "csi", "Moderate-low\nCSI"),
        ("Mist", "recall", "Moderate-low\nRecall"),
        ("Mist", "precision", "Moderate-low\nPrecision"),
    ]
    x = np.arange(len(groups), dtype=float)
    offsets, bar_width = grouped_bar_geometry(len(labels), total_width=0.82)
    rows: List[Dict[str, object]] = []
    for offset, label in zip(offsets, labels):
        values: List[float] = []
        for class_name, metric, metric_label in groups:
            sub = df[
                (df["class_name"].astype(str).isin(class_aliases[class_name]))
                & (df["display_label"].astype(str) == label)
            ]
            value = float(sub.iloc[0][metric]) if not sub.empty and metric in sub.columns else np.nan
            values.append(value)
            rows.append(
                {
                    "display_label": label,
                    "class_name": class_display(class_name),
                    "internal_class_name": class_name,
                    "metric": metric,
                    "metric_label": metric_label.replace("\n", " "),
                    "value": value,
                }
            )
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=method_color(label),
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    ax.axvline(2.5, color="#D1D5DB", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, _, name in groups])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ultra-low and Moderate-low skill")
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1, frameon=False)
    return pd.DataFrame(rows)


def lowvis_prediction_mix(confusion: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pred_order = ["Fog", "Mist", "Clear"]
    for label in labels:
        sub = confusion[
            (confusion["display_label"].astype(str) == label)
            & (confusion["true_class_name"].astype(str).isin(["Fog", "Mist"]))
        ]
        total = float(sub["count"].sum()) if not sub.empty and "count" in sub.columns else 0.0
        for pred in pred_order:
            count = float(sub[sub["pred_class_name"].astype(str) == pred]["count"].sum()) if total else 0.0
            rows.append(
                {
                    "display_label": label,
                    "pred_class_name": class_display(pred),
                    "internal_pred_class_name": pred,
                    "count": count,
                    "fraction": count / total if total else np.nan,
                    "true_scope": "Ultra-low+Moderate-low",
                }
            )
    return pd.DataFrame(rows)


def panel_lowvis_mix(ax, confusion: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    mix = lowvis_prediction_mix(confusion, labels)
    x = np.arange(len(labels), dtype=float)
    bottom = np.zeros(len(labels), dtype=float)
    for pred in ["Fog", "Mist", "Clear"]:
        pred_label = class_display(pred)
        vals = []
        for label in labels:
            sub = mix[(mix["display_label"] == label) & (mix["pred_class_name"] == pred_label)]
            vals.append(float(sub.iloc[0]["fraction"]) if not sub.empty else np.nan)
        vals_arr = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0)
        ax.bar(
            x,
            vals_arr,
            width=0.64,
            bottom=bottom,
            color=CLASS_COLORS[pred],
            edgecolor="white",
            linewidth=0.45,
            label=f"Pred {pred_label}",
        )
        bottom += vals_arr
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(label) for label in labels])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of observed Low-vis event")
    ax.set_title("Observed Low-vis event prediction mix")
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6)
    ax.text(
        0.98,
        0.96,
        "Pred Clear = missed Low-vis event",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1, frameon=False)
    return mix


def save_outputs(
    fig,
    out_dir: Path,
    stem: str,
    dpi: int,
    source_csvs: Sequence[Path],
    source_data: pd.DataFrame,
    notes: str,
) -> None:
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
                "notes": notes,
            }
        ]
    )
    manifest.to_csv(out_dir / f"{stem}_source_manifest.csv", index=False)


def write_caption(out_dir: Path, stem: str) -> None:
    text = (
        "Figure caption draft\n"
        "Sampling targets for Ultra-low and Moderate-low classes in the no-oversampling and Low-vis event oversampling settings. "
        "A separate panel reports test-set Low-vis event CSI, recall and precision. "
        "Clear-condition false-positive rate is reported as its own panel, with lower values indicating fewer clear samples misclassified as Low-vis event. "
        "A third panel reports Ultra-low and Moderate-low CSI, recall and precision separately, showing whether changes in aggregate Low-vis event skill are driven by one class. "
        "The prediction-mix panel shows the distribution of predictions among observed Low-vis event samples, where the Clear segment corresponds to missed Low-vis events.\n"
    )
    (out_dir / f"{stem}_caption.md").write_text(text, encoding="utf-8")


def save_split_panels(
    out_dir: Path,
    figure_stem: str,
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    labels: Sequence[str],
    sources: Sequence[Path],
    dpi: int,
) -> None:
    panels = [
        ("sampling_design", (4.0, 3.0), lambda ax: panel_sampling_design(ax, overall, labels)),
        ("lowvis_event_metrics", (4.7, 3.05), lambda ax: panel_overall(ax, overall, labels)),
        ("clear_fpr", (3.65, 2.85), lambda ax: panel_fpr(ax, overall, labels)),
        ("ultra_moderate_metrics", (5.0, 3.1), lambda ax: panel_ultra_moderate(ax, per_class, labels)),
        ("lowvis_prediction_mix", (4.1, 3.0), lambda ax: panel_lowvis_mix(ax, confusion, labels)),
    ]
    for suffix, figsize, draw in panels:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        source_data = draw(ax)
        save_outputs(
            fig,
            out_dir,
            f"{figure_stem}_{suffix}",
            dpi,
            sources,
            source_data.assign(panel=suffix),
            f"Separate {suffix.replace('_', ' ')} figure for the Static-RNN sampling-method ablation.",
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    eval_dir = Path(args.eval_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else eval_dir
    overall, per_class, confusion, sources = read_inputs(eval_dir)
    labels = ordered_labels(overall)

    save_split_panels(out_dir, args.figure_stem, overall, per_class, confusion, labels, sources, args.dpi)
    write_caption(out_dir, args.figure_stem)


if __name__ == "__main__":
    main()
