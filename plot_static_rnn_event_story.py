#!/usr/bin/env python3
"""Draw a publication-grade three-event verification story from saved predictions.

The figure is intentionally verification-first: every event is shown at the
national scale as observed classes, predicted classes, TP/FP/FN outcomes, and a
seven-hour footprint-count evolution. It never crops false alarms away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CLASS_COLORS = {0: "#A51C30", 1: "#E6A45C", 2: "#D9D9D9"}
VERIFY_COLORS = {"TP": "#2A6F97", "FP": "#D95F02", "FN": "#7B2CBF", "TN": "#D9D9D9"}
EXTENT = (73.0, 135.0, 18.0, 54.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-sample-csv", required=True)
    p.add_argument("--event-summary-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shp-path", default="")
    p.add_argument("--candidate-label", default="Static MLP + GRU")
    p.add_argument("--decision-rule", choices=["argmax", "lowvis_gate"], default="argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lowvis-gate", type=float, default=0.5)
    p.add_argument("--window-hours", type=int, default=3)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_table(path: Path) -> pd.DataFrame:
    usecols = [
        "station_id",
        "lat",
        "lon",
        "time",
        "y_true",
        "vis_raw_m",
        "pmst_pred",
        "pmst_p_fog",
        "pmst_p_mist",
        "pmst_p_clear",
    ]
    df = pd.read_csv(path, usecols=lambda name: name in usecols)
    required = {"station_id", "lat", "lon", "time", "y_true", "pmst_pred"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.floor("h")
    return df[df["time"].notna()].copy()


def calibrated_probs(df: pd.DataFrame, temperature: float) -> np.ndarray:
    names = ["pmst_p_fog", "pmst_p_mist", "pmst_p_clear"]
    missing = sorted(set(names).difference(df.columns))
    if missing:
        raise KeyError(f"Probability columns required for lowvis_gate: {missing}")
    probs = df[names].to_numpy(dtype=np.float64)
    logits = np.log(np.clip(probs, 1e-9, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predictions(df: pd.DataFrame, rule: str, temperature: float, gate: float) -> np.ndarray:
    if rule == "argmax":
        return df["pmst_pred"].to_numpy(dtype=np.int8)
    probs = calibrated_probs(df, temperature)
    pred = np.full(len(df), 2, dtype=np.int8)
    passed = probs[:, 0] + probs[:, 1] >= float(gate)
    pred[passed] = np.argmax(probs[passed, :2], axis=1).astype(np.int8)
    return pred


def load_boundaries(path: str) -> list[list[np.ndarray]]:
    if not path:
        return []
    shp_path = Path(path)
    if not shp_path.exists():
        print(f"[warn] boundary shapefile not found: {shp_path}")
        return []
    try:
        import shapefile

        reader = shapefile.Reader(str(shp_path), encoding="gbk")
        output: list[list[np.ndarray]] = []
        for shape in reader.shapes():
            points = np.asarray(shape.points, dtype=float)
            cuts = list(shape.parts) + [len(points)]
            output.append([points[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)])
        return output
    except Exception as exc:
        print(f"[warn] pyshp boundary reader failed: {exc}")
    try:
        import geopandas as gpd

        gdf = gpd.read_file(shp_path)
        output = []
        for geometry in gdf.geometry:
            if geometry is None:
                continue
            polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            output.append([np.asarray(polygon.exterior.coords, dtype=float) for polygon in polygons])
        return output
    except Exception as exc:
        print(f"[warn] geopandas boundary reader failed: {exc}")
        return []


def draw_base(ax: plt.Axes, boundaries: Iterable[Iterable[np.ndarray]]) -> None:
    for shape in boundaries:
        for part in shape:
            if len(part):
                ax.plot(part[:, 0], part[:, 1], color="#4A4A4A", linewidth=0.35, zorder=1)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(36.0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scatter_classes(ax: plt.Axes, sub: pd.DataFrame, values: np.ndarray) -> None:
    ax.scatter(sub["lon"], sub["lat"], s=1.0, color=CLASS_COLORS[2], alpha=0.42, linewidths=0, zorder=2)
    for cls, size in ((1, 7.0), (0, 8.5)):
        mask = values == cls
        ax.scatter(
            sub.loc[mask, "lon"],
            sub.loc[mask, "lat"],
            s=size,
            color=CLASS_COLORS[cls],
            edgecolors="white",
            linewidths=0.18,
            alpha=0.94,
            zorder=4,
        )


def verification_codes(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true_low = y <= 1
    pred_low = pred <= 1
    out = np.full(len(y), "TN", dtype=object)
    out[true_low & pred_low] = "TP"
    out[~true_low & pred_low] = "FP"
    out[true_low & ~pred_low] = "FN"
    return out


def scatter_verification(ax: plt.Axes, sub: pd.DataFrame, codes: np.ndarray) -> None:
    order = (("TN", 1.0, 0.30), ("TP", 7.5, 0.95), ("FP", 7.5, 0.90), ("FN", 9.0, 0.95))
    for code, size, alpha in order:
        mask = codes == code
        ax.scatter(
            sub.loc[mask, "lon"],
            sub.loc[mask, "lat"],
            s=size,
            color=VERIFY_COLORS[code],
            alpha=alpha,
            edgecolors="white" if code != "TN" else "none",
            linewidths=0.18,
            zorder=3 if code == "TN" else 5,
        )


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = y <= 1
    predicted = pred <= 1
    tp = int(np.sum(true & predicted))
    fp = int(np.sum(~true & predicted))
    fn = int(np.sum(true & ~predicted))
    tn = int(np.sum(~true & ~predicted))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "csi": tp / max(tp + fp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "area_ratio": int(np.sum(predicted)) / max(int(np.sum(true)), 1),
    }


def hourly_source(df: pd.DataFrame, pred: np.ndarray, peak: pd.Timestamp, window: int, event: int) -> pd.DataFrame:
    rows = []
    for offset in range(-window, window + 1):
        time = peak + pd.Timedelta(hours=offset)
        mask = (df["time"] == time).to_numpy()
        y = df.loc[mask, "y_true"].to_numpy(dtype=np.int8)
        p = pred[mask]
        met = binary_metrics(y, p)
        rows.append(
            {
                "event": event,
                "offset_hour": offset,
                "time": time,
                "observed_low_vis_stations": int(np.sum(y <= 1)),
                "predicted_low_vis_stations": int(np.sum(p <= 1)),
                **met,
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", facecolor="white")


def main() -> None:
    args = parse_args()
    configure_style()
    input_path = Path(args.per_sample_csv).expanduser()
    event_path = Path(args.event_summary_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)
    pred = predictions(df, args.decision_rule, args.temperature, args.lowvis_gate)
    events = pd.read_csv(event_path)
    events["peak_time"] = pd.to_datetime(events["peak_time"], errors="coerce")
    events = events.dropna(subset=["peak_time"]).sort_values("peak_time").head(3).reset_index(drop=True)
    if len(events) != 3:
        raise ValueError(f"Expected three events, found {len(events)}")
    boundaries = load_boundaries(args.shp_path)

    fig = plt.figure(figsize=(183 / 25.4, 168 / 25.4), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.18],
        left=0.035,
        right=0.995,
        bottom=0.105,
        top=0.91,
        wspace=0.045,
        hspace=0.08,
    )
    axes = np.asarray([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)])
    map_rows = []
    hourly_rows = []

    for row_idx, event in events.iterrows():
        peak = pd.Timestamp(event["peak_time"])
        mask = (df["time"] == peak).to_numpy()
        sub = df.loc[mask].copy().reset_index(drop=True)
        y = sub["y_true"].to_numpy(dtype=np.int8)
        p = pred[mask]
        met = binary_metrics(y, p)
        codes = verification_codes(y, p)

        for ax in axes[row_idx, :3]:
            draw_base(ax, boundaries)
        scatter_classes(axes[row_idx, 0], sub, y)
        scatter_classes(axes[row_idx, 1], sub, p)
        scatter_verification(axes[row_idx, 2], sub, codes)

        row_label = f"{chr(97 + row_idx)}  {peak:%Y-%m-%d %H:%M UTC}"
        axes[row_idx, 0].text(
            -0.01,
            1.02,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
        axes[row_idx, 1].text(
            0.5,
            0.015,
            f"Recall {met['recall']:.2f} · CSI {met['csi']:.2f}\nPrecision {met['precision']:.2f} · Area {met['area_ratio']:.2f}",
            transform=axes[row_idx, 1].transAxes,
            ha="center",
            va="bottom",
            fontsize=5.5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.92},
        )

        hourly = hourly_source(df, pred, peak, args.window_hours, row_idx + 1)
        hourly_rows.append(hourly)
        ax = axes[row_idx, 3]
        ax.plot(hourly["offset_hour"], hourly["observed_low_vis_stations"], color="#343A40", marker="o", ms=2.8, lw=1.2)
        ax.plot(hourly["offset_hour"], hourly["predicted_low_vis_stations"], color="#2A6F97", marker="o", ms=2.8, lw=1.2)
        ax.axvline(0, color="#8C8C8C", lw=0.7, ls=(0, (2, 2)))
        ax.fill_between(hourly["offset_hour"], hourly["observed_low_vis_stations"], hourly["predicted_low_vis_stations"], color="#D95F02", alpha=0.10)
        ax.set_xticks(range(-args.window_hours, args.window_hours + 1))
        ax.set_xlabel("Hours from event peak")
        ax.set_ylabel("Low-vis stations")
        ax.grid(axis="y", color="#E5E5E5", lw=0.45)
        ax.set_ylim(bottom=0)

        peak_source = sub[["station_id", "lat", "lon", "y_true"]].copy()
        peak_source["prediction"] = p
        peak_source["verification"] = codes
        peak_source["event"] = row_idx + 1
        peak_source["peak_time"] = peak
        map_rows.append(peak_source)

    for col, title in enumerate(("Observed class", "Model prediction", "Low-vis verification", "Temporal evolution")):
        axes[0, col].set_title(title, pad=11, fontweight="bold")

    class_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=CLASS_COLORS[0], label="Ultra-low", markersize=4.5),
        Line2D([0], [0], marker="o", linestyle="", color=CLASS_COLORS[1], label="Moderate-low", markersize=4.5),
        Line2D([0], [0], marker="o", linestyle="", color=CLASS_COLORS[2], label="Clear", markersize=4.5),
    ]
    verify_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=VERIFY_COLORS[k], label=k, markersize=4.5)
        for k in ("TP", "FP", "FN")
    ]
    line_handles = [
        Line2D([0], [0], color="#343A40", marker="o", lw=1.2, label="Observed count", markersize=3),
        Line2D([0], [0], color="#2A6F97", marker="o", lw=1.2, label="Predicted count", markersize=3),
    ]
    fig.legend(handles=class_handles + verify_handles + line_handles, loc="lower center", ncol=8, bbox_to_anchor=(0.52, 0.025), columnspacing=1.25, handletextpad=0.4)
    rule_text = "three-class argmax" if args.decision_rule == "argmax" else f"validation gate={args.lowvis_gate:.3f}, T={args.temperature:.3f}"
    fig.suptitle(f"Three widespread low-visibility events — {args.candidate_label} ({rule_text})", x=0.515, y=0.972, fontsize=8.4, fontweight="bold")

    stem = out_dir / "fig_three_event_verification_story"
    save_figure(fig, stem, args.dpi)
    plt.close(fig)

    pd.concat(map_rows, ignore_index=True).to_csv(out_dir / "fig_three_event_verification_story_map_source.csv", index=False)
    pd.concat(hourly_rows, ignore_index=True).to_csv(out_dir / "fig_three_event_verification_story_hourly_source.csv", index=False, float_format="%.6f")
    config = {
        "candidate_label": args.candidate_label,
        "decision_rule": args.decision_rule,
        "temperature": args.temperature,
        "lowvis_gate": args.lowvis_gate if args.decision_rule == "lowvis_gate" else None,
        "window_hours": args.window_hours,
        "events": [str(x) for x in events["peak_time"]],
        "national_extent": list(EXTENT),
        "false_alarm_cropping": False,
        "outputs": [str(stem.with_suffix(ext)) for ext in (".png", ".svg", ".pdf", ".tiff")],
    }
    (out_dir / "fig_three_event_verification_story_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[figure] {stem}.png/.svg/.pdf/.tiff")


if __name__ == "__main__":
    main()
