#!/usr/bin/env python3
"""Redraw the VisCast-versus-IFS main figure from evaluation source tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

FIGURE_WIDTH = 9.20
VISCAST = "#2E5A87"
IFS = "#7A7F87"
INK = "#17191B"
GRID = "#E8EAEB"


def delta_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "viscast_ifs_delta",
        [(0.00, "#9E1F36"), (0.24, "#D6604D"), (0.50, "#FFFFFF"), (0.76, "#67A9CF"), (1.00, "#08306B")],
        N=256,
    )


def lead_delta_cmap() -> LinearSegmentedColormap:
    """Signed gain map: purple for losses and green for gains."""
    return LinearSegmentedColormap.from_list(
        "viscast_ifs_lead_delta",
        [
            (0.000, "#6F526E"),
            (0.167, "#9A7B96"),
            (0.333, "#D2C3CF"),
            (0.500, "#F3F1ED"),
            (0.667, "#C4DAD5"),
            (0.833, "#75A79F"),
            (1.000, "#347A73"),
        ],
        N=256,
    )


def nice_delta_limit(values: np.ndarray, quantile: float = 98.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.20
    raw = float(np.nanpercentile(np.abs(vals), quantile))
    if not np.isfinite(raw) or raw <= 0:
        raw = float(np.nanmax(np.abs(vals))) if vals.size else 0.20
    return min(1.0, max(0.10, math.ceil(raw / 0.05) * 0.05))


def nice_hist_limits(values: np.ndarray, min_abs: float) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -min_abs, min_abs
    lo = min(math.floor(float(np.nanmin(vals)) / 0.10) * 0.10, -min_abs)
    hi = max(math.ceil(float(np.nanmax(vals)) / 0.10) * 0.10, min_abs)
    return (lo - 0.10, hi + 0.10) if lo == hi else (lo, hi)


def gaussian_density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.array([], dtype=float)
    std = float(np.nanstd(vals, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return np.array([], dtype=float)
    bw = 1.06 * std * (vals.size ** (-1.0 / 5.0))
    z = (grid[:, None] - vals[None, :]) / bw
    return np.exp(-0.5 * z * z).mean(axis=1) / (bw * math.sqrt(2.0 * math.pi))


def read_shapefile(path: str):
    shp = Path(path)
    if not path or not shp.is_file():
        return None
    try:
        import geopandas as gpd

        return gpd.read_file(str(shp))
    except Exception as exc:
        try:
            import shapefile

            segments = []
            with shapefile.Reader(str(shp)) as reader:
                for shape in reader.shapes():
                    points = np.asarray(shape.points, dtype=float)
                    bounds = list(shape.parts) + [len(points)]
                    for start, end in zip(bounds[:-1], bounds[1:]):
                        if end - start >= 2:
                            segments.append(points[start:end])
            return segments
        except Exception as fallback_exc:
            print(
                f"[WARN] boundary unavailable: geopandas={exc}; pyshp={fallback_exc}",
                flush=True,
            )
            return None


def draw_boundary(ax, shp, color: str = "#404040", linewidth: float = 0.5, zorder: int = 6) -> None:
    if shp is None:
        return
    if hasattr(shp, "boundary"):
        shp.boundary.plot(ax=ax, color=color, linewidth=linewidth, zorder=zorder)
        return
    for segment in shp:
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth, zorder=zorder)


def draw_basemap(ax, shp) -> None:
    draw_boundary(ax, shp)
    ax.set_xlim(72, 136)
    ax.set_ylim(17, 54)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#8A8A8A")
        spine.set_linewidth(0.6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--shp-path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    p.add_argument("--figure-stem", default="fig_viscast_vs_ifs_main_composite")
    p.add_argument("--min-station-lowvis", type=int, default=5)
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
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
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


def load_inputs(eval_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    station_path = eval_dir / "station_model_vs_ifs_metrics.csv"
    lead_path = eval_dir / "model_vs_ifs_metrics_by_display_lead_hour_48h.csv"
    overall_path = eval_dir / "overall_metrics.csv"
    if not lead_path.is_file():
        lead_path = eval_dir / "model_vs_ifs_metrics_by_lead_hour_48h.csv"
    for path in (station_path, lead_path, overall_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    station = pd.read_csv(station_path)
    lead = pd.read_csv(lead_path)
    overall = pd.read_csv(overall_path)
    require_columns(
        station,
        [
            "lat",
            "lon",
            "n_low_vis",
            "delta_low_vis_recall",
            "pmst_fog_recall",
            "ifs_fog_recall",
            "pmst_mist_recall",
            "ifs_mist_recall",
            "pmst_low_vis_recall",
            "ifs_low_vis_recall",
            "pmst_fog_csi",
            "ifs_fog_csi",
            "pmst_mist_csi",
            "ifs_mist_csi",
            "pmst_low_vis_csi",
            "ifs_low_vis_csi",
        ],
        station_path,
    )
    lead_axis = "display_lead_hour" if "display_lead_hour" in lead.columns else "lead_hour"
    require_columns(
        lead,
        [
            lead_axis,
            "Fog_CSI_model",
            "Fog_CSI_ifs",
            "Mist_CSI_model",
            "Mist_CSI_ifs",
            "low_vis_csi_model",
            "low_vis_csi_ifs",
            "Fog_R_model",
            "Fog_R_ifs",
            "Mist_R_model",
            "Mist_R_ifs",
            "low_vis_recall_model",
            "low_vis_recall_ifs",
        ],
        lead_path,
    )
    require_columns(overall, ["source", "Fog_CSI", "Fog_R", "Mist_CSI", "Mist_R", "low_vis_csi", "low_vis_recall"], overall_path)
    return station, lead, overall, {"station": station_path, "lead": lead_path, "overall": overall_path}


def panel_label(ax, letter: str, x: float = -0.12) -> None:
    ax.text(x, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)


def draw_station_map(ax, station: pd.DataFrame, shp, min_count: int) -> pd.DataFrame:
    source = station.copy()
    source["n_low_vis"] = pd.to_numeric(source["n_low_vis"], errors="coerce")
    source["delta_low_vis_recall"] = pd.to_numeric(source["delta_low_vis_recall"], errors="coerce")
    source = source[(source["n_low_vis"] >= min_count) & np.isfinite(source["delta_low_vis_recall"])].copy()
    if source.empty:
        raise ValueError("No station rows remain for the recall-difference map")
    draw_basemap(ax, shp)
    values = source["delta_low_vis_recall"].to_numpy(dtype=float)
    lim = nice_delta_limit(values)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = delta_cmap()
    counts = source["n_low_vis"].to_numpy(dtype=float)
    ref = max(1.0, float(np.nanpercentile(counts, 95)))
    sizes = 12.0 + 24.0 * np.sqrt(np.minimum(counts, ref) / ref)
    order = np.argsort(np.abs(values))
    plotted = source.iloc[order]
    sc = ax.scatter(
        plotted["lon"],
        plotted["lat"],
        c=plotted["delta_low_vis_recall"],
        s=sizes[order],
        marker="D",
        cmap=cmap,
        norm=norm,
        linewidths=0.35,
        edgecolors="#6F6F6F",
        alpha=0.94,
        zorder=3,
    )
    draw_boundary(ax, shp, color="#1F2937", linewidth=0.55, zorder=6)
    # Reserve a shallow, in-frame southern margin for the distribution and
    # color key.  Both components remain part of panel a without masking the
    # station-rich eastern half of the map.
    ax.set_ylim(12.5, 54)
    hist = ax.inset_axes([0.025, 0.035, 0.285, 0.225], zorder=9)
    cax = ax.inset_axes([0.525, 0.060, 0.405, 0.040], zorder=9)
    hist.set_facecolor("white")
    hist.patch.set_alpha(0.97)
    cax.set_facecolor("white")
    cax.patch.set_alpha(0.97)
    better = int((values > 0).sum())
    worse = int((values < 0).sum())
    ax.set_title("Station-level Low-vis recall difference", loc="left", fontweight="bold", pad=5)
    ax.text(
        0.015,
        0.985,
        f"VisCast better: {100 * better / len(source):.0f}%\nIFS better: {100 * worse / len(source):.0f}%\nmedian Δ={np.nanmedian(values):+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#1F2937",
        bbox={"facecolor": "white", "edgecolor": "#B8B8B8", "linewidth": 0.35, "alpha": 0.88, "pad": 2.2},
        zorder=8,
    )
    hlo, hhi = nice_hist_limits(values, lim)
    bins = np.linspace(hlo, hhi, 18)
    density, edges = np.histogram(values, bins=bins, density=True)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    for center, height, width in zip(centers, density, widths):
        hist.bar(center, height, width=width * 0.94, color=cmap(norm(np.clip(center, -lim, lim))), edgecolor="#383838", linewidth=0.2)
    grid = np.linspace(hlo, hhi, 200)
    curve = gaussian_density_curve(values, grid)
    if curve.size:
        hist.plot(grid, curve, color="#1F2937", linewidth=0.9)
    hist.axvline(0.0, color="#C62828", linestyle="--", linewidth=0.8)
    hist.set_xlim(hlo, hhi)
    hist.set_xlabel("Δ recall", fontsize=6.7)
    hist.set_ylabel("Density", fontsize=6.7)
    hist.tick_params(axis="both", labelsize=6.2, length=2)
    hist.grid(False)
    for spine in hist.spines.values():
        spine.set_color("#50555A")
        spine.set_linewidth(0.55)
    cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("")
    cax.set_title("VisCast − IFS recall", fontsize=6.8, pad=1)
    cb.ax.tick_params(labelsize=6.8)
    return source


def select_overall_rows(overall: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    pmst = overall[overall["source"].astype(str).str.lower() == "pmst"].copy()
    if "sample_scope" in pmst.columns:
        matched = pmst[pmst["sample_scope"].astype(str).str.contains("ifs_diagnostic_matched", case=False, na=False)]
        if not matched.empty:
            pmst = matched
    ifs = overall[overall["source"].astype(str).str.lower() == "ifs_diagnostic"].copy()
    if pmst.empty or ifs.empty:
        raise ValueError("overall_metrics.csv must contain PMST matched-test and IFS diagnostic rows")
    return pmst.iloc[0], ifs.iloc[0]


def draw_metric_pair(ax, overall: pd.DataFrame, title: str, csi_metric: str, recall_metric: str, add_legend: bool = False) -> pd.DataFrame:
    vis_row, ifs_row = select_overall_rows(overall)
    vis_csi, ifs_csi = float(vis_row[csi_metric]), float(ifs_row[csi_metric])
    vis_recall, ifs_recall = float(vis_row[recall_metric]), float(ifs_row[recall_metric])
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, [vis_csi, vis_recall], width, color=VISCAST, label="VisCast")
    ax.bar(x + width / 2, [ifs_csi, ifs_recall], width, color=IFS, label="IFS diagnostic VIS")
    ax.set_xticks(x, ["CSI", "Recall"])
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.grid(False)
    if add_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.3)
    return pd.DataFrame(
        {
            "visibility_class": title,
            "metric": ["CSI", "CSI", "Recall", "Recall"],
            "source": ["VisCast", "IFS diagnostic VIS", "VisCast", "IFS diagnostic VIS"],
            "value": [vis_csi, ifs_csi, vis_recall, ifs_recall],
        }
    )


def signed_row_display(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    out[finite & (values == 0)] = 0.0
    for sign in (1.0, -1.0):
        mask = finite & ((values * sign) > 0)
        if not np.any(mask):
            continue
        magnitudes = np.abs(values[mask])
        lo, hi = float(np.nanmin(magnitudes)), float(np.nanmax(magnitudes))
        if np.isclose(lo, hi):
            scaled = np.full_like(magnitudes, 0.65)
        else:
            floor = min(0.18, lo / hi) if hi > 0 else 0.0
            scaled = floor + (1.0 - floor) * (magnitudes - lo) / (hi - lo)
        out[mask] = sign * scaled
    return out


def lead_edges(leads: np.ndarray) -> np.ndarray:
    if len(leads) == 1:
        return np.asarray([leads[0] - 0.5, leads[0] + 0.5])
    mid = (leads[:-1] + leads[1:]) / 2.0
    return np.concatenate([[leads[0] - (mid[0] - leads[0])], mid, [leads[-1] + (leads[-1] - mid[-1])]])


def draw_lead_heatmap(ax, cax, lead: pd.DataFrame) -> pd.DataFrame:
    lead_axis = "display_lead_hour" if "display_lead_hour" in lead.columns else "lead_hour"
    table = lead.copy().sort_values(lead_axis)
    specs = [
        ("Fog_CSI", "Ultra-low CSI"),
        ("Mist_CSI", "Moderate-low CSI"),
        ("low_vis_csi", "Low-vis event CSI"),
        ("Fog_R", "Ultra-low recall"),
        ("Mist_R", "Moderate-low recall"),
        ("low_vis_recall", "Low-vis event recall"),
    ]
    leads = pd.to_numeric(table[lead_axis], errors="coerce").to_numpy(dtype=float)
    matrix = np.full((len(specs), len(table)), np.nan)
    raw_rows = []
    for row_idx, (metric, label) in enumerate(specs):
        model = pd.to_numeric(table[f"{metric}_model"], errors="coerce").to_numpy(dtype=float)
        baseline = pd.to_numeric(table[f"{metric}_ifs"], errors="coerce").to_numpy(dtype=float)
        delta = 100.0 * (model - baseline)
        matrix[row_idx] = signed_row_display(delta)
        for lead_hour, model_value, ifs_value, delta_value, display_value in zip(
            leads,
            model,
            baseline,
            delta,
            matrix[row_idx],
        ):
            raw_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "display_lead_hour": lead_hour,
                    "viscast_value": model_value,
                    "ifs_value": ifs_value,
                    "skill_delta_percentage_points": delta_value,
                    "within_row_display_value": display_value,
                }
            )
    cmap = lead_delta_cmap()
    mesh = ax.pcolormesh(
        lead_edges(leads),
        np.arange(len(specs) + 1) - 0.5,
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        shading="flat",
    )
    ax.set_ylim(len(specs) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(specs)), [label for _, label in specs])
    ticks = [value for value in (0, 6, 12, 24, 36, 48) if np.nanmin(leads) <= value <= np.nanmax(leads)]
    ax.set_xticks(ticks)
    ax.set_xlabel("Display lead time (h)")
    ax.set_title("48 h skill gain over IFS", loc="left", fontweight="bold", pad=5)
    cb = ax.figure.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(-1.0, 1.0, 5))
    cb.set_label("Within-metric normalized VisCast − IFS skill gain", fontsize=7.3)
    cax.tick_params(labelsize=6.8)
    return pd.DataFrame(raw_rows)


def export(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"dpi": dpi} if ext in {"png", "tiff"} else {}
        fig.savefig(out_dir / f"{stem}.{ext}", facecolor="white", **kwargs)


def main() -> None:
    args = parse_args()
    setup_style()
    eval_dir = args.eval_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else eval_dir / "manuscript_figures"
    station, lead, overall, paths = load_inputs(eval_dir)
    shp = read_shapefile(args.shp_path)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 7.55))
    top = fig.add_gridspec(
        1,
        2,
        width_ratios=[4.25, 1.0],
        left=0.045,
        right=0.985,
        top=0.972,
        bottom=0.485,
        wspace=0.17,
    )
    ax_map = fig.add_subplot(top[0, 0])
    right = top[0, 1].subgridspec(3, 1, hspace=0.46)
    small_axes = [fig.add_subplot(right[i, 0]) for i in range(3)]
    bottom = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.11],
        left=0.145,
        right=0.985,
        top=0.395,
        bottom=0.070,
        hspace=0.46,
    )
    ax_heat = fig.add_subplot(bottom[0, 0])
    cax = fig.add_subplot(bottom[1, 0])

    station_source = draw_station_map(
        ax_map,
        station,
        shp,
        args.min_station_lowvis,
    )
    small_sources = [
        draw_metric_pair(small_axes[0], overall, "Ultra-low", "Fog_CSI", "Fog_R", add_legend=True),
        draw_metric_pair(small_axes[1], overall, "Moderate-low", "Mist_CSI", "Mist_R"),
        draw_metric_pair(small_axes[2], overall, "Low-vis event", "low_vis_csi", "low_vis_recall"),
    ]
    lead_source = draw_lead_heatmap(ax_heat, cax, lead)
    panel_label(ax_map, "a", x=-0.06)
    for letter, ax in zip("bcd", small_axes):
        panel_label(ax, letter, x=-0.22)
    panel_label(ax_heat, "e", x=-0.06)

    export(fig, out_dir, args.figure_stem, args.dpi)
    plt.close(fig)
    station_source.to_csv(out_dir / f"{args.figure_stem}_station_source.csv", index=False, float_format="%.8f")
    pd.concat(small_sources, ignore_index=True).to_csv(out_dir / f"{args.figure_stem}_summary_source.csv", index=False, float_format="%.8f")
    lead_source.to_csv(out_dir / f"{args.figure_stem}_lead_source.csv", index=False, float_format="%.8f")
    manifest = {
        "station_metrics": str(paths["station"]),
        "lead_metrics": str(paths["lead"]),
        "overall_metrics": str(paths["overall"]),
        "figure": str(out_dir / f"{args.figure_stem}.pdf"),
        "rendering": "all panels redrawn from source tables on one canvas",
    }
    (out_dir / f"{args.figure_stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_dir / f"{args.figure_stem}.png")


if __name__ == "__main__":
    main()
