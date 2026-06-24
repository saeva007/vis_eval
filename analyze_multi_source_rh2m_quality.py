#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-source key-variable quality and extremeness analysis for overlap datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
if str(VIS_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_EVAL_DIR))

from analyze_key_variable_quality import (  # noqa: E402
    choose_obs_time_shift,
    clean_physical_values,
    convert_forecast_units,
    feature_info,
    metric_row,
    read_build_config,
    read_meta,
    split_list,
)
from feature_catalog_pm10_pm25 import dynamic_features_for_count  # noqa: E402

DEFAULT_KEY_FEATURES = "RH2M,Q_1000,DP_1000,RH_925,PRECIP"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare key-variable extremeness and station-observation accuracy across source datasets.")
    ap.add_argument(
        "--sources",
        required=True,
        help=(
            "Semicolon-separated sources, each as tag=dataset_dir or "
            "tag=dataset_dir|label|group|note. Example: "
            "tianji=/.../ml_dataset...;T2ND=/...|T2ND raw|2025"
        ),
    )
    ap.add_argument("--obs_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--features", default=DEFAULT_KEY_FEATURES)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=27)
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--timezone_probe_files", type=int, default=96)
    ap.add_argument("--thresholds", default="90,95,98,99")
    ap.add_argument("--min_pairs", type=int, default=100)
    return ap.parse_args()


def split_thresholds(value: str) -> List[float]:
    out = []
    for raw in str(value or "").replace(";", ",").split(","):
        raw = raw.strip()
        if raw:
            out.append(float(raw))
    return out or [90.0, 95.0, 98.0, 99.0]


def parse_sources(text: str) -> List[Dict[str, str]]:
    rows = []
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Source entry must be tag=path: {chunk!r}")
        tag, rest = chunk.split("=", 1)
        parts = [p.strip() for p in rest.split("|")]
        path = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else tag.strip()
        group = parts[2] if len(parts) > 2 and parts[2] else ""
        note = parts[3] if len(parts) > 3 else ""
        rows.append({"tag": tag.strip(), "path": path, "label": label, "group": group, "note": note})
    if not rows:
        raise ValueError("No sources parsed.")
    return rows


def feature_columns(features: List[str], window: int, dyn_vars_count: int) -> Dict[str, int]:
    lookup = {item["feature"]: i for i, item in enumerate(dynamic_features_for_count(dyn_vars_count))}
    missing = [feature for feature in features if feature not in lookup]
    if missing:
        raise KeyError(f"Feature(s) not present in dynamic feature catalog: {missing}")
    return {
        feature: (int(window) - 1) * int(dyn_vars_count) + int(lookup[feature])
        for feature in features
    }


def infer_group(data_dir: Path, meta: pd.DataFrame, explicit: str) -> str:
    if explicit:
        return explicit
    cfg = read_build_config(data_dir)
    year = cfg.get("year")
    if year:
        return str(year)
    times = pd.to_datetime(meta["time"], errors="coerce").dropna()
    if not times.empty:
        years = sorted(times.dt.year.astype(int).unique().tolist())
        return "-".join(str(y) for y in years)
    return "unknown"


def tail_curve(values: np.ndarray, thresholds: np.ndarray, direction: str = "high") -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.full(len(thresholds), np.nan, dtype=float)
    if direction == "low":
        return np.asarray([100.0 * np.mean(vals <= th) for th in thresholds], dtype=float)
    return np.asarray([100.0 * np.mean(vals >= th) for th in thresholds], dtype=float)


def source_note(tag: str, cfg: Dict[str, object], explicit_note: str) -> str:
    if explicit_note:
        return explicit_note
    tag_low = tag.lower()
    if "pangu" in tag_low:
        return "RH2M is a proxy approximated from 1000 hPa humidity in the Pangu preprocessing."
    rh_source = cfg.get("rh2m_source")
    if rh_source:
        return f"rh2m_source={rh_source}"
    return ""


def is_t2nd_rh2m_only_source(tag: str, label: str = "") -> bool:
    text = f"{tag} {label}".lower()
    return "t2nd" in text and "rh2m" in text


def load_source(
    entry: Dict[str, str],
    args: argparse.Namespace,
    columns: Dict[str, int],
    features: List[str],
    thresholds: List[float],
) -> Tuple[pd.DataFrame, List[Dict[str, object]], pd.DataFrame, pd.DataFrame]:
    data_dir = Path(entry["path"]).expanduser()
    cfg = read_build_config(data_dir)
    meta = read_meta(data_dir, args.limit_samples)
    group = infer_group(data_dir, meta, entry.get("group", ""))
    X = np.load(data_dir / "X_test.npy", mmap_mode="r")
    best_shift, obs, tz_diag = choose_obs_time_shift(Path(args.obs_root), meta, args.timezone_probe_files)
    merged = meta.merge(obs, on=["time", "station_key"], how="left")
    sample = merged[["time", "station_key", "dup", "row_idx"]].copy()
    sample["tag"] = entry["tag"]
    sample["label"] = entry["label"]
    sample["group"] = group
    rows: List[Dict[str, object]] = []

    for feature in features:
        info = feature_info(feature)
        vals = convert_forecast_units(
            feature,
            np.asarray(X[meta["row_idx"].to_numpy(dtype=np.int64), columns[feature]], dtype=np.float64),
        )
        obs_col = info.get("obs")
        obs_vals = (
            clean_physical_values(feature, pd.to_numeric(merged[str(obs_col)], errors="coerce").to_numpy(dtype=float))
            if obs_col and str(obs_col) in merged
            else np.full(len(merged), np.nan, dtype=float)
        )
        if feature == "PRECIP" and np.isfinite(obs_vals).any():
            obs_vals = np.maximum(obs_vals, 0.0)
        sample[f"forecast_{feature}"] = vals
        sample[f"obs_{feature}"] = obs_vals
        if feature == "RH2M":
            sample["forecast_rh2m"] = vals
            sample["obs_rh2m"] = obs_vals

        finite = vals[np.isfinite(vals)]
        obs_finite = obs_vals[np.isfinite(obs_vals)]
        if obs_finite.size:
            row = metric_row(feature, entry["tag"], vals, obs_vals, str(info.get("extreme", "high")), info.get("threshold"))
        else:
            row = {"feature": feature, "source": entry["tag"], "n": 0}
        row.update(
            {
                "tag": entry["tag"],
                "source_label": entry["label"],
                "group": group,
                "data_dir": str(data_dir),
                "feature_set": cfg.get("feature_set", ""),
                "source_kind": cfg.get("source_kind", ""),
                "source_note": source_note(entry["tag"], cfg, entry.get("note", "")),
                "obs_time_shift_to_utc_hours": float(best_shift),
                "obs_time_interpretation": "raw_obs_time_is_bjt" if best_shift == -8.0 else "raw_obs_time_is_utc",
                "n_forecast_finite": int(finite.size),
                "feature_label": info.get("label", feature),
                "feature_unit": info.get("unit", ""),
            }
        )
        for q in (50, 75, 90, 95, 98, 99):
            row[f"forecast_p{q}"] = float(np.nanpercentile(finite, q)) if finite.size else math.nan
            row[f"obs_p{q}"] = float(np.nanpercentile(obs_finite, q)) if obs_finite.size else math.nan
        if feature == "RH2M":
            for th in thresholds:
                key = str(th).replace(".", "p")
                row[f"forecast_tail_ge_{key}"] = float(np.mean(finite >= th)) if finite.size else math.nan
                row[f"obs_tail_ge_{key}"] = float(np.mean(obs_finite >= th)) if obs_finite.size else math.nan
        rows.append(row)

    return sample, rows, tz_diag.assign(tag=entry["tag"], group=group), pd.DataFrame([cfg])


def pairwise_rows(samples: Dict[str, pd.DataFrame], meta_rows: pd.DataFrame, features: List[str], thresholds: List[float], min_pairs: int) -> pd.DataFrame:
    rows = []
    by_tag = {tag: df for tag, df in samples.items()}
    tags = list(by_tag)
    unique_meta = meta_rows.drop_duplicates("tag", keep="first")
    labels = dict(zip(unique_meta["tag"], unique_meta["source_label"]))
    groups = dict(zip(unique_meta["tag"], unique_meta["group"]))
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            if groups.get(a) != groups.get(b):
                continue
            for feature in features:
                col = f"forecast_{feature}"
                if col not in by_tag[a] or col not in by_tag[b]:
                    continue
                left = by_tag[a][["time", "station_key", "dup", col]].rename(columns={col: "a"})
                right = by_tag[b][["time", "station_key", "dup", col]].rename(columns={col: "b"})
                joined = left.merge(right, on=["time", "station_key", "dup"], how="inner")
                aa = pd.to_numeric(joined["a"], errors="coerce").to_numpy(dtype=float)
                bb = pd.to_numeric(joined["b"], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(aa) & np.isfinite(bb)
                aa = aa[mask]
                bb = bb[mask]
                if aa.size < int(min_pairs):
                    continue
                pooled = np.concatenate([aa, bb])
                info = feature_info(feature)
                direction = str(info.get("extreme", "high"))
                fixed_threshold = info.get("threshold")
                threshold = float(fixed_threshold) if fixed_threshold is not None else float(np.nanpercentile(pooled, 90.0))
                row = {
                    "feature": feature,
                    "feature_label": info.get("label", feature),
                    "feature_unit": info.get("unit", ""),
                    "group": groups.get(a),
                    "source_a": a,
                    "source_b": b,
                    "label_a": labels.get(a, a),
                    "label_b": labels.get(b, b),
                    "n_pair": int(aa.size),
                    "bias_b_minus_a": float(np.nanmean(bb - aa)),
                    "mae_b_minus_a": float(np.nanmean(np.abs(bb - aa))),
                    "corr": float(np.corrcoef(aa, bb)[0, 1]) if aa.size >= 3 and np.nanstd(aa) > 0 and np.nanstd(bb) > 0 else math.nan,
                    "tail_threshold": threshold,
                    "tail_threshold_basis": "fixed_physical_threshold" if fixed_threshold is not None else "paired_forecast_p90",
                    "extreme_type": direction,
                    "a_tail_rate": float(np.mean(aa <= threshold) if direction == "low" else np.mean(aa >= threshold)),
                    "b_tail_rate": float(np.mean(bb <= threshold) if direction == "low" else np.mean(bb >= threshold)),
                }
                for q in (50, 75, 90, 95, 98, 99):
                    row[f"a_p{q}"] = float(np.nanpercentile(aa, q))
                    row[f"b_p{q}"] = float(np.nanpercentile(bb, q))
                    row[f"b_minus_a_p{q}"] = row[f"b_p{q}"] - row[f"a_p{q}"]
                if feature == "RH2M":
                    for th in thresholds:
                        key = str(th).replace(".", "p")
                        row[f"a_tail_ge_{key}"] = float(np.mean(aa >= th))
                        row[f"b_tail_ge_{key}"] = float(np.mean(bb >= th))
                rows.append(row)
    return pd.DataFrame(rows)


def plot_tail_curves(samples: Dict[str, pd.DataFrame], meta_rows: pd.DataFrame, out_dir: Path, features: List[str]) -> None:
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
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    unique_meta = meta_rows.drop_duplicates("tag", keep="first")
    for group, group_meta in unique_meta.groupby("group", sort=False):
        safe_group = str(group).replace("/", "_").replace(" ", "_")
        for feature in features:
            info = feature_info(feature)
            direction = str(info.get("extreme", "high"))
            forecast_values: List[np.ndarray] = []
            obs_values: List[np.ndarray] = []
            for _, row in group_meta.iterrows():
                sample = samples[str(row["tag"])]
                col = f"forecast_{feature}"
                if col in sample:
                    vals = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        forecast_values.append(vals)
                obs_col = f"obs_{feature}"
                if obs_col in sample:
                    obs = pd.to_numeric(sample[obs_col], errors="coerce").to_numpy(dtype=float)
                    obs = obs[np.isfinite(obs)]
                    if obs.size:
                        obs_values.append(obs)
            if not forecast_values:
                continue
            if feature == "RH2M":
                thresholds = np.linspace(50, 100, 101)
            else:
                basis = np.concatenate(obs_values if obs_values else forecast_values)
                lo = float(np.nanpercentile(basis, 50))
                hi = float(np.nanpercentile(basis, 99))
                if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                    continue
                thresholds = np.linspace(lo, hi, 80)

            fig, ax = plt.subplots(figsize=(4.4, 3.1), dpi=240)
            tail_table = pd.DataFrame({"threshold": thresholds})
            status_rows: List[Dict[str, object]] = []
            plotted_values: List[np.ndarray] = []
            plotted_curves: List[Tuple[str, str, np.ndarray]] = []
            missing_notes: List[str] = []
            obs_plotted = False
            for _, row in group_meta.iterrows():
                sample = samples[str(row["tag"])]
                col = f"forecast_{feature}"
                source_tag = str(row["tag"])
                source_label = str(row["source_label"])
                if feature != "RH2M" and is_t2nd_rh2m_only_source(source_tag, source_label):
                    status_rows.append(
                        {
                            "feature": feature,
                            "group": group,
                            "tag": source_tag,
                            "source_label": source_label,
                            "status": "same_as_tianji_for_non_rh2m",
                            "threshold_min": float(thresholds[0]),
                            "threshold_max": float(thresholds[-1]),
                        }
                    )
                    continue
                if col not in sample:
                    status_rows.append(
                        {
                            "feature": feature,
                            "group": group,
                            "tag": source_tag,
                            "source_label": source_label,
                            "status": "missing_forecast_column",
                            "threshold_min": float(thresholds[0]),
                            "threshold_max": float(thresholds[-1]),
                        }
                    )
                    continue
                vals = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                curve = tail_curve(vals, thresholds, direction)
                finite_curve = curve[np.isfinite(curve)]
                tail_table[source_tag] = curve
                curve_status = "ok"
                line_label = source_label
                duplicate_of_tag = ""
                duplicate_of_label = ""
                if finite_vals.size == 0 or finite_curve.size == 0:
                    curve_status = "no_finite_forecast"
                    missing_notes.append(f"{source_label}: no finite {feature}")
                else:
                    curve_max = float(np.nanmax(finite_curve))
                    if curve_max <= 0.0:
                        curve_status = "zero_tail_in_threshold_range"
                        line_label = f"{source_label} (0 in range)"
                        ax.plot(
                            thresholds,
                            curve,
                            lw=1.5,
                            ls=":",
                            marker="o",
                            ms=2.4,
                            markevery=max(1, len(thresholds) // 8),
                            zorder=5,
                            label=line_label,
                        )
                    else:
                        duplicate_match = None
                        for prev_tag, prev_label, prev_curve in plotted_curves:
                            same = np.isfinite(prev_curve) & np.isfinite(curve)
                            if same.any() and np.allclose(prev_curve[same], curve[same], rtol=1e-5, atol=1e-6):
                                duplicate_match = (prev_tag, prev_label)
                                break
                        if duplicate_match is not None:
                            duplicate_of_tag, duplicate_of_label = duplicate_match
                            curve_status = "duplicate_curve"
                            line_label = f"{source_label} = {duplicate_of_label}"
                            ax.plot(
                                thresholds,
                                curve,
                                lw=1.45,
                                ls=(0, (4, 2)),
                                alpha=0.88,
                                zorder=6,
                                label=line_label,
                            )
                        else:
                            ax.plot(thresholds, curve, lw=1.8, label=line_label)
                        plotted_curves.append((source_tag, source_label, curve.copy()))
                    plotted_values.append(finite_curve)
                status_rows.append(
                    {
                        "feature": feature,
                        "group": group,
                        "tag": source_tag,
                        "source_label": source_label,
                        "status": curve_status,
                        "n_forecast_finite": int(finite_vals.size),
                        "forecast_min": float(np.nanmin(finite_vals)) if finite_vals.size else math.nan,
                        "forecast_p50": float(np.nanpercentile(finite_vals, 50)) if finite_vals.size else math.nan,
                        "forecast_p90": float(np.nanpercentile(finite_vals, 90)) if finite_vals.size else math.nan,
                        "forecast_p99": float(np.nanpercentile(finite_vals, 99)) if finite_vals.size else math.nan,
                        "forecast_max": float(np.nanmax(finite_vals)) if finite_vals.size else math.nan,
                        "threshold_min": float(thresholds[0]),
                        "threshold_max": float(thresholds[-1]),
                        "curve_min": float(np.nanmin(finite_curve)) if finite_curve.size else math.nan,
                        "curve_max": float(np.nanmax(finite_curve)) if finite_curve.size else math.nan,
                        "duplicate_of_tag": duplicate_of_tag,
                        "duplicate_of_label": duplicate_of_label,
                    }
                )
                obs_col = f"obs_{feature}"
                if not obs_plotted and obs_col in sample:
                    obs = pd.to_numeric(sample[obs_col], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(obs).sum() > 0:
                        obs_curve = tail_curve(obs, thresholds, direction)
                        ax.plot(thresholds, obs_curve, color="#111111", lw=2.2, ls="--", label="Observed")
                        tail_table["observed"] = obs_curve
                        obs_plotted = True
            relation = "below" if direction == "low" else "above"
            label = str(info.get("label", feature))
            unit = str(info.get("unit", ""))
            ax.set_xlabel(f"{label} threshold" + (f" ({unit})" if unit else ""))
            ax.set_ylabel(f"Frequency {relation} threshold (%)")
            ax.set_title(f"{label} tail recovery ({group})")
            if feature == "RH2M":
                ax.set_xlim(50, 100)
            if plotted_values:
                plotted = np.concatenate(plotted_values)
                plotted = plotted[np.isfinite(plotted)]
                y_top = float(np.nanmax(plotted)) if plotted.size else 0.0
                ax.set_ylim(0, max(1.0, y_top * 1.08))
            else:
                ax.set_ylim(0, 1.0)
            ax.spines["bottom"].set_position(("outward", 3))
            if missing_notes:
                ax.text(
                    0.02,
                    0.03,
                    "\n".join(missing_notes[:3]),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.5,
                    color="#666666",
                )
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout()
            safe_feature = feature.lower().replace("_", "")
            tail_table.to_csv(out_dir / f"key_variable_tail_curve_{safe_feature}_{safe_group}.csv", index=False, float_format="%.6f")
            pd.DataFrame(status_rows).to_csv(
                out_dir / f"key_variable_tail_curve_status_{safe_feature}_{safe_group}.csv",
                index=False,
                float_format="%.6f",
            )
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"fig_key_variable_tail_{safe_feature}_{safe_group}.{ext}", bbox_inches="tight")
            if feature == "RH2M":
                tail_table.to_csv(out_dir / f"rh2m_tail_curve_{safe_group}.csv", index=False, float_format="%.6f")
                for ext in ("png", "pdf"):
                    fig.savefig(out_dir / f"fig_rh2m_tail_multi_source_{safe_group}.{ext}", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = split_thresholds(args.thresholds)
    entries = parse_sources(args.sources)
    features = split_list(args.features)
    columns = feature_columns(features, args.window, args.dyn_vars_count)
    sample_by_tag: Dict[str, pd.DataFrame] = {}
    metric_rows = []
    tz_diags = []
    cfg_rows = []
    for entry in entries:
        sample, rows, tz_diag, cfg_df = load_source(entry, args, columns, features, thresholds)
        sample_by_tag[entry["tag"]] = sample
        metric_rows.extend(rows)
        tz_diags.append(tz_diag)
        cfg_df.insert(0, "tag", entry["tag"])
        cfg_rows.append(cfg_df)
        sample.to_csv(out_dir / f"key_variable_quality_samples_{entry['tag']}.csv", index=False, float_format="%.6f")
        if "RH2M" in features:
            rh_cols = [c for c in sample.columns if c in {"time", "station_key", "dup", "row_idx", "tag", "label", "group", "forecast_rh2m", "obs_rh2m"}]
            sample[rh_cols].to_csv(out_dir / f"rh2m_quality_samples_{entry['tag']}.csv", index=False, float_format="%.6f")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "key_variable_source_quality_metrics.csv", index=False, float_format="%.6f")
    if "feature" in metrics:
        rh_metrics = metrics[metrics["feature"].astype(str) == "RH2M"].copy()
        if not rh_metrics.empty:
            rh_metrics.to_csv(out_dir / "rh2m_source_quality_metrics.csv", index=False, float_format="%.6f")
    pairwise = pairwise_rows(sample_by_tag, metrics, features, thresholds, args.min_pairs)
    pairwise.to_csv(out_dir / "key_variable_source_pairwise_distribution.csv", index=False, float_format="%.6f")
    if "feature" in pairwise:
        rh_pairwise = pairwise[pairwise["feature"].astype(str) == "RH2M"].copy()
        if not rh_pairwise.empty:
            rh_pairwise.to_csv(out_dir / "rh2m_source_pairwise_distribution.csv", index=False, float_format="%.6f")
    if tz_diags:
        tz_all = pd.concat(tz_diags, ignore_index=True)
        tz_all.to_csv(out_dir / "key_variable_observation_time_alignment_diagnostics.csv", index=False)
        tz_all.to_csv(out_dir / "rh2m_observation_time_alignment_diagnostics.csv", index=False)
    if cfg_rows:
        cfg_all = pd.concat(cfg_rows, ignore_index=True, sort=False)
        cfg_all.to_csv(out_dir / "key_variable_source_dataset_configs.csv", index=False)
        cfg_all.to_csv(out_dir / "rh2m_source_dataset_configs.csv", index=False)
    plot_tail_curves(sample_by_tag, metrics, out_dir, features)
    run_config = {
        "sources": entries,
        "obs_root": args.obs_root,
        "features": features,
        "thresholds": thresholds,
        "window": args.window,
        "dyn_vars_count": args.dyn_vars_count,
        "limit_samples": args.limit_samples,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote key-variable multi-source quality outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
