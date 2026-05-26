#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-source RH2M quality and extremeness analysis for overlap datasets."""

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
    metric_row,
    read_build_config,
    read_meta,
)
from feature_catalog_pm10_pm25 import dynamic_features_for_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare RH2M extremeness and station-observation accuracy across source datasets.")
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


def feature_column(window: int, dyn_vars_count: int) -> int:
    lookup = {item["feature"]: i for i, item in enumerate(dynamic_features_for_count(dyn_vars_count))}
    if "RH2M" not in lookup:
        raise KeyError("RH2M not present in dynamic feature catalog.")
    return (int(window) - 1) * int(dyn_vars_count) + int(lookup["RH2M"])


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


def tail_curve(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.full(len(thresholds), np.nan, dtype=float)
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


def load_source(entry: Dict[str, str], args: argparse.Namespace, rh_col: int, thresholds: List[float]) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame, pd.DataFrame]:
    data_dir = Path(entry["path"]).expanduser()
    cfg = read_build_config(data_dir)
    meta = read_meta(data_dir, args.limit_samples)
    group = infer_group(data_dir, meta, entry.get("group", ""))
    X = np.load(data_dir / "X_test.npy", mmap_mode="r")
    vals = convert_forecast_units("RH2M", np.asarray(X[meta["row_idx"].to_numpy(dtype=np.int64), rh_col], dtype=np.float64))
    best_shift, obs, tz_diag = choose_obs_time_shift(Path(args.obs_root), meta, args.timezone_probe_files)
    merged = meta.merge(obs, on=["time", "station_key"], how="left")
    obs_vals = (
        clean_physical_values("RH2M", pd.to_numeric(merged["rhu"], errors="coerce").to_numpy(dtype=float))
        if "rhu" in merged
        else np.full(len(merged), np.nan, dtype=float)
    )
    metrics = metric_row("RH2M", entry["tag"], vals, obs_vals, "high", 90.0)
    finite = vals[np.isfinite(vals)]
    obs_finite = obs_vals[np.isfinite(obs_vals)]
    row: Dict[str, object] = {
        **metrics,
        "tag": entry["tag"],
        "label": entry["label"],
        "group": group,
        "data_dir": str(data_dir),
        "feature_set": cfg.get("feature_set", ""),
        "source_kind": cfg.get("source_kind", ""),
        "source_note": source_note(entry["tag"], cfg, entry.get("note", "")),
        "obs_time_shift_to_utc_hours": float(best_shift),
        "obs_time_interpretation": "raw_obs_time_is_bjt" if best_shift == -8.0 else "raw_obs_time_is_utc",
        "n_forecast_finite": int(finite.size),
    }
    for q in (50, 75, 90, 95, 98, 99):
        row[f"forecast_p{q}"] = float(np.nanpercentile(finite, q)) if finite.size else math.nan
        row[f"obs_p{q}"] = float(np.nanpercentile(obs_finite, q)) if obs_finite.size else math.nan
    for th in thresholds:
        key = str(th).replace(".", "p")
        row[f"forecast_tail_ge_{key}"] = float(np.mean(finite >= th)) if finite.size else math.nan
        row[f"obs_tail_ge_{key}"] = float(np.mean(obs_finite >= th)) if obs_finite.size else math.nan
    sample = merged[["time", "station_key", "dup", "row_idx"]].copy()
    sample["tag"] = entry["tag"]
    sample["label"] = entry["label"]
    sample["group"] = group
    sample["forecast_rh2m"] = vals
    sample["obs_rh2m"] = obs_vals
    return sample, row, tz_diag.assign(tag=entry["tag"], group=group), pd.DataFrame([cfg])


def pairwise_rows(samples: Dict[str, pd.DataFrame], meta_rows: pd.DataFrame, thresholds: List[float], min_pairs: int) -> pd.DataFrame:
    rows = []
    by_tag = {tag: df for tag, df in samples.items()}
    tags = list(by_tag)
    labels = dict(zip(meta_rows["tag"], meta_rows["label"]))
    groups = dict(zip(meta_rows["tag"], meta_rows["group"]))
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            if groups.get(a) != groups.get(b):
                continue
            left = by_tag[a][["time", "station_key", "dup", "forecast_rh2m"]].rename(columns={"forecast_rh2m": "a"})
            right = by_tag[b][["time", "station_key", "dup", "forecast_rh2m"]].rename(columns={"forecast_rh2m": "b"})
            joined = left.merge(right, on=["time", "station_key", "dup"], how="inner")
            aa = pd.to_numeric(joined["a"], errors="coerce").to_numpy(dtype=float)
            bb = pd.to_numeric(joined["b"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(aa) & np.isfinite(bb)
            aa = aa[mask]
            bb = bb[mask]
            if aa.size < int(min_pairs):
                continue
            row = {
                "group": groups.get(a),
                "source_a": a,
                "source_b": b,
                "label_a": labels.get(a, a),
                "label_b": labels.get(b, b),
                "n_pair": int(aa.size),
                "bias_b_minus_a": float(np.nanmean(bb - aa)),
                "mae_b_minus_a": float(np.nanmean(np.abs(bb - aa))),
                "corr": float(np.corrcoef(aa, bb)[0, 1]) if aa.size >= 3 and np.nanstd(aa) > 0 and np.nanstd(bb) > 0 else math.nan,
            }
            for q in (90, 95, 98, 99):
                row[f"a_p{q}"] = float(np.nanpercentile(aa, q))
                row[f"b_p{q}"] = float(np.nanpercentile(bb, q))
                row[f"b_minus_a_p{q}"] = row[f"b_p{q}"] - row[f"a_p{q}"]
            for th in thresholds:
                key = str(th).replace(".", "p")
                row[f"a_tail_ge_{key}"] = float(np.mean(aa >= th))
                row[f"b_tail_ge_{key}"] = float(np.mean(bb >= th))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_tail_curves(samples: Dict[str, pd.DataFrame], meta_rows: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    thresholds = np.linspace(50, 100, 101)
    for group, group_meta in meta_rows.groupby("group", sort=False):
        fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=240)
        obs_plotted = False
        tail_table = pd.DataFrame({"threshold": thresholds})
        for _, row in group_meta.iterrows():
            sample = samples[str(row["tag"])]
            vals = pd.to_numeric(sample["forecast_rh2m"], errors="coerce").to_numpy(dtype=float)
            curve = tail_curve(vals, thresholds)
            ax.plot(thresholds, curve, lw=1.8, label=str(row["label"]))
            tail_table[str(row["tag"])] = curve
            if not obs_plotted:
                obs = pd.to_numeric(sample["obs_rh2m"], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(obs).sum() > 0:
                    obs_curve = tail_curve(obs, thresholds)
                    ax.plot(thresholds, obs_curve, color="#111111", lw=2.2, ls="--", label="Observed RH")
                    tail_table["observed"] = obs_curve
                    obs_plotted = True
        ax.set_xlabel("RH2M threshold (%)")
        ax.set_ylabel("Frequency above threshold (%)")
        ax.set_title(f"Near-saturation tail recovery ({group})")
        ax.set_xlim(50, 100)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        safe_group = str(group).replace("/", "_").replace(" ", "_")
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
    rh_col = feature_column(args.window, args.dyn_vars_count)
    sample_by_tag: Dict[str, pd.DataFrame] = {}
    metric_rows = []
    tz_diags = []
    cfg_rows = []
    for entry in entries:
        sample, row, tz_diag, cfg_df = load_source(entry, args, rh_col, thresholds)
        sample_by_tag[entry["tag"]] = sample
        metric_rows.append(row)
        tz_diags.append(tz_diag)
        cfg_df.insert(0, "tag", entry["tag"])
        cfg_rows.append(cfg_df)
        sample.to_csv(out_dir / f"rh2m_quality_samples_{entry['tag']}.csv", index=False, float_format="%.6f")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "rh2m_source_quality_metrics.csv", index=False, float_format="%.6f")
    pairwise = pairwise_rows(sample_by_tag, metrics, thresholds, args.min_pairs)
    pairwise.to_csv(out_dir / "rh2m_source_pairwise_distribution.csv", index=False, float_format="%.6f")
    if tz_diags:
        pd.concat(tz_diags, ignore_index=True).to_csv(out_dir / "rh2m_observation_time_alignment_diagnostics.csv", index=False)
    if cfg_rows:
        pd.concat(cfg_rows, ignore_index=True, sort=False).to_csv(out_dir / "rh2m_source_dataset_configs.csv", index=False)
    plot_tail_curves(sample_by_tag, metrics, out_dir)
    run_config = {
        "sources": entries,
        "obs_root": args.obs_root,
        "thresholds": thresholds,
        "window": args.window,
        "dyn_vars_count": args.dyn_vars_count,
        "limit_samples": args.limit_samples,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote RH2M multi-source quality outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
