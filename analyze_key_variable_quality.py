#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Tianji and IFS key-variable forecast quality against station observations.

The script uses paired overlap datasets so that Tianji and IFS values share the
same station, valid time, target label and feature layout. Observation CSV time
alignment is selected automatically between raw UTC and raw BJT-to-UTC.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
if str(VIS_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_EVAL_DIR))

from feature_catalog_pm10_pm25 import dynamic_features_for_count


OBS_VAR_MAP = {
    "RH2M": {"obs": "rhu", "label": "2 m RH", "unit": "%", "extreme": "high", "threshold": 90.0, "valid": (0.0, 100.0)},
    "T2M": {"obs": "tem", "label": "2 m temperature", "unit": "degC", "extreme": "low", "threshold": None, "valid": (-80.0, 60.0)},
    "WSPD10": {"obs": "win_s_avg_10mi", "label": "10 m wind speed", "unit": "m s-1", "extreme": "high", "threshold": None, "valid": (0.0, 80.0)},
    "MSLP": {"obs": "prs_sea", "label": "Sea-level pressure", "unit": "hPa", "extreme": "high", "threshold": None, "valid": (800.0, 1100.0)},
    "PRECIP": {"obs": "pre_1h", "label": "Hourly precipitation", "unit": "mm h-1", "extreme": "high", "threshold": 0.1, "valid": (0.0, 500.0)},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Key-variable forecast quality: Tianji vs IFS vs station observations.")
    ap.add_argument("--tianji_data_dir", required=True)
    ap.add_argument("--ifs_data_dir", required=True)
    ap.add_argument("--obs_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--features", default="RH2M,T2M,WSPD10,MSLP,PRECIP")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=27)
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--timezone_probe_files", type=int, default=96)
    ap.add_argument("--min_pairs", type=int, default=100)
    return ap.parse_args()


def split_list(value: str) -> List[str]:
    return [x.strip() for chunk in str(value or "").split(";") for x in chunk.split(",") if x.strip()]


def read_build_config(data_dir: Path) -> Dict[str, object]:
    path = data_dir / "dataset_build_config.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warn] cannot read {path}: {exc}", flush=True)
        return {}


def available_dynamic_features(data_dir: Path, dyn_vars_count: int) -> Set[str]:
    """Features that are physically populated in this dataset, not just present as zero slots."""
    cfg = read_build_config(data_dir)
    explicit = cfg.get("overlap_vars") or cfg.get("dynamic_features") or cfg.get("feature_order")
    if explicit:
        available = {str(v) for v in explicit}
        if {"U10", "V10"}.issubset(available):
            available.add("WSPD10")
        return available
    return {item["feature"] for item in dynamic_features_for_count(dyn_vars_count)}


def clean_physical_values(feature: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e5] = np.nan
    valid = OBS_VAR_MAP.get(feature, {}).get("valid")
    if valid is not None:
        lo, hi = valid
        arr[(arr < float(lo)) | (arr > float(hi))] = np.nan
    return arr


def normalize_station_ids(values) -> pd.Series:
    s = pd.Series(values)
    numeric = pd.to_numeric(s, errors="coerce")
    out = s.astype(str)
    mask = numeric.notna()
    if mask.any():
        out.loc[mask] = numeric.loc[mask].astype(np.int64).astype(str)
    return out


def read_meta(data_dir: Path, limit_samples: int) -> pd.DataFrame:
    path = data_dir / "meta_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing meta_test.csv: {path}")
    df = pd.read_csv(path)
    if limit_samples and limit_samples > 0:
        df = df.iloc[: int(limit_samples)].copy()
    if "time" not in df or "station_id" not in df:
        raise KeyError(f"{path} must contain time and station_id columns")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["station_key"] = normalize_station_ids(df["station_id"].values).values
    df["dup"] = df.groupby(["time", "station_key"]).cumcount()
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return df


def align_meta(tianji_meta: pd.DataFrame, ifs_meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    left = tianji_meta[["time", "station_key", "dup", "row_idx"]].rename(columns={"row_idx": "idx_tianji"})
    right = ifs_meta[["time", "station_key", "dup", "row_idx"]].rename(columns={"row_idx": "idx_ifs"})
    joined = left.merge(right, on=["time", "station_key", "dup"], how="inner", sort=False)
    if joined.empty:
        raise RuntimeError("No paired Tianji/IFS test rows by (time, station_id).")
    meta = tianji_meta.iloc[joined["idx_tianji"].to_numpy(dtype=np.int64)].reset_index(drop=True).copy()
    meta["idx_tianji"] = joined["idx_tianji"].to_numpy(dtype=np.int64)
    meta["idx_ifs"] = joined["idx_ifs"].to_numpy(dtype=np.int64)
    return meta["idx_tianji"].to_numpy(dtype=np.int64), meta["idx_ifs"].to_numpy(dtype=np.int64), meta


def obs_file_candidates(obs_root: Path, raw_time: pd.Timestamp) -> List[Path]:
    yyyymm = raw_time.strftime("%Y%m")
    stamp = raw_time.strftime("%Y%m%d%H")
    day_plain = str(int(raw_time.strftime("%d")))
    day_2 = raw_time.strftime("%d")
    return [
        obs_root / yyyymm / day_plain / f"{stamp}.csv",
        obs_root / yyyymm / day_2 / f"{stamp}.csv",
        obs_root / yyyymm / f"{stamp}.csv",
    ]


def read_one_obs_file(path: Path, obs_time_shift_to_utc_hours: float, needed_stations: set) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"station_id_c": str, "station_id": str})
    except Exception as exc:
        print(f"[obs] failed to read {path}: {exc}", flush=True)
        return pd.DataFrame()
    station_col = "station_id_c" if "station_id_c" in df else "station_id" if "station_id" in df else ""
    if not station_col:
        return pd.DataFrame()
    df["station_key"] = normalize_station_ids(df[station_col].values).values
    df = df[df["station_key"].isin(needed_stations)].copy()
    if df.empty:
        return df
    if "obs_time" in df:
        raw_time = pd.to_datetime(df["obs_time"], errors="coerce")
    else:
        raw_time = pd.to_datetime(path.stem, format="%Y%m%d%H", errors="coerce")
        raw_time = pd.Series(raw_time, index=df.index)
    df["time"] = raw_time + pd.to_timedelta(float(obs_time_shift_to_utc_hours), unit="h")
    keep_cols = ["time", "station_key"] + [c for c in sorted({v["obs"] for v in OBS_VAR_MAP.values()}) if c in df]
    return df[keep_cols]


def load_obs_for_meta(
    obs_root: Path,
    meta: pd.DataFrame,
    obs_time_shift_to_utc_hours: float,
    max_files: int = 0,
) -> pd.DataFrame:
    needed_stations = set(meta["station_key"].astype(str))
    times = pd.DatetimeIndex(pd.to_datetime(meta["time"], errors="coerce").dropna().unique()).sort_values()
    if max_files and max_files > 0:
        times = times[: int(max_files)]
    frames: List[pd.DataFrame] = []
    seen_files = set()
    for valid_time in times:
        raw_time = pd.Timestamp(valid_time) - pd.to_timedelta(float(obs_time_shift_to_utc_hours), unit="h")
        selected: Optional[Path] = None
        for cand in obs_file_candidates(obs_root, raw_time):
            if cand.exists():
                selected = cand
                break
        if selected is None or selected in seen_files:
            continue
        seen_files.add(selected)
        part = read_one_obs_file(selected, obs_time_shift_to_utc_hours, needed_stations)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["time", "station_key"])
    obs = pd.concat(frames, ignore_index=True)
    obs = obs.drop_duplicates(["time", "station_key"], keep="first")
    return obs


def choose_obs_time_shift(obs_root: Path, meta: pd.DataFrame, probe_files: int) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    rows = []
    best_shift = 0.0
    best_obs = pd.DataFrame()
    best_matches = -1
    for shift, label in [(0.0, "raw_obs_time_is_utc"), (-8.0, "raw_obs_time_is_bjt")]:
        obs = load_obs_for_meta(obs_root, meta, shift, max_files=probe_files)
        probe = meta[["time", "station_key"]].merge(obs[["time", "station_key"]], on=["time", "station_key"], how="inner")
        matches = int(len(probe))
        rows.append({"candidate": label, "obs_time_shift_to_utc_hours": shift, "matched_pairs_probe": matches})
        if matches > best_matches:
            best_matches = matches
            best_shift = shift
            best_obs = obs
    diag = pd.DataFrame(rows)
    final_obs = load_obs_for_meta(obs_root, meta, best_shift, max_files=0)
    return best_shift, final_obs, diag


def convert_forecast_units(feature: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if feature in {"T2M", "T_925"} and np.nanmedian(arr) > 150:
        arr = arr - 273.15
    if feature in {"MSLP"} and np.nanmedian(np.abs(arr)) > 2000:
        arr = arr / 100.0
    if feature == "PRECIP":
        arr = np.maximum(arr, 0.0)
    if feature == "RH2M":
        arr = np.clip(arr, 0.0, 100.0)
    return clean_physical_values(feature, arr)


def metric_row(feature: str, source: str, pred: np.ndarray, obs: np.ndarray, extreme_type: str, threshold: Optional[float]) -> Dict[str, object]:
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]
    if len(obs) == 0:
        return {"feature": feature, "source": source, "n": 0}
    err = pred - obs
    corr = float(np.corrcoef(pred, obs)[0, 1]) if len(obs) >= 3 and np.std(pred) > 0 and np.std(obs) > 0 else math.nan
    if threshold is None:
        threshold = float(np.nanpercentile(obs, 90 if extreme_type == "high" else 10))
    if extreme_type == "low":
        obs_extreme = obs <= threshold
        pred_extreme = pred <= threshold
    else:
        obs_extreme = obs >= threshold
        pred_extreme = pred >= threshold
    hit = float(np.mean(pred_extreme[obs_extreme])) if int(obs_extreme.sum()) > 0 else math.nan
    false_alarm = float(np.mean(pred_extreme[~obs_extreme])) if int((~obs_extreme).sum()) > 0 else math.nan
    return {
        "feature": feature,
        "source": source,
        "n": int(len(obs)),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": corr,
        "obs_p10": float(np.nanpercentile(obs, 10)),
        "obs_p50": float(np.nanpercentile(obs, 50)),
        "obs_p90": float(np.nanpercentile(obs, 90)),
        "forecast_p90": float(np.nanpercentile(pred, 90)),
        "extreme_type": extreme_type,
        "extreme_threshold": float(threshold),
        "obs_extreme_count": int(obs_extreme.sum()),
        "extreme_hit_rate": hit,
        "extreme_false_alarm_rate": false_alarm,
    }


def make_quality_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    tianji_dir = Path(args.tianji_data_dir)
    ifs_dir = Path(args.ifs_data_dir)
    obs_root = Path(args.obs_root)
    t_meta = read_meta(tianji_dir, args.limit_samples)
    i_meta = read_meta(ifs_dir, args.limit_samples)
    idx_t, idx_i, meta = align_meta(t_meta, i_meta)
    best_shift, obs, tz_diag = choose_obs_time_shift(obs_root, meta, args.timezone_probe_files)
    merged = meta.merge(obs, on=["time", "station_key"], how="left", suffixes=("", "_obs"))
    matched_obs = int(merged[[v["obs"] for v in OBS_VAR_MAP.values() if v["obs"] in merged]].notna().any(axis=1).sum())

    feature_lookup = {item["feature"]: i for i, item in enumerate(dynamic_features_for_count(args.dyn_vars_count))}
    features = split_list(args.features)
    t_available = available_dynamic_features(tianji_dir, args.dyn_vars_count)
    i_available = available_dynamic_features(ifs_dir, args.dyn_vars_count)
    x_t = np.load(tianji_dir / "X_test.npy", mmap_mode="r")
    x_i = np.load(ifs_dir / "X_test.npy", mmap_mode="r")
    rows = []
    skipped_features = []
    sample = merged[["time", "station_key", "idx_tianji", "idx_ifs"]].copy()
    for feature in features:
        if feature not in feature_lookup or feature not in OBS_VAR_MAP:
            print(f"[skip] feature={feature} lacks dynamic index or obs mapping.", flush=True)
            skipped_features.append({"feature": feature, "reason": "missing_index_or_obs_mapping"})
            continue
        missing_sources = [name for name, available in (("tianji", t_available), ("ifs", i_available)) if feature not in available]
        if missing_sources:
            print(
                f"[skip] feature={feature} is not populated in {','.join(missing_sources)} overlap dataset(s).",
                flush=True,
            )
            skipped_features.append({"feature": feature, "reason": "not_populated", "sources": missing_sources})
            continue
        obs_col = OBS_VAR_MAP[feature]["obs"]
        if obs_col not in merged:
            print(f"[skip] obs column {obs_col!r} missing for feature={feature}", flush=True)
            skipped_features.append({"feature": feature, "reason": f"missing_obs_column:{obs_col}"})
            continue
        col = (int(args.window) - 1) * int(args.dyn_vars_count) + int(feature_lookup[feature])
        obs_vals = clean_physical_values(feature, pd.to_numeric(merged[obs_col], errors="coerce").to_numpy(dtype=float))
        tj_vals = convert_forecast_units(feature, np.asarray(x_t[idx_t, col], dtype=np.float64))
        ifs_vals = convert_forecast_units(feature, np.asarray(x_i[idx_i, col], dtype=np.float64))
        if feature == "PRECIP":
            obs_vals = np.maximum(obs_vals, 0.0)
        info = OBS_VAR_MAP[feature]
        rows.append(metric_row(feature, "tianji", tj_vals, obs_vals, info["extreme"], info["threshold"]))
        rows.append(metric_row(feature, "ifs", ifs_vals, obs_vals, info["extreme"], info["threshold"]))
        sample[f"obs_{feature}"] = obs_vals
        sample[f"tianji_{feature}"] = tj_vals
        sample[f"ifs_{feature}"] = ifs_vals
    metrics = pd.DataFrame(rows)
    diag = {
        "paired_rows": int(len(meta)),
        "matched_obs_any_variable": matched_obs,
        "obs_time_shift_to_utc_hours": float(best_shift),
        "obs_time_interpretation": "raw_obs_time_is_bjt" if best_shift == -8.0 else "raw_obs_time_is_utc",
        "features": features,
        "evaluated_features": sorted(metrics["feature"].dropna().unique().tolist()) if not metrics.empty else [],
        "skipped_features": skipped_features,
    }
    return metrics, sample, tz_diag, diag


def plot_quality(metrics: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
        }
    )
    label_map = {k: v["label"] for k, v in OBS_VAR_MAP.items()}
    colors = {"tianji": "#2E5A87", "ifs": "#7A7A7A"}
    features = [f for f in split_list(",".join(metrics["feature"].dropna().astype(str).unique())) if f in set(metrics["feature"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), gridspec_kw={"height_ratios": [1.0, 1.05]})
    ax = axes[0, 0]
    x = np.arange(len(features))
    width = 0.36
    for j, source in enumerate(["tianji", "ifs"]):
        vals = []
        for feature in features:
            row = metrics[(metrics["feature"] == feature) & (metrics["source"] == source)]
            vals.append(float(row["mae"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (j - 0.5) * width, vals, width * 0.92, color=colors[source], label=source.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(f, f) for f in features], rotation=25, ha="right")
    ax.set_ylabel("MAE (native units)")
    ax.set_title("Absolute forecast error")
    ax.legend(frameon=False)
    ax.text(-0.12, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[0, 1]
    delta_rows = []
    for feature in features:
        tj = metrics[(metrics["feature"] == feature) & (metrics["source"] == "tianji")]
        ii = metrics[(metrics["feature"] == feature) & (metrics["source"] == "ifs")]
        if tj.empty or ii.empty:
            continue
        delta_rows.append((feature, float(ii["mae"].iloc[0]) - float(tj["mae"].iloc[0])))
    ax.barh(
        np.arange(len(delta_rows)),
        [v for _, v in delta_rows],
        color=["#18864B" if v > 0 else "#B45B43" for _, v in delta_rows],
    )
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(np.arange(len(delta_rows)))
    ax.set_yticklabels([label_map.get(f, f) for f, _ in delta_rows])
    ax.set_xlabel("MAE advantage (IFS - Tianji)")
    ax.set_title("Positive values favor Tianji")
    ax.text(-0.12, 1.04, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1, 0]
    tail_feature = ""
    if "obs_RH2M" in sample and sample["obs_RH2M"].notna().sum() > 0:
        tail_feature = "RH2M"
        thresholds = np.arange(60, 101, 2, dtype=float)
    else:
        for candidate in features:
            if OBS_VAR_MAP.get(candidate, {}).get("threshold") is not None and f"obs_{candidate}" in sample:
                tail_feature = candidate
                break
        if tail_feature == "PRECIP":
            thresholds = np.asarray([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=float)
        elif tail_feature:
            vals = pd.to_numeric(sample[f"obs_{tail_feature}"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            thresholds = np.linspace(np.nanpercentile(vals, 50), np.nanpercentile(vals, 99), 24) if vals.size else np.asarray([])
        else:
            thresholds = np.asarray([])
    if tail_feature and thresholds.size:
        info = OBS_VAR_MAP[tail_feature]
        direction = info["extreme"]
        for col, label, color, ls in [
            (f"obs_{tail_feature}", "Observed", "#111111", "-"),
            (f"tianji_{tail_feature}", "Tianji", colors["tianji"], "-"),
            (f"ifs_{tail_feature}", "IFS", colors["ifs"], "--"),
        ]:
            if col not in sample:
                continue
            vals = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                if direction == "low":
                    yvals = [100.0 * np.mean(vals <= th) for th in thresholds]
                else:
                    yvals = [100.0 * np.mean(vals >= th) for th in thresholds]
                ax.plot(thresholds, yvals, color=color, lw=2.0, ls=ls, label=label)
        unit = info["unit"]
        relation = "below" if direction == "low" else "above"
        ax.set_xlabel(f"{label_map.get(tail_feature, tail_feature)} threshold ({unit})")
        ax.set_ylabel(f"Samples {relation} threshold (%)")
        ax.set_title(f"{label_map.get(tail_feature, tail_feature)} tail capture")
        if tail_feature == "PRECIP":
            ax.set_xscale("log")
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No threshold-based shared variable",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    ax.text(-0.12, 1.04, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1, 1]
    extreme = metrics[np.isfinite(pd.to_numeric(metrics.get("extreme_hit_rate", np.nan), errors="coerce"))].copy()
    extreme_features = [f for f in features if f in set(extreme["feature"])]
    x2 = np.arange(len(extreme_features))
    for j, source in enumerate(["tianji", "ifs"]):
        vals = []
        for feature in extreme_features:
            row = extreme[(extreme["feature"] == feature) & (extreme["source"] == source)]
            vals.append(float(row["extreme_hit_rate"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x2 + (j - 0.5) * width, vals, width * 0.92, color=colors[source], label=source.capitalize())
    ax.set_xticks(x2)
    ax.set_xticklabels([label_map.get(f, f) for f in extreme_features], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Extreme-event hit rate")
    ax.set_title("Can the source reproduce critical tails?")
    ax.legend(frameon=False)
    ax.text(-0.12, 1.04, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"fig_key_variable_quality_tianji_vs_ifs.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, sample, tz_diag, diag = make_quality_tables(args)
    metrics_path = out_dir / "key_variable_quality_metrics.csv"
    sample_path = out_dir / "key_variable_quality_samples.csv"
    tz_path = out_dir / "observation_time_alignment_diagnostics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    sample.to_csv(sample_path, index=False, float_format="%.6f")
    tz_diag.to_csv(tz_path, index=False)
    with open(out_dir / "key_variable_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[table] {metrics_path}", flush=True)
    print(f"[table] {sample_path}", flush=True)
    print(f"[time] {diag['obs_time_interpretation']} shift={diag['obs_time_shift_to_utc_hours']:+g} h", flush=True)
    plot_quality(metrics, sample, out_dir)
    print(f"[OK] key-variable quality outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
