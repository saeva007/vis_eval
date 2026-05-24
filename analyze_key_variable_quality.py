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


DEFAULT_QUALITY_FEATURES = "RH2M,Q_1000,DP_1000,RH_925,PRECIP"

FEATURE_ALIASES = {
    "RH2M": "RH2M",
    "RH_2M": "RH2M",
    "RH2": "RH2M",
    "Q1000": "Q_1000",
    "Q_1000": "Q_1000",
    "DP1000": "DP_1000",
    "DPT1000": "DP_1000",
    "DP_1000": "DP_1000",
    "RH925": "RH_925",
    "R925": "RH_925",
    "RH_925": "RH_925",
    "PRECIP": "PRECIP",
    "PRATE": "PRECIP",
    "PRATE_SFC": "PRECIP",
}

FEATURE_META = {
    "RH2M": {
        "obs": "rhu",
        "label": "2 m RH",
        "unit": "%",
        "extreme": "high",
        "threshold": 90.0,
        "valid": (0.0, 100.0),
    },
    "Q_1000": {
        "obs": None,
        "label": "1000 hPa specific humidity",
        "unit": "g kg$^{-1}$",
        "extreme": "high",
        "threshold": None,
        "valid": (0.0, 40.0),
    },
    "DP_1000": {
        "obs": None,
        "label": "1000 hPa dew point",
        "unit": "degC",
        "extreme": "high",
        "threshold": None,
        "valid": (-90.0, 60.0),
    },
    "RH_925": {
        "obs": None,
        "label": "925 hPa RH",
        "unit": "%",
        "extreme": "high",
        "threshold": 90.0,
        "valid": (0.0, 100.0),
    },
    "T2M": {
        "obs": "tem",
        "label": "2 m temperature",
        "unit": "degC",
        "extreme": "low",
        "threshold": None,
        "valid": (-80.0, 60.0),
    },
    "WSPD10": {
        "obs": "win_s_avg_10mi",
        "label": "10 m wind speed",
        "unit": "m s$^{-1}$",
        "extreme": "high",
        "threshold": None,
        "valid": (0.0, 80.0),
    },
    "MSLP": {
        "obs": "prs_sea",
        "label": "Sea-level pressure",
        "unit": "hPa",
        "extreme": "high",
        "threshold": None,
        "valid": (800.0, 1100.0),
    },
    "PRECIP": {
        "obs": "pre_1h",
        "label": "Hourly precipitation",
        "unit": "mm h$^{-1}$",
        "extreme": "high",
        "threshold": 0.1,
        "valid": (0.0, 500.0),
    },
}

OBS_VAR_MAP = {k: v for k, v in FEATURE_META.items() if v.get("obs")}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Key-variable forecast quality: Tianji vs IFS vs station observations.")
    ap.add_argument("--tianji_data_dir", required=True)
    ap.add_argument("--ifs_data_dir", required=True)
    ap.add_argument("--obs_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--features", default=DEFAULT_QUALITY_FEATURES)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=27)
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--timezone_probe_files", type=int, default=96)
    ap.add_argument("--min_pairs", type=int, default=100)
    return ap.parse_args()


def split_list(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value or "").split(";"):
        for raw in chunk.split(","):
            name = raw.strip()
            if not name:
                continue
            key = name.upper().replace("-", "_").replace(" ", "_")
            canon = FEATURE_ALIASES.get(key, name)
            if canon not in out:
                out.append(canon)
    return out


def feature_info(feature: str) -> Dict[str, object]:
    return FEATURE_META.get(
        feature,
        {
            "obs": None,
            "label": feature,
            "unit": "",
            "extreme": "high",
            "threshold": None,
            "valid": None,
        },
    )


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
    valid = feature_info(feature).get("valid")
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
    keep_cols = ["time", "station_key"] + [c for c in sorted({v["obs"] for v in OBS_VAR_MAP.values() if v.get("obs")}) if c in df]
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
    finite = arr[np.isfinite(arr)]
    med = float(np.nanmedian(finite)) if finite.size else math.nan
    if feature in {"T2M", "T_925", "DP_1000", "DP_925"} and math.isfinite(med) and med > 150:
        arr = arr - 273.15
    if feature in {"Q_1000", "Q_925"} and math.isfinite(med) and med < 0.2:
        arr = arr * 1000.0
    if feature in {"MSLP"} and math.isfinite(med) and np.nanmedian(np.abs(finite)) > 2000:
        arr = arr / 100.0
    if feature == "PRECIP":
        arr = np.maximum(arr, 0.0)
    if feature in {"RH2M", "RH_925"}:
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


def tail_threshold(feature: str, tianji: np.ndarray, ifs: np.ndarray, obs: Optional[np.ndarray] = None) -> Tuple[float, str]:
    info = feature_info(feature)
    fixed = info.get("threshold")
    if fixed is not None:
        return float(fixed), "fixed_physical_threshold"
    reference = obs if obs is not None and np.isfinite(obs).sum() >= 100 else None
    basis = "observation_p90"
    if reference is None:
        reference = np.concatenate([tianji[np.isfinite(tianji)], ifs[np.isfinite(ifs)]])
        basis = "paired_forecast_p90"
    if reference.size == 0:
        return math.nan, basis
    return float(np.nanpercentile(reference, 90.0)), basis


def paired_distribution_row(
    feature: str,
    tianji: np.ndarray,
    ifs: np.ndarray,
    obs: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    tianji = np.asarray(tianji, dtype=np.float64)
    ifs = np.asarray(ifs, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64) if obs is not None else None
    mask = np.isfinite(tianji) & np.isfinite(ifs)
    if obs_arr is not None:
        obs_arr = obs_arr[mask]
    tj = tianji[mask]
    ii = ifs[mask]
    if tj.size == 0:
        return {"feature": feature, "n_pair": 0}
    diff = ii - tj
    q_grid = np.array([50, 75, 90, 95, 97, 99], dtype=float)
    threshold, threshold_basis = tail_threshold(feature, tj, ii, obs_arr)
    direction = str(feature_info(feature).get("extreme", "high"))

    def tail_rate(values: np.ndarray) -> float:
        if not math.isfinite(threshold):
            return math.nan
        return float(np.mean(values <= threshold) if direction == "low" else np.mean(values >= threshold))

    row: Dict[str, object] = {
        "feature": feature,
        "label": feature_info(feature).get("label", feature),
        "unit": feature_info(feature).get("unit", ""),
        "n_pair": int(tj.size),
        "paired_bias_ifs_minus_tianji": float(np.nanmean(diff)),
        "paired_mae_ifs_minus_tianji": float(np.nanmean(np.abs(diff))),
        "paired_corr": float(np.corrcoef(tj, ii)[0, 1]) if tj.size >= 3 and np.nanstd(tj) > 0 and np.nanstd(ii) > 0 else math.nan,
        "threshold": threshold,
        "threshold_basis": threshold_basis,
        "extreme_type": direction,
        "tail_rate_tianji": tail_rate(tj),
        "tail_rate_ifs": tail_rate(ii),
        "tail_rate_ratio_ifs_over_tianji": float(tail_rate(ii) / tail_rate(tj)) if tail_rate(tj) and math.isfinite(tail_rate(tj)) else math.nan,
    }
    if obs_arr is not None and np.isfinite(obs_arr).sum() > 0:
        oo = obs_arr[np.isfinite(obs_arr)]
        row["tail_rate_obs"] = tail_rate(oo)
    for q in q_grid:
        row[f"tianji_p{int(q)}"] = float(np.nanpercentile(tj, q))
        row[f"ifs_p{int(q)}"] = float(np.nanpercentile(ii, q))
        row[f"ifs_minus_tianji_p{int(q)}"] = float(np.nanpercentile(ii, q) - np.nanpercentile(tj, q))
        if obs_arr is not None and np.isfinite(obs_arr).sum() > 0:
            row[f"obs_p{int(q)}"] = float(np.nanpercentile(obs_arr[np.isfinite(obs_arr)], q))
    return row


def make_quality_tables(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    tianji_dir = Path(args.tianji_data_dir)
    ifs_dir = Path(args.ifs_data_dir)
    obs_root = Path(args.obs_root)
    t_meta = read_meta(tianji_dir, args.limit_samples)
    i_meta = read_meta(ifs_dir, args.limit_samples)
    idx_t, idx_i, meta = align_meta(t_meta, i_meta)
    best_shift, obs, tz_diag = choose_obs_time_shift(obs_root, meta, args.timezone_probe_files)
    merged = meta.merge(obs, on=["time", "station_key"], how="left", suffixes=("", "_obs"))
    obs_columns = [v["obs"] for v in OBS_VAR_MAP.values() if v.get("obs") in merged]
    matched_obs = int(merged[obs_columns].notna().any(axis=1).sum()) if obs_columns else 0

    feature_lookup = {item["feature"]: i for i, item in enumerate(dynamic_features_for_count(args.dyn_vars_count))}
    features = split_list(args.features)
    t_available = available_dynamic_features(tianji_dir, args.dyn_vars_count)
    i_available = available_dynamic_features(ifs_dir, args.dyn_vars_count)
    x_t = np.load(tianji_dir / "X_test.npy", mmap_mode="r")
    x_i = np.load(ifs_dir / "X_test.npy", mmap_mode="r")
    rows = []
    dist_rows = []
    skipped_features = []
    sample = merged[["time", "station_key", "idx_tianji", "idx_ifs"]].copy()
    for feature in features:
        if feature not in feature_lookup:
            print(f"[skip] feature={feature} lacks dynamic index.", flush=True)
            skipped_features.append({"feature": feature, "reason": "missing_dynamic_index"})
            continue
        missing_sources = [name for name, available in (("tianji", t_available), ("ifs", i_available)) if feature not in available]
        if missing_sources:
            print(
                f"[skip] feature={feature} is not populated in {','.join(missing_sources)} overlap dataset(s).",
                flush=True,
            )
            skipped_features.append({"feature": feature, "reason": "not_populated", "sources": missing_sources})
            continue
        col = (int(args.window) - 1) * int(args.dyn_vars_count) + int(feature_lookup[feature])
        tj_vals = convert_forecast_units(feature, np.asarray(x_t[idx_t, col], dtype=np.float64))
        ifs_vals = convert_forecast_units(feature, np.asarray(x_i[idx_i, col], dtype=np.float64))
        sample[f"tianji_{feature}"] = tj_vals
        sample[f"ifs_{feature}"] = ifs_vals
        obs_vals: Optional[np.ndarray] = None
        info = feature_info(feature)
        obs_col = info.get("obs")
        if obs_col and obs_col in merged:
            obs_vals = clean_physical_values(feature, pd.to_numeric(merged[str(obs_col)], errors="coerce").to_numpy(dtype=float))
            if feature == "PRECIP":
                obs_vals = np.maximum(obs_vals, 0.0)
            rows.append(metric_row(feature, "tianji", tj_vals, obs_vals, str(info["extreme"]), info.get("threshold")))
            rows.append(metric_row(feature, "ifs", ifs_vals, obs_vals, str(info["extreme"]), info.get("threshold")))
            sample[f"obs_{feature}"] = obs_vals
        elif obs_col:
            print(f"[obs-skip] obs column {obs_col!r} missing for feature={feature}; keeping paired source distribution.", flush=True)
        dist_rows.append(paired_distribution_row(feature, tj_vals, ifs_vals, obs_vals))
    metrics = pd.DataFrame(rows)
    dist_metrics = pd.DataFrame(dist_rows)
    diag = {
        "paired_rows": int(len(meta)),
        "matched_obs_any_variable": matched_obs,
        "obs_time_shift_to_utc_hours": float(best_shift),
        "obs_time_interpretation": "raw_obs_time_is_bjt" if best_shift == -8.0 else "raw_obs_time_is_utc",
        "features": features,
        "evaluated_features": sorted(metrics["feature"].dropna().unique().tolist()) if not metrics.empty else [],
        "distribution_features": sorted(dist_metrics["feature"].dropna().unique().tolist()) if not dist_metrics.empty else [],
        "skipped_features": skipped_features,
        "precipitation_note": "Tianji PRECIP is expected to have been converted from accumulated amount to hourly increments during dataset building; IFS PRECIP is treated as an hourly amount/rate.",
        "primary_tail_feature": "RH2M",
    }
    return metrics, dist_metrics, sample, tz_diag, diag


def _tail_curve(values: np.ndarray, thresholds: np.ndarray, direction: str) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.full(len(thresholds), np.nan, dtype=float)
    if direction == "low":
        return np.asarray([100.0 * np.mean(vals <= th) for th in thresholds], dtype=float)
    return np.asarray([100.0 * np.mean(vals >= th) for th in thresholds], dtype=float)


def _column_values(sample: pd.DataFrame, column: str) -> np.ndarray:
    if column not in sample:
        return np.asarray([], dtype=float)
    return pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=float)


def plot_quality(metrics: pd.DataFrame, dist_metrics: pd.DataFrame, sample: pd.DataFrame, out_dir: Path) -> None:
    if dist_metrics.empty:
        return
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
            "axes.axisbelow": True,
        }
    )
    label_map = {k: str(v["label"]) for k, v in FEATURE_META.items()}
    unit_map = {k: str(v.get("unit", "")) for k, v in FEATURE_META.items()}
    colors = {"obs": "#111111", "tianji": "#1F5A7A", "ifs": "#8A8A8A"}
    features = [str(f) for f in dist_metrics["feature"].dropna().astype(str).tolist()]
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.3), gridspec_kw={"height_ratios": [1.05, 1.0]})

    # Hero evidence: can the forecast source recover the near-saturated RH2M tail?
    rh_feature = "RH2M" if "RH2M" in features else features[0]
    info = feature_info(rh_feature)
    direction = str(info.get("extreme", "high"))
    thresholds = np.arange(60, 101, 2, dtype=float) if rh_feature == "RH2M" else np.linspace(
        float(dist_metrics.loc[dist_metrics["feature"] == rh_feature, "tianji_p50"].iloc[0]),
        float(dist_metrics.loc[dist_metrics["feature"] == rh_feature, "tianji_p99"].iloc[0]),
        24,
    )
    tail_rows = pd.DataFrame({"threshold": thresholds})
    ax = axes[0, 0]
    for key, label, ls, lw in [
        ("obs", "Observed", "-", 2.2),
        ("tianji", "Tianji", "-", 2.0),
        ("ifs", "IFS", "--", 2.0),
    ]:
        vals = _column_values(sample, f"{key}_{rh_feature}")
        if vals.size == 0:
            continue
        curve = _tail_curve(vals, thresholds, direction)
        tail_rows[label.lower()] = curve
        ax.plot(thresholds, curve, color=colors[key], ls=ls, lw=lw, label=label)
    tail_rows.to_csv(out_dir / f"{rh_feature.lower()}_tail_curve.csv", index=False, float_format="%.6f")
    relation = "below" if direction == "low" else "above"
    ax.set_xlabel(f"{label_map.get(rh_feature, rh_feature)} threshold ({unit_map.get(rh_feature, '')})")
    ax.set_ylabel(f"Samples {relation} threshold (%)")
    ax.set_title(f"{label_map.get(rh_feature, rh_feature)} long-tail recovery")
    ax.legend(frameon=False)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[0, 1]
    quantiles = np.arange(50, 100, 2, dtype=float)
    quant_rows = pd.DataFrame({"percentile": quantiles})
    for key, label, ls, lw in [
        ("obs", "Observed", "-", 2.2),
        ("tianji", "Tianji", "-", 2.0),
        ("ifs", "IFS", "--", 2.0),
    ]:
        vals = _column_values(sample, f"{key}_{rh_feature}")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        qv = np.nanpercentile(vals, quantiles)
        quant_rows[label.lower()] = qv
        ax.plot(quantiles, qv, color=colors[key], ls=ls, lw=lw, label=label)
    quant_rows.to_csv(out_dir / f"{rh_feature.lower()}_upper_quantiles.csv", index=False, float_format="%.6f")
    ax.axvspan(90, 99, color="#CBD5E1", alpha=0.22, lw=0)
    ax.set_xlabel("Percentile")
    ax.set_ylabel(f"{label_map.get(rh_feature, rh_feature)} ({unit_map.get(rh_feature, '')})")
    ax.set_title("Upper-quantile structure")
    ax.legend(frameon=False)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1, 0]
    x = np.arange(len(features))
    for j, (source, label) in enumerate([("tianji", "Tianji"), ("ifs", "IFS")]):
        vals = pd.to_numeric(dist_metrics[f"tail_rate_{source}"], errors="coerce").to_numpy(dtype=float) * 100.0
        ax.bar(x + (j - 0.5) * width, vals, width * 0.92, color=colors[source], label=label)
    if "tail_rate_obs" in dist_metrics:
        obs_vals = pd.to_numeric(dist_metrics["tail_rate_obs"], errors="coerce").to_numpy(dtype=float) * 100.0
        ok = np.isfinite(obs_vals)
        ax.scatter(x[ok], obs_vals[ok], s=26, color=colors["obs"], zorder=5, label="Observed")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(f, f) for f in features], rotation=24, ha="right")
    ax.set_ylabel("Tail frequency (%)")
    ax.set_title("Critical-tail frequency across shared variables")
    ax.legend(frameon=False, ncol=3)
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1, 1]
    delta = pd.to_numeric(dist_metrics.get("ifs_minus_tianji_p90", np.nan), errors="coerce").to_numpy(dtype=float)
    y = np.arange(len(features))
    ax.barh(y, delta, color=["#B85C48" if v < 0 else "#2F855A" for v in delta])
    ax.axvline(0, color="#222222", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(f, f) for f in features])
    ax.invert_yaxis()
    ax.set_xlabel("IFS - Tianji at P90 (native units)")
    ax.set_title("High-quantile displacement")
    ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg", "tiff"):
        path = out_dir / f"fig_key_variable_quality_tianji_vs_ifs.{ext}"
        fig.savefig(path, dpi=600 if ext in {"png", "tiff"} else 300, bbox_inches="tight")
        print(f"[figure] {path}", flush=True)
    plt.close(fig)

    if rh_feature == "RH2M":
        fig2, ax2 = plt.subplots(figsize=(5.2, 3.8))
        for key, label, ls, lw in [
            ("obs", "Observed", "-", 2.4),
            ("tianji", "Tianji", "-", 2.2),
            ("ifs", "IFS", "--", 2.2),
        ]:
            vals = _column_values(sample, f"{key}_{rh_feature}")
            if vals.size:
                ax2.plot(thresholds, _tail_curve(vals, thresholds, "high"), color=colors[key], ls=ls, lw=lw, label=label)
        ax2.axvline(90, color="#475569", lw=0.9, ls=":")
        ax2.set_xlabel("RH2M threshold (%)")
        ax2.set_ylabel("Samples >= threshold (%)")
        ax2.set_title("Near-saturation tail")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        for ext in ("png", "pdf", "svg", "tiff"):
            path = out_dir / f"fig_rh2m_tail_tianji_vs_ifs.{ext}"
            fig2.savefig(path, dpi=600 if ext in {"png", "tiff"} else 300, bbox_inches="tight")
            print(f"[figure] {path}", flush=True)
        plt.close(fig2)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, dist_metrics, sample, tz_diag, diag = make_quality_tables(args)
    metrics_path = out_dir / "key_variable_quality_metrics.csv"
    dist_path = out_dir / "key_variable_distribution_metrics.csv"
    sample_path = out_dir / "key_variable_quality_samples.csv"
    tz_path = out_dir / "observation_time_alignment_diagnostics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    dist_metrics.to_csv(dist_path, index=False, float_format="%.6f")
    sample.to_csv(sample_path, index=False, float_format="%.6f")
    tz_diag.to_csv(tz_path, index=False)
    with open(out_dir / "key_variable_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[table] {metrics_path}", flush=True)
    print(f"[table] {dist_path}", flush=True)
    print(f"[table] {sample_path}", flush=True)
    print(f"[time] {diag['obs_time_interpretation']} shift={diag['obs_time_shift_to_utc_hours']:+g} h", flush=True)
    plot_quality(metrics, dist_metrics, sample, out_dir)
    print(f"[OK] key-variable quality outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
