#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q1000-focused mechanism analysis for source-full and q-core experiments.

This script deliberately treats ERA5 as a reference analysis, not truth. It
never derives RH2M from Pangu Q_1000; it only compares Q_1000, DP_1000, and
vertical humidity-structure diagnostics that are present in dynamic inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FINAL_FEATURE_ORDER: List[str] = [
    "RH2M",
    "T2M",
    "PRECIP",
    "MSLP",
    "SW_RAD",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "CAPE",
    "LCC",
    "T_925",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "W_925",
    "W_1000",
    "DPD",
    "INVERSION",
]

Q_CORE_NO_RH2M_ORDER: List[str] = [
    "T2M",
    "MSLP",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "ZENITH",
    "PM10_ugm3",
    "PM25_ugm3",
]

DEFAULT_FEATURES = "Q_1000,DP_1000,Q_925,Q1000_MINUS_Q925"

FEATURE_ALIASES = {
    "Q1000": "Q_1000",
    "Q_1000": "Q_1000",
    "Q925": "Q_925",
    "Q_925": "Q_925",
    "DP1000": "DP_1000",
    "DP_1000": "DP_1000",
    "DPT1000": "DP_1000",
    "DP925": "DP_925",
    "DP_925": "DP_925",
    "Q1000_MINUS_Q925": "Q1000_MINUS_Q925",
    "Q_1000_MINUS_Q_925": "Q1000_MINUS_Q925",
    "Q1000_Q925_DIFF": "Q1000_MINUS_Q925",
    "Q_1000_Q_925_DIFF": "Q1000_MINUS_Q925",
}

FEATURE_META = {
    "Q_1000": {"label": "Q1000", "unit": "g kg-1", "valid": (0.0, 40.0)},
    "Q_925": {"label": "Q925", "unit": "g kg-1", "valid": (0.0, 40.0)},
    "DP_1000": {"label": "DP1000", "unit": "degC", "valid": (-90.0, 60.0)},
    "DP_925": {"label": "DP925", "unit": "degC", "valid": (-90.0, 60.0)},
    "Q1000_MINUS_Q925": {"label": "Q1000-Q925", "unit": "g kg-1", "valid": (-30.0, 30.0)},
}

SOURCE_STYLE = {
    "tianji": {"color": "#2878A5", "marker": "o"},
    "ifs": {"color": "#C47A1D", "marker": "^"},
    "era5": {"color": "#6C7280", "marker": "D"},
    "pangu": {"color": "#7651A8", "marker": "P"},
}

CASE_LABELS = {
    "numerical_hit_pangu_miss": "Numerical hit\nPangu miss",
    "both_hit": "Both hit",
    "both_miss": "Both miss",
    "pangu_hit_numerical_miss": "Pangu hit\nNumerical miss",
}


def source_style(tag: str, label: str = "") -> Dict[str, str]:
    text = f"{tag} {label}".lower()
    for family in ("pangu", "tianji", "era5", "ifs"):
        if family in text:
            return SOURCE_STYLE[family]
    return SOURCE_STYLE["era5"]


@dataclass
class SourceSpec:
    tag: str
    data_dir: Path
    label: str
    group: str = ""


@dataclass
class SourceData:
    spec: SourceSpec
    cfg: Dict[str, object]
    dynamic_order: List[str]
    window: int
    dyn_vars: int
    rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compare Q1000/DP1000 quality and low-visibility informativeness "
            "across Tianji, IFS, ERA5 reference analysis, and Pangu."
        )
    )
    ap.add_argument(
        "--sources",
        required=True,
        help=(
            "Semicolon-separated source specs: tag=/path/to/data|Label|Group. "
            "Group is optional."
        ),
    )
    ap.add_argument("--reference_source", default="era5_2025_source_full")
    ap.add_argument("--label_source", default="", help="Source whose y_test labels define low-vis; default=reference_source.")
    ap.add_argument("--pangu_tag", default="pangu2025_source_full")
    ap.add_argument("--numerical_tags", default="", help="Comma list for case-control; default=all non-Pangu sources.")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_dir", default="", help="Optional directory with per_sample_<tag>.csv files.")
    ap.add_argument(
        "--require_case_control",
        action="store_true",
        help="Fail with the expected per-sample filenames when case-control inputs are unavailable or do not match.",
    )
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--low_vis_threshold_m", type=float, default=1000.0)
    ap.add_argument("--bootstrap_iters", type=int, default=0)
    ap.add_argument("--bootstrap_seed", type=int, default=20260626)
    ap.add_argument("--min_pairs", type=int, default=100)
    return ap.parse_args()


def canonical_feature(name: str) -> str:
    key = str(name or "").strip().upper().replace("-", "_").replace(" ", "_")
    return FEATURE_ALIASES.get(key, key)


def split_features(value: str) -> List[str]:
    out: List[str] = []
    for raw in str(value or "").replace(";", ",").split(","):
        feat = canonical_feature(raw)
        if feat and feat not in out:
            out.append(feat)
    return out


def parse_sources(value: str) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    for raw in str(value or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if "=" in raw:
            tag, rest = raw.split("=", 1)
            tag = tag.strip()
        else:
            rest = raw
            tag = Path(rest.split("|", 1)[0].strip()).name
        parts = [p.strip() for p in rest.split("|")]
        data_dir = Path(parts[0])
        label = parts[1] if len(parts) > 1 and parts[1] else tag
        group = parts[2] if len(parts) > 2 else ""
        if not tag:
            raise ValueError(f"Bad source spec: {raw!r}")
        specs.append(SourceSpec(tag=tag, data_dir=data_dir, label=label, group=group))
    if not specs:
        raise ValueError("--sources is empty")
    tags = [s.tag for s in specs]
    dup = sorted({t for t in tags if tags.count(t) > 1})
    if dup:
        raise ValueError(f"Duplicate source tag(s): {dup}")
    t2nd = [s.tag for s in specs if "t2nd" in f"{s.tag} {s.label}".lower()]
    if t2nd:
        raise ValueError(
            "Tianji T2ND differs from Tianji only through RH2M and must not be "
            f"duplicated in Q1000 analysis. Remove source(s): {t2nd}."
        )
    return specs


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_station_ids(values: Iterable[object]) -> pd.Series:
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
    df["_key"] = make_key(df)
    return df


def make_key(df: pd.DataFrame) -> pd.Series:
    time_ns = pd.to_datetime(df["time"], errors="coerce").astype("int64").astype(str)
    return time_ns + "|" + df["station_key"].astype(str) + "|" + df["dup"].astype(str)


def infer_dynamic_order(cfg: Dict[str, object], dyn_vars: int) -> List[str]:
    order = cfg.get("dynamic_feature_order")
    if isinstance(order, list) and len(order) == dyn_vars:
        return [str(v) for v in order]
    feature_set = str(cfg.get("feature_set") or "").lower()
    if feature_set == "q_core_no_rh2m" and dyn_vars == len(Q_CORE_NO_RH2M_ORDER):
        return list(Q_CORE_NO_RH2M_ORDER)
    if dyn_vars == 27:
        return [*FINAL_FEATURE_ORDER, "ZENITH", "PM10_ugm3", "PM25_ugm3"]
    if dyn_vars == 24:
        return [
            "RH2M",
            "T2M",
            "PRECIP",
            "MSLP",
            "SW_RAD",
            "U10",
            "WSPD10",
            "V10",
            "WDIR10",
            "LCC",
            "RH_925",
            "U_925",
            "WSPD925",
            "V_925",
            "DP_1000",
            "DP_925",
            "Q_1000",
            "Q_925",
            "W_925",
            "W_1000",
            "DPD",
            "ZENITH",
            "PM10_ugm3",
            "PM25_ugm3",
        ]
    if dyn_vars == 19:
        return [
            "T2M",
            "MSLP",
            "U10",
            "WSPD10",
            "V10",
            "WDIR10",
            "T_925",
            "RH_925",
            "U_925",
            "WSPD925",
            "V_925",
            "DP_1000",
            "DP_925",
            "Q_1000",
            "Q_925",
            "INVERSION",
            "ZENITH",
            "PM10_ugm3",
            "PM25_ugm3",
        ]
    raise ValueError(
        "dataset_build_config.json lacks usable dynamic_feature_order and "
        f"dyn_vars={dyn_vars} has no safe fallback."
    )


def load_y(data_dir: Path, n: int) -> np.ndarray:
    path = data_dir / "y_test.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing y_test.npy: {path}")
    y = np.load(path)
    y = np.asarray(y).reshape(-1)
    return y[:n]


def y_to_low_vis(y: np.ndarray, threshold_m: float) -> np.ndarray:
    arr = np.asarray(y)
    finite = np.isfinite(arr)
    uniques = np.unique(arr[finite]) if finite.any() else np.array([])
    if len(uniques) and len(uniques) <= 4 and set(np.round(uniques).astype(int)).issubset({0, 1, 2}):
        return finite & (arr.astype(float) <= 1.0)
    return finite & (arr.astype(float) <= float(threshold_m))


def convert_units(feature: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e6] = np.nan
    if feature in {"Q_1000", "Q_925"}:
        finite = arr[np.isfinite(arr)]
        if finite.size and np.nanmedian(np.abs(finite)) < 0.2:
            arr *= 1000.0
    if feature in {"DP_1000", "DP_925"}:
        finite = arr[np.isfinite(arr)]
        if finite.size and np.nanmedian(finite) > 150.0:
            arr -= 273.15
    valid = FEATURE_META.get(feature, {}).get("valid")
    if valid is not None:
        lo, hi = valid
        arr[(arr < float(lo)) | (arr > float(hi))] = np.nan
    return arr


def read_last_hour_features(
    data_dir: Path,
    dynamic_order: Sequence[str],
    window: int,
    dyn_vars: int,
    features: Sequence[str],
    limit_samples: int,
) -> Dict[str, np.ndarray]:
    path = data_dir / "X_test.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing X_test.npy: {path}")
    x = np.load(path, mmap_mode="r")
    n = int(x.shape[0])
    if limit_samples and limit_samples > 0:
        n = min(n, int(limit_samples))
    order_lookup = {canonical_feature(v): i for i, v in enumerate(dynamic_order)}
    needed_base = sorted(
        {
            f
            for f in features
            if f in {"Q_1000", "Q_925", "DP_1000", "DP_925"}
        }
        | ({"Q_1000", "Q_925"} if "Q1000_MINUS_Q925" in features else set())
    )
    raw_values: Dict[str, np.ndarray] = {}
    if x.ndim == 3:
        if x.shape[1] < window or x.shape[2] != dyn_vars:
            raise ValueError(f"{path} shape {x.shape} does not match window={window}, dyn_vars={dyn_vars}")
        for feat in needed_base:
            if feat not in order_lookup:
                raw_values[feat] = np.full(n, np.nan, dtype=np.float64)
                continue
            raw_values[feat] = np.asarray(x[:n, window - 1, order_lookup[feat]], dtype=np.float64)
    else:
        dyn_flat = int(window) * int(dyn_vars)
        if x.shape[1] < dyn_flat:
            raise ValueError(f"{path} has {x.shape[1]} columns, less than dynamic flat dim {dyn_flat}")
        for feat in needed_base:
            if feat not in order_lookup:
                raw_values[feat] = np.full(n, np.nan, dtype=np.float64)
                continue
            col = (int(window) - 1) * int(dyn_vars) + int(order_lookup[feat])
            raw_values[feat] = np.asarray(x[:n, col], dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for feat, values in raw_values.items():
        out[feat] = convert_units(feat, values)
    if "Q1000_MINUS_Q925" in features:
        q1000 = out.get("Q_1000", np.full(n, np.nan))
        q925 = out.get("Q_925", np.full(n, np.nan))
        out["Q1000_MINUS_Q925"] = convert_units("Q1000_MINUS_Q925", q1000 - q925)
    for feat in features:
        out.setdefault(feat, np.full(n, np.nan, dtype=np.float64))
    return out


def load_source(spec: SourceSpec, features: Sequence[str], limit_samples: int, low_vis_threshold: float) -> SourceData:
    cfg = read_json(spec.data_dir / "dataset_build_config.json")
    meta = read_meta(spec.data_dir, limit_samples)
    y = load_y(spec.data_dir, len(meta))
    meta["y_raw"] = y
    meta["low_vis"] = y_to_low_vis(y, low_vis_threshold)
    dyn_vars = int(cfg.get("dyn_vars") or cfg.get("dyn_vars_count") or 0)
    if dyn_vars <= 0:
        x = np.load(spec.data_dir / "X_test.npy", mmap_mode="r")
        window = int(cfg.get("window") or 12)
        dyn_vars = int((x.shape[1] - int(cfg.get("fe_dim", 0))) // window) if x.ndim == 2 else int(x.shape[-1])
    window = int(cfg.get("window") or 12)
    dynamic_order = infer_dynamic_order(cfg, dyn_vars)
    values = read_last_hour_features(spec.data_dir, dynamic_order, window, dyn_vars, features, limit_samples)
    for feat, arr in values.items():
        meta[feat] = arr[: len(meta)]
    return SourceData(spec=spec, cfg=cfg, dynamic_order=dynamic_order, window=window, dyn_vars=dyn_vars, rows=meta)


def season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def common_key_subset(sources: Dict[str, SourceData]) -> List[str]:
    key_sets = [set(src.rows["_key"]) for src in sources.values()]
    common = set.intersection(*key_sets)
    if not common:
        raise RuntimeError("No common (valid time, station, duplicate index) rows across selected sources.")
    return sorted(common)


def filter_common(src: SourceData, common_keys: Sequence[str]) -> pd.DataFrame:
    key_order = pd.DataFrame({"_key": list(common_keys), "_common_order": np.arange(len(common_keys), dtype=np.int64)})
    df = src.rows.merge(key_order, on="_key", how="inner")
    df = df.sort_values("_common_order").reset_index(drop=True)
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["season"] = df["month"].map(season_from_month)
    return df


def finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    return aa[mask], bb[mask]


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return math.nan
    if np.nanstd(a) <= 0 or np.nanstd(b) <= 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def wasserstein_quantile(a: np.ndarray, b: np.ndarray, qn: int = 201) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    qs = np.linspace(0.0, 100.0, int(qn))
    return float(np.nanmean(np.abs(np.nanpercentile(a, qs) - np.nanpercentile(b, qs))))


def quality_metric_dict(src: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    a, b = finite_pair(src, ref)
    if len(a) == 0:
        return {
            "n": 0,
            "bias": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "corr": math.nan,
            "wasserstein": math.nan,
            "p90_source": math.nan,
            "p90_reference": math.nan,
            "p90_delta": math.nan,
            "p95_source": math.nan,
            "p95_reference": math.nan,
            "p95_delta": math.nan,
            "p99_source": math.nan,
            "p99_reference": math.nan,
            "p99_delta": math.nan,
        }
    diff = a - b
    out = {
        "n": int(len(a)),
        "bias": float(np.nanmean(diff)),
        "mae": float(np.nanmean(np.abs(diff))),
        "rmse": float(np.sqrt(np.nanmean(diff * diff))),
        "corr": corr_safe(a, b),
        "wasserstein": wasserstein_quantile(a, b),
    }
    for q in (90, 95, 99):
        src_q = float(np.nanpercentile(a, q))
        ref_q = float(np.nanpercentile(b, q))
        out[f"p{q}_source"] = src_q
        out[f"p{q}_reference"] = ref_q
        out[f"p{q}_delta"] = src_q - ref_q
    return out


def add_quality_date_bootstrap(
    row: Dict[str, object],
    rng: np.random.Generator,
    src: np.ndarray,
    ref: np.ndarray,
    dates: np.ndarray,
    iters: int,
) -> None:
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    d = np.asarray(dates)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b, d = a[finite], b[finite], d[finite]
    if iters <= 0 or len(a) < 5:
        return
    diff = a - b
    codes, unique_dates = pd.factorize(pd.Series(d).astype(str), sort=True)
    n_dates = len(unique_dates)
    if n_dates < 5:
        return
    daily = np.column_stack(
        [
            np.bincount(codes, minlength=n_dates),
            np.bincount(codes, weights=diff, minlength=n_dates),
            np.bincount(codes, weights=np.abs(diff), minlength=n_dates),
            np.bincount(codes, weights=diff * diff, minlength=n_dates),
        ]
    )
    values = {"bias": [], "mae": [], "rmse": []}
    for _ in range(int(iters)):
        n, err_sum, abs_sum, sq_sum = daily[rng.integers(0, n_dates, size=n_dates)].sum(axis=0)
        values["bias"].append(err_sum / n)
        values["mae"].append(abs_sum / n)
        values["rmse"].append(np.sqrt(sq_sum / n))
    for name, draws in values.items():
        arr = np.asarray(draws, dtype=np.float64)
        row[f"{name}_ci_low"] = float(np.nanpercentile(arr, 2.5))
        row[f"{name}_ci_high"] = float(np.nanpercentile(arr, 97.5))
    # Marginal percentile displacement is descriptive. Extreme-event uncertainty
    # is handled by the date-block contingency analysis below.
    for name in ("p95_delta", "p99_delta"):
        row[f"{name}_ci_low"] = math.nan
        row[f"{name}_ci_high"] = math.nan


def reference_quality_tables(
    common: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    ref_tag: str,
    features: Sequence[str],
    min_pairs: int,
    bootstrap_iters: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ref_tag not in common:
        raise KeyError(f"reference_source={ref_tag!r} not found in sources")
    rng = np.random.default_rng(seed)
    ref_df = common[ref_tag]
    rows: List[Dict[str, object]] = []
    tail_rows: List[Dict[str, object]] = []
    scopes: List[Tuple[str, np.ndarray]] = [("all", np.ones(len(ref_df), dtype=bool))]
    for month in sorted(ref_df["month"].dropna().unique()):
        scopes.append((f"month_{int(month):02d}", (ref_df["month"].to_numpy() == int(month))))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        scopes.append((f"season_{season}", (ref_df["season"].to_numpy() == season)))

    for tag, df in common.items():
        if tag == ref_tag:
            continue
        for feat in features:
            if feat not in df or feat not in ref_df:
                continue
            for scope, mask in scopes:
                src_v = df.loc[mask, feat].to_numpy(dtype=np.float64)
                ref_v = ref_df.loc[mask, feat].to_numpy(dtype=np.float64)
                metrics = quality_metric_dict(src_v, ref_v)
                if metrics["n"] < min_pairs:
                    continue
                row: Dict[str, object] = {
                    "source": tag,
                    "source_label": labels.get(tag, tag),
                    "reference_source": ref_tag,
                    "reference_label": labels.get(ref_tag, ref_tag),
                    "feature": feat,
                    "feature_label": FEATURE_META.get(feat, {}).get("label", feat),
                    "unit": FEATURE_META.get(feat, {}).get("unit", ""),
                    "scope": scope,
                    **metrics,
                }
                if scope == "all" and bootstrap_iters > 0:
                    scope_dates = pd.to_datetime(ref_df.loc[mask, "time"]).dt.strftime("%Y-%m-%d").to_numpy()
                    add_quality_date_bootstrap(row, rng, src_v, ref_v, scope_dates, bootstrap_iters)
                rows.append(row)

            for month in sorted(ref_df["month"].dropna().unique()):
                month_mask = ref_df["month"].to_numpy() == int(month)
                src_v = df.loc[month_mask, feat].to_numpy(dtype=np.float64)
                ref_v = ref_df.loc[month_mask, feat].to_numpy(dtype=np.float64)
                a, b = finite_pair(src_v, ref_v)
                if len(a) < min_pairs:
                    continue
                for q in (90, 95, 99):
                    threshold = float(np.nanpercentile(b, q))
                    ref_extreme = b >= threshold
                    src_extreme = a >= threshold
                    hit = int(np.sum(ref_extreme & src_extreme))
                    miss = int(np.sum(ref_extreme & ~src_extreme))
                    false = int(np.sum(~ref_extreme & src_extreme))
                    denom_ref = max(int(np.sum(ref_extreme)), 1)
                    denom_non = max(int(np.sum(~ref_extreme)), 1)
                    tail_rows.append(
                        {
                            "source": tag,
                            "source_label": labels.get(tag, tag),
                            "reference_source": ref_tag,
                            "feature": feat,
                            "month": int(month),
                            "quantile": q,
                            "n": int(len(a)),
                            "reference_threshold": threshold,
                            "reference_extreme_count": int(ref_extreme.sum()),
                            "non_reference_count": int((~ref_extreme).sum()),
                            "hit_count": hit,
                            "miss_count": miss,
                            "false_extreme_count": false,
                            "hit_rate": hit / denom_ref,
                            "miss_rate": miss / denom_ref,
                            "false_extreme_rate": false / denom_non,
                            "source_tail_mean": float(np.nanmean(a[src_extreme])) if src_extreme.any() else math.nan,
                            "reference_tail_mean": float(np.nanmean(b[ref_extreme])) if ref_extreme.any() else math.nan,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(tail_rows)


def summarize_reference_tail(tail: pd.DataFrame) -> pd.DataFrame:
    """Pool month-specific tail diagnostics without mixing seasonal Q1000 levels."""
    if tail.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    group_cols = ["source", "source_label", "reference_source", "feature", "quantile"]
    for keys, cur in tail.groupby(group_cols, dropna=False, sort=False):
        source, source_label, reference_source, feature, quantile = keys
        if {"hit_count", "miss_count", "false_extreme_count", "reference_extreme_count", "non_reference_count"}.issubset(cur.columns):
            hit = float(cur["hit_count"].sum())
            miss = float(cur["miss_count"].sum())
            false = float(cur["false_extreme_count"].sum())
            ref_count = float(cur["reference_extreme_count"].sum())
            non_count = float(cur["non_reference_count"].sum())
        else:
            ref_weight = cur["n"].to_numpy(dtype=float) * (1.0 - float(quantile) / 100.0)
            non_weight = cur["n"].to_numpy(dtype=float) - ref_weight
            hit = float(np.nansum(cur["hit_rate"].to_numpy(dtype=float) * ref_weight))
            miss = float(np.nansum(cur["miss_rate"].to_numpy(dtype=float) * ref_weight))
            false = float(np.nansum(cur["false_extreme_rate"].to_numpy(dtype=float) * non_weight))
            ref_count = float(np.nansum(ref_weight))
            non_count = float(np.nansum(non_weight))
        tail_bias = cur["source_tail_mean"].to_numpy(dtype=float) - cur["reference_tail_mean"].to_numpy(dtype=float)
        weights = cur.get("reference_extreme_count", cur["n"]).to_numpy(dtype=float)
        finite = np.isfinite(tail_bias) & np.isfinite(weights) & (weights > 0)
        rows.append(
            {
                "source": source,
                "source_label": source_label,
                "reference_source": reference_source,
                "feature": feature,
                "quantile": int(quantile),
                "n": int(cur["n"].sum()),
                "reference_extreme_count": int(round(ref_count)),
                "hit_rate": hit / ref_count if ref_count > 0 else math.nan,
                "miss_rate": miss / ref_count if ref_count > 0 else math.nan,
                "false_extreme_rate": false / non_count if non_count > 0 else math.nan,
                "source_minus_reference_tail_mean": (
                    float(np.average(tail_bias[finite], weights=weights[finite])) if finite.any() else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def contingency_scores(hit, miss, false, correct) -> Dict[str, np.ndarray]:
    """Return deterministic rare-event scores for scalar or array counts."""
    h = np.asarray(hit, dtype=np.float64)
    m = np.asarray(miss, dtype=np.float64)
    f = np.asarray(false, dtype=np.float64)
    c = np.asarray(correct, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pod = h / (h + m)
        pofd = f / (f + c)
        far = f / (h + f)
        csi = h / (h + m + f)
        frequency_bias = (h + f) / (h + m)
        total = h + m + f + c
        random_hit = (h + m) * (h + f) / total
        ets = (h - random_hit) / (h + m + f - random_hit)
    eps = 1.0e-12
    h_clip = np.clip(pod, eps, 1.0 - eps)
    f_clip = np.clip(pofd, eps, 1.0 - eps)
    numerator = np.log(f_clip) - np.log(h_clip) - np.log1p(-f_clip) + np.log1p(-h_clip)
    denominator = np.log(f_clip) + np.log(h_clip) + np.log1p(-f_clip) + np.log1p(-h_clip)
    with np.errstate(divide="ignore", invalid="ignore"):
        sedi = numerator / denominator
    valid_sedi = (pod > 0.0) & (pod < 1.0) & (pofd > 0.0) & (pofd < 1.0)
    sedi = np.where(valid_sedi, sedi, np.nan)
    sedi = np.where((pod == 1.0) & (pofd == 0.0), 1.0, sedi)
    sedi = np.where((pod == 0.0) & (pofd == 1.0), -1.0, sedi)
    return {
        "pod": pod,
        "pofd": pofd,
        "far": far,
        "csi": csi,
        "ets": ets,
        "sedi": sedi,
        "frequency_bias": frequency_bias,
    }


def _daily_extreme_sufficient_statistics(
    dates: np.ndarray,
    reference_event: np.ndarray,
    source_event: np.ndarray,
    error: np.ndarray,
) -> pd.DataFrame:
    day_codes, unique_days = pd.factorize(pd.Series(dates).astype(str), sort=True)
    n_days = len(unique_days)
    ref = np.asarray(reference_event, dtype=bool)
    src = np.asarray(source_event, dtype=bool)
    err = np.asarray(error, dtype=np.float64)

    def counts(mask: np.ndarray) -> np.ndarray:
        return np.bincount(day_codes, weights=mask.astype(np.float64), minlength=n_days)

    return pd.DataFrame(
        {
            "date": unique_days,
            "hit": counts(ref & src),
            "miss": counts(ref & ~src),
            "false": counts(~ref & src),
            "correct": counts(~ref & ~src),
            "reference_count": counts(ref),
            "error_sum": np.bincount(day_codes, weights=np.where(ref, err, 0.0), minlength=n_days),
            "abs_error_sum": np.bincount(day_codes, weights=np.where(ref, np.abs(err), 0.0), minlength=n_days),
            "sq_error_sum": np.bincount(day_codes, weights=np.where(ref, err * err, 0.0), minlength=n_days),
        }
    )


def _add_date_block_bootstrap(
    row: Dict[str, object],
    daily: pd.DataFrame,
    iters: int,
    seed: int,
) -> None:
    if iters <= 0 or len(daily) < 5:
        return
    rng = np.random.default_rng(seed)
    values = daily[[
        "hit", "miss", "false", "correct", "reference_count",
        "error_sum", "abs_error_sum", "sq_error_sum",
    ]].to_numpy(dtype=np.float64)
    n_days = len(values)
    boot: Dict[str, List[float]] = {
        name: [] for name in (
            "pod", "pofd", "far", "csi", "ets", "sedi", "frequency_bias",
            "conditional_bias", "conditional_mae", "conditional_rmse",
        )
    }
    for _ in range(int(iters)):
        sample = values[rng.integers(0, n_days, size=n_days)].sum(axis=0)
        h, m, f, c, n_ref, err_sum, abs_sum, sq_sum = sample
        scores = contingency_scores(h, m, f, c)
        for name in ("pod", "pofd", "far", "csi", "ets", "sedi", "frequency_bias"):
            boot[name].append(float(scores[name]))
        boot["conditional_bias"].append(float(err_sum / n_ref) if n_ref > 0 else math.nan)
        boot["conditional_mae"].append(float(abs_sum / n_ref) if n_ref > 0 else math.nan)
        boot["conditional_rmse"].append(float(np.sqrt(sq_sum / n_ref)) if n_ref > 0 else math.nan)
    for name, vals in boot.items():
        arr = np.asarray(vals, dtype=np.float64)
        row[f"{name}_ci_low"] = float(np.nanpercentile(arr, 2.5))
        row[f"{name}_ci_high"] = float(np.nanpercentile(arr, 97.5))


def extreme_spatiotemporal_table(
    common: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    ref_tag: str,
    features: Sequence[str],
    min_pairs: int,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    """Verify whether upper-tail moisture occurs at the same station and valid time.

    Exact-reference thresholds retain amplitude calibration. Quantile-matched
    thresholds equalize monthly event frequency and isolate rank/localization.
    Conditional errors on reference extremes are diagnostics only because
    outcome-conditioned ranking is vulnerable to the forecaster's dilemma.
    """
    if ref_tag not in common:
        raise KeyError(f"reference_source={ref_tag!r} not found in sources")
    ref_df = common[ref_tag]
    rows: List[Dict[str, object]] = []
    for source_i, (tag, df) in enumerate(common.items()):
        if tag == ref_tag:
            continue
        for feature_i, feat in enumerate(features):
            if feat not in df or feat not in ref_df:
                continue
            src_all = df[feat].to_numpy(dtype=np.float64)
            ref_all = ref_df[feat].to_numpy(dtype=np.float64)
            months_all = ref_df["month"].to_numpy(dtype=np.int64)
            dates_all = pd.to_datetime(ref_df["time"]).dt.strftime("%Y-%m-%d").to_numpy()
            finite = np.isfinite(src_all) & np.isfinite(ref_all)
            if int(finite.sum()) < min_pairs:
                continue
            src = src_all[finite]
            ref = ref_all[finite]
            months = months_all[finite]
            dates = dates_all[finite]
            error = src - ref
            for quantile in (90, 95, 99):
                reference_event = np.zeros(len(src), dtype=bool)
                exact_source_event = np.zeros(len(src), dtype=bool)
                matched_source_event = np.zeros(len(src), dtype=bool)
                ref_thresholds: List[float] = []
                src_thresholds: List[float] = []
                for month in sorted(np.unique(months)):
                    month_mask = months == month
                    ref_threshold = float(np.nanpercentile(ref[month_mask], quantile))
                    src_threshold = float(np.nanpercentile(src[month_mask], quantile))
                    reference_event[month_mask] = ref[month_mask] >= ref_threshold
                    exact_source_event[month_mask] = src[month_mask] >= ref_threshold
                    matched_source_event[month_mask] = src[month_mask] >= src_threshold
                    ref_thresholds.append(ref_threshold)
                    src_thresholds.append(src_threshold)

                for mode_i, (event_definition, source_event) in enumerate(
                    (
                        ("exact_reference_threshold", exact_source_event),
                        ("quantile_matched", matched_source_event),
                    )
                ):
                    hit = int(np.sum(reference_event & source_event))
                    miss = int(np.sum(reference_event & ~source_event))
                    false = int(np.sum(~reference_event & source_event))
                    correct = int(np.sum(~reference_event & ~source_event))
                    scores = contingency_scores(hit, miss, false, correct)
                    ref_error = error[reference_event]
                    row: Dict[str, object] = {
                        "source": tag,
                        "source_label": labels.get(tag, tag),
                        "reference_source": ref_tag,
                        "reference_label": labels.get(ref_tag, ref_tag),
                        "feature": feat,
                        "feature_label": FEATURE_META.get(feat, {}).get("label", feat),
                        "unit": FEATURE_META.get(feat, {}).get("unit", ""),
                        "quantile": int(quantile),
                        "threshold_scope": "calendar_month",
                        "event_definition": event_definition,
                        "n": int(len(src)),
                        "n_dates": int(pd.Series(dates).nunique()),
                        "reference_extreme_count": int(reference_event.sum()),
                        "source_extreme_count": int(source_event.sum()),
                        "hit_count": hit,
                        "miss_count": miss,
                        "false_alarm_count": false,
                        "correct_negative_count": correct,
                        "mean_monthly_reference_threshold": float(np.mean(ref_thresholds)),
                        "mean_monthly_source_threshold": float(np.mean(src_thresholds)),
                        "conditional_bias": float(np.mean(ref_error)),
                        "conditional_mae": float(np.mean(np.abs(ref_error))),
                        "conditional_rmse": float(np.sqrt(np.mean(ref_error * ref_error))),
                    }
                    for name, value in scores.items():
                        row[name] = float(value)
                    daily = _daily_extreme_sufficient_statistics(dates, reference_event, source_event, error)
                    _add_date_block_bootstrap(
                        row,
                        daily,
                        bootstrap_iters,
                        seed + source_i * 1000 + feature_i * 100 + quantile * 2 + mode_i,
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_pos = float(ranks[y].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float(np.sum(precision[y_sorted]) / n_pos)


def odds_ratio(y_true: np.ndarray, top: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    t = np.asarray(top, dtype=bool)
    a = float(np.sum(t & y))
    b = float(np.sum(t & ~y))
    c = float(np.sum(~t & y))
    d = float(np.sum(~t & ~y))
    return float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))


def top_decile_flags(score: np.ndarray, month: Optional[np.ndarray] = None) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    out = np.zeros(len(s), dtype=bool)
    finite = np.isfinite(s)
    if month is None:
        if finite.any():
            out = finite & (s >= np.nanpercentile(s[finite], 90))
        return out
    month_arr = np.asarray(month)
    for m in np.unique(month_arr[finite]):
        mask = finite & (month_arr == m)
        if mask.any():
            out[mask] = s[mask] >= np.nanpercentile(s[mask], 90)
    return out


def informativeness_metric_dict(score: np.ndarray, low_vis: np.ndarray, month: np.ndarray) -> Dict[str, float]:
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(low_vis, dtype=bool)
    mask = np.isfinite(s)
    if not mask.any():
        return {
            "n": 0,
            "low_vis_count": 0,
            "auc": math.nan,
            "ap": math.nan,
            "top_decile_lowvis_rate": math.nan,
            "background_lowvis_rate": math.nan,
            "enrichment_ratio": math.nan,
            "odds_ratio": math.nan,
            "monthly_top_decile_lowvis_rate": math.nan,
            "monthly_enrichment_ratio": math.nan,
            "monthly_odds_ratio": math.nan,
        }
    s = s[mask]
    y = y[mask]
    m = np.asarray(month)[mask]
    top = top_decile_flags(s)
    monthly_top = top_decile_flags(s, m)
    bg = float(np.mean(y)) if len(y) else math.nan
    top_rate = float(np.mean(y[top])) if top.any() else math.nan
    monthly_rate = float(np.mean(y[monthly_top])) if monthly_top.any() else math.nan
    return {
        "n": int(len(s)),
        "low_vis_count": int(y.sum()),
        "auc": auc_rank(y, s),
        "ap": average_precision(y, s),
        "top_decile_lowvis_rate": top_rate,
        "background_lowvis_rate": bg,
        "enrichment_ratio": top_rate / bg if np.isfinite(top_rate) and bg > 0 else math.nan,
        "odds_ratio": odds_ratio(y, top),
        "monthly_top_decile_lowvis_rate": monthly_rate,
        "monthly_enrichment_ratio": monthly_rate / bg if np.isfinite(monthly_rate) and bg > 0 else math.nan,
        "monthly_odds_ratio": odds_ratio(y, monthly_top),
    }


def bootstrap_informativeness(
    rng: np.random.Generator,
    score: np.ndarray,
    low_vis: np.ndarray,
    month: np.ndarray,
    iters: int,
) -> Dict[str, float]:
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(low_vis, dtype=bool)
    m = np.asarray(month)
    mask = np.isfinite(s)
    s, y, m = s[mask], y[mask], m[mask]
    if iters <= 0 or len(s) < 10 or y.sum() == 0 or y.sum() == len(y):
        return {}
    vals = {name: np.empty(int(iters), dtype=np.float64) for name in ["auc", "ap", "monthly_enrichment_ratio"]}
    n = len(s)
    for i in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        met = informativeness_metric_dict(s[idx], y[idx], m[idx])
        for name in vals:
            vals[name][i] = met[name]
    out: Dict[str, float] = {}
    for name, arr in vals.items():
        out[f"{name}_ci_low"] = float(np.nanpercentile(arr, 2.5))
        out[f"{name}_ci_high"] = float(np.nanpercentile(arr, 97.5))
    return out


def informativeness_table(
    common: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    label_tag: str,
    features: Sequence[str],
    min_pairs: int,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    if label_tag not in common:
        raise KeyError(f"label_source={label_tag!r} not found in sources")
    label_df = common[label_tag]
    y = label_df["low_vis"].to_numpy(dtype=bool)
    month = label_df["month"].to_numpy()
    rng = np.random.default_rng(seed + 17)
    rows: List[Dict[str, object]] = []
    for tag, df in common.items():
        for feat in features:
            if feat not in df:
                continue
            score = df[feat].to_numpy(dtype=np.float64)
            met = informativeness_metric_dict(score, y, month)
            if met["n"] < min_pairs:
                continue
            row: Dict[str, object] = {
                "source": tag,
                "source_label": labels.get(tag, tag),
                "label_source": label_tag,
                "feature": feat,
                "feature_label": FEATURE_META.get(feat, {}).get("label", feat),
                "unit": FEATURE_META.get(feat, {}).get("unit", ""),
                **met,
            }
            if bootstrap_iters > 0:
                row.update(bootstrap_informativeness(rng, score, y, month, bootstrap_iters))
            rows.append(row)
    return pd.DataFrame(rows)


def read_eval_sample(eval_dir: Path, tag: str) -> Optional[pd.DataFrame]:
    path = eval_dir / f"per_sample_{tag}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "time" not in df or "station_id" not in df or "pred" not in df:
        print(f"[warn] skip {path}: expected time, station_id, pred columns", file=sys.stderr)
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["station_key"] = normalize_station_ids(df["station_id"].values).values
    df["dup"] = df.groupby(["time", "station_key"]).cumcount()
    df["_key"] = make_key(df)
    if "y_cls" not in df:
        if "vis_raw_m" in df:
            raw = df["vis_raw_m"].to_numpy(dtype=np.float64)
            y_cls = np.full(len(df), 2, dtype=np.int64)
            y_cls[raw <= 1000.0] = 1
            y_cls[raw <= 200.0] = 0
            df["y_cls"] = y_cls
        else:
            df["y_cls"] = np.nan
    return df


def date_block_mean_ci(values: np.ndarray, times: Sequence[object], iters: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    dates = pd.to_datetime(pd.Series(times), errors="coerce").dt.floor("D").to_numpy()
    mask = np.isfinite(vals) & pd.notna(dates)
    vals = vals[mask]
    dates = dates[mask]
    unique, codes = np.unique(dates, return_inverse=True)
    if iters <= 0 or len(unique) < 3 or not vals.size:
        return math.nan, math.nan
    counts = np.bincount(codes, minlength=len(unique)).astype(np.float64)
    sums = np.bincount(codes, weights=vals, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(int(iters), len(unique)))
    replicate = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.nanpercentile(replicate, 2.5)), float(np.nanpercentile(replicate, 97.5))


def case_control_table(
    common: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    ref_tag: str,
    eval_dir: str,
    pangu_tag: str,
    numerical_tags: Sequence[str],
    features: Sequence[str],
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if not eval_dir:
        return pd.DataFrame()
    eval_path = Path(eval_dir)
    pangu_eval = read_eval_sample(eval_path, pangu_tag)
    if pangu_eval is None or pangu_tag not in common:
        print("[warn] Pangu per-sample CSV or source data not available; skip case-control.", file=sys.stderr)
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    pangu_eval = pangu_eval[["_key", "y_cls", "pred"]].rename(columns={"pred": "pangu_pred", "y_cls": "y_cls_pangu"})
    pangu_feat = common[pangu_tag][["_key", *[f for f in features if f in common[pangu_tag]]]]
    ref_feat = common.get(ref_tag, pd.DataFrame())
    for num_tag in numerical_tags:
        if num_tag not in common or num_tag == pangu_tag:
            continue
        num_eval = read_eval_sample(eval_path, num_tag)
        if num_eval is None:
            print(f"[warn] missing per_sample_{num_tag}.csv; skip this numerical source.", file=sys.stderr)
            continue
        num_eval = num_eval[["_key", "time", "y_cls", "pred"]].rename(
            columns={"time": "valid_time", "pred": "num_pred", "y_cls": "y_cls_num"}
        )
        merged = num_eval.merge(pangu_eval, on="_key", how="inner")
        if merged.empty:
            continue
        merged["true_low"] = pd.to_numeric(merged["y_cls_num"], errors="coerce") <= 1
        merged["num_pred_low"] = pd.to_numeric(merged["num_pred"], errors="coerce") <= 1
        merged["pangu_pred_low"] = pd.to_numeric(merged["pangu_pred"], errors="coerce") <= 1
        merged = merged.merge(
            common[num_tag][["_key", *[f for f in features if f in common[num_tag]]]].add_prefix("num_"),
            left_on="_key",
            right_on="num__key",
            how="left",
        )
        merged = merged.merge(
            pangu_feat.add_prefix("pangu_"),
            left_on="_key",
            right_on="pangu__key",
            how="left",
        )
        if ref_tag in common:
            merged = merged.merge(
                common[ref_tag][["_key", *[f for f in features if f in common[ref_tag]]]].add_prefix("ref_"),
                left_on="_key",
                right_on="ref__key",
                how="left",
            )
        case_masks = {
            "numerical_hit_pangu_miss": merged["true_low"] & merged["num_pred_low"] & ~merged["pangu_pred_low"],
            "both_hit": merged["true_low"] & merged["num_pred_low"] & merged["pangu_pred_low"],
            "both_miss": merged["true_low"] & ~merged["num_pred_low"] & ~merged["pangu_pred_low"],
            "pangu_hit_numerical_miss": merged["true_low"] & ~merged["num_pred_low"] & merged["pangu_pred_low"],
        }
        for case_name, mask in case_masks.items():
            sub = merged.loc[mask].copy()
            for feat in features:
                num_col = f"num_{feat}"
                pangu_col = f"pangu_{feat}"
                ref_col = f"ref_{feat}"
                if num_col not in sub or pangu_col not in sub:
                    continue
                num_v = pd.to_numeric(sub[num_col], errors="coerce").to_numpy(dtype=np.float64)
                pangu_v = pd.to_numeric(sub[pangu_col], errors="coerce").to_numpy(dtype=np.float64)
                ref_v = (
                    pd.to_numeric(sub[ref_col], errors="coerce").to_numpy(dtype=np.float64)
                    if ref_col in sub
                    else np.full(len(sub), np.nan)
                )
                finite = np.isfinite(num_v) | np.isfinite(pangu_v)
                diff_num = pangu_v - num_v
                diff_ref = pangu_v - ref_v
                num_lo, num_hi = date_block_mean_ci(
                    diff_num,
                    sub["valid_time"],
                    bootstrap_iters,
                    bootstrap_seed + len(rows) * 17,
                )
                ref_lo, ref_hi = date_block_mean_ci(
                    diff_ref,
                    sub["valid_time"],
                    bootstrap_iters,
                    bootstrap_seed + len(rows) * 17 + 3,
                )
                rows.append(
                    {
                        "numerical_source": num_tag,
                        "numerical_label": labels.get(num_tag, num_tag),
                        "pangu_source": pangu_tag,
                        "case": case_name,
                        "feature": feat,
                        "unit": FEATURE_META.get(feat, {}).get("unit", ""),
                        "n": int(mask.sum()),
                        "n_feature": int(finite.sum()),
                        "numerical_mean": float(np.nanmean(num_v)) if np.isfinite(num_v).any() else math.nan,
                        "pangu_mean": float(np.nanmean(pangu_v)) if np.isfinite(pangu_v).any() else math.nan,
                        "reference_mean": float(np.nanmean(ref_v)) if np.isfinite(ref_v).any() else math.nan,
                        "pangu_minus_numerical": (
                            float(np.nanmean(diff_num))
                            if np.isfinite(diff_num).any()
                            else math.nan
                        ),
                        "pangu_minus_numerical_ci_low": num_lo,
                        "pangu_minus_numerical_ci_high": num_hi,
                        "pangu_minus_reference": (
                            float(np.nanmean(diff_ref))
                            if np.isfinite(diff_ref).any()
                            else math.nan
                        ),
                        "pangu_minus_reference_ci_low": ref_lo,
                        "pangu_minus_reference_ci_high": ref_hi,
                    }
                )
    return pd.DataFrame(rows)


def savefig_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def plot_reference_quality(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["scope"] == "all") & (df["feature"].isin(["Q_1000", "DP_1000", "Q1000_MINUS_Q925"]))].copy()
    if sub.empty:
        return
    features = [f for f in ["Q_1000", "DP_1000", "Q1000_MINUS_Q925"] if f in set(sub["feature"])]
    labels = sub["source_label"].drop_duplicates().tolist()
    set_plot_style()
    fig, axes = plt.subplots(len(features), 2, figsize=(7.2, 2.0 + 1.55 * len(features)), squeeze=False)
    y = np.arange(len(labels))
    for row_idx, feat in enumerate(features):
        cur = sub[sub["feature"] == feat].set_index("source_label").reindex(labels)
        unit = str(FEATURE_META.get(feat, {}).get("unit", ""))
        for idx, label in enumerate(labels):
            source = str(cur.loc[label, "source"])
            style = source_style(source, label)
            for col_idx, metric in enumerate(("mae", "p95_delta")):
                value = float(cur.loc[label, metric])
                lo = float(cur.loc[label, f"{metric}_ci_low"]) if f"{metric}_ci_low" in cur else math.nan
                hi = float(cur.loc[label, f"{metric}_ci_high"]) if f"{metric}_ci_high" in cur else math.nan
                xerr = np.array([[value - lo], [hi - value]]) if np.isfinite(lo) and np.isfinite(hi) else None
                axes[row_idx, col_idx].errorbar(
                    value, idx, xerr=xerr, fmt=style["marker"], ms=5.2,
                    color=style["color"], capsize=2.4, lw=1.0,
                )
        feature_label = FEATURE_META.get(feat, {}).get("label", feat)
        axes[row_idx, 0].set_ylabel(str(feature_label))
        axes[row_idx, 0].set_xlabel(f"MAE ({unit}; lower is better)")
        axes[row_idx, 1].set_xlabel(f"P95 source - reference ({unit})")
        axes[row_idx, 1].axvline(0.0, color="#202124", lw=0.8)
        for ax in axes[row_idx]:
            ax.set_yticks(y)
            ax.set_yticklabels(labels if ax is axes[row_idx, 0] else [])
            ax.invert_yaxis()
    axes[0, 0].set_title("a  Overall agreement with ERA5 reference")
    axes[0, 1].set_title("b  Upper-tail displacement")
    fig.suptitle("Low-level moisture quality on the common sample", fontsize=10, y=1.01)
    fig.tight_layout()
    savefig_all(fig, out_dir, "fig_q1000_reference_quality")
    plt.close(fig)


def plot_informativeness(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["feature"].isin(["Q_1000", "DP_1000", "Q1000_MINUS_Q925"])].copy()
    if sub.empty:
        return
    features = [f for f in ["Q_1000", "DP_1000", "Q1000_MINUS_Q925"] if f in set(sub["feature"])]
    labels = sub["source_label"].drop_duplicates().tolist()
    set_plot_style()
    fig, axes = plt.subplots(len(features), 2, figsize=(7.2, 2.0 + 1.55 * len(features)), squeeze=False)
    x = np.arange(len(labels))
    colors = [source_style(str(sub[sub["source_label"] == label]["source"].iloc[0]), label)["color"] for label in labels]
    for row_idx, feat in enumerate(features):
        cur = sub[sub["feature"] == feat].set_index("source_label").reindex(labels)
        for col_idx, metric in enumerate(("ap", "monthly_enrichment_ratio")):
            values = cur[metric].to_numpy(dtype=float)
            lo_col = f"{metric}_ci_low"
            hi_col = f"{metric}_ci_high"
            if lo_col in cur and hi_col in cur:
                lo = cur[lo_col].to_numpy(dtype=float)
                hi = cur[hi_col].to_numpy(dtype=float)
                valid = np.isfinite(values) & np.isfinite(lo) & np.isfinite(hi)
                yerr = np.vstack([np.where(valid, values - lo, 0.0), np.where(valid, hi - values, 0.0)])
            else:
                yerr = None
            axes[row_idx, col_idx].bar(
                x,
                values,
                color=colors,
                width=0.68,
                yerr=yerr,
                error_kw={"ecolor": "#202124", "elinewidth": 0.8, "capsize": 2},
            )
        axes[row_idx, 1].axhline(1.0, color="#202124", lw=0.8)
        axes[row_idx, 0].set_ylabel(str(FEATURE_META.get(feat, {}).get("label", feat)))
        for ax in axes[row_idx]:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=26, ha="right")
        axes[row_idx, 0].set_ylim(bottom=0)
        axes[row_idx, 1].set_ylim(bottom=0)
    axes[0, 0].set_title("a  Average precision for observed low visibility")
    axes[0, 1].set_title("b  Monthly top-decile enrichment")
    fig.suptitle("Low-visibility information retained by moisture variables", fontsize=10, y=1.01)
    fig.tight_layout()
    savefig_all(fig, out_dir, "fig_q1000_lowvis_enrichment")
    plt.close(fig)


def plot_extreme_spatiotemporal(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["feature"] == "Q_1000"].copy()
    if sub.empty:
        return
    labels = sub["source_label"].drop_duplicates().tolist()
    quantiles = [q for q in (90, 95, 99) if q in set(sub["quantile"])]
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    offsets = np.linspace(-0.16, 0.16, max(len(labels), 1))
    x = np.arange(len(quantiles))
    for offset, label in zip(offsets, labels):
        source_rows = sub[sub["source_label"] == label]
        source_tag = str(source_rows["source"].iloc[0])
        style = source_style(source_tag, label)
        exact = source_rows[source_rows["event_definition"] == "exact_reference_threshold"].set_index("quantile").reindex(quantiles)
        matched = source_rows[source_rows["event_definition"] == "quantile_matched"].set_index("quantile").reindex(quantiles)
        for ax, cur, metric in (
            (axes[0], exact, "conditional_mae"),
            (axes[1], exact, "sedi"),
            (axes[2], matched, "sedi"),
        ):
            values = cur[metric].to_numpy(dtype=float)
            lo_col, hi_col = f"{metric}_ci_low", f"{metric}_ci_high"
            yerr = None
            if lo_col in cur and hi_col in cur:
                lo = cur[lo_col].to_numpy(dtype=float)
                hi = cur[hi_col].to_numpy(dtype=float)
                valid = np.isfinite(values) & np.isfinite(lo) & np.isfinite(hi)
                yerr = np.vstack([np.where(valid, values - lo, 0.0), np.where(valid, hi - values, 0.0)])
            ax.errorbar(
                x + offset,
                values,
                yerr=yerr,
                marker=style["marker"],
                ms=4.5,
                color=style["color"],
                lw=1.0,
                capsize=2.0,
                label=label,
            )
    axes[0].set_title("a  Error when ERA5 is extreme")
    axes[0].set_ylabel("Conditional MAE (g kg-1)")
    axes[1].set_title("b  Absolute-threshold concurrence")
    axes[1].set_ylabel("SEDI")
    axes[2].set_title("c  Quantile-matched concurrence")
    axes[2].set_ylabel("SEDI")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{q}" for q in quantiles])
        ax.set_xlabel("ERA5 monthly upper-tail threshold")
        ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    axes[1].axhline(0.0, color="#202124", lw=0.7)
    axes[2].axhline(0.0, color="#202124", lw=0.7)
    handles, legend_labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    savefig_all(fig, out_dir, "fig_q1000_extreme_spatiotemporal")
    plt.close(fig)


def plot_case_control(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    sub = df[df["feature"].isin(["Q_1000", "DP_1000", "Q1000_MINUS_Q925"])].copy()
    if sub.empty:
        return
    features = [f for f in ["Q_1000", "DP_1000", "Q1000_MINUS_Q925"] if f in set(sub["feature"])]
    numerical_labels = sub["numerical_label"].drop_duplicates().tolist()
    cases = [case for case in CASE_LABELS if case in set(sub["case"])]
    set_plot_style()
    fig, axes = plt.subplots(len(features), 1, figsize=(7.2, 1.8 + 1.65 * len(features)), squeeze=False)
    x = np.arange(len(cases))
    offsets = np.linspace(-0.18, 0.18, max(len(numerical_labels), 1))
    for row_idx, feat in enumerate(features):
        ax = axes[row_idx, 0]
        for offset, numerical_label in zip(offsets, numerical_labels):
            cur = sub[(sub["feature"] == feat) & (sub["numerical_label"] == numerical_label)].set_index("case").reindex(cases)
            source_tag = str(cur["numerical_source"].dropna().iloc[0]) if cur["numerical_source"].notna().any() else numerical_label
            style = source_style(source_tag, numerical_label)
            values = cur["pangu_minus_numerical"].to_numpy(dtype=float)
            lo = cur["pangu_minus_numerical_ci_low"].to_numpy(dtype=float) if "pangu_minus_numerical_ci_low" in cur else np.full(len(cur), np.nan)
            hi = cur["pangu_minus_numerical_ci_high"].to_numpy(dtype=float) if "pangu_minus_numerical_ci_high" in cur else np.full(len(cur), np.nan)
            valid_ci = np.isfinite(values) & np.isfinite(lo) & np.isfinite(hi)
            yerr = np.vstack([np.where(valid_ci, values - lo, 0.0), np.where(valid_ci, hi - values, 0.0)])
            ax.errorbar(
                x + offset, values, yerr=yerr, linestyle="none",
                marker=style["marker"], ms=5.2, color=style["color"],
                capsize=2.2, lw=0.9, label=numerical_label,
            )
        unit = str(FEATURE_META.get(feat, {}).get("unit", ""))
        ax.axhline(0.0, color="#202124", lw=0.8)
        ax.set_ylabel(f"{FEATURE_META.get(feat, {}).get('label', feat)}\nPangu - numerical ({unit})")
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LABELS[case] for case in cases])
    axes[0, 0].legend(ncol=min(3, len(numerical_labels)), loc="best")
    fig.suptitle("Moisture differences conditional on low-visibility forecast outcomes", fontsize=10, y=1.01)
    fig.tight_layout()
    savefig_all(fig, out_dir, "fig_q1000_pangu_case_control")
    plt.close(fig)


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    common_n: int,
    quality: pd.DataFrame,
    tail_summary: pd.DataFrame,
    extreme_spatiotemporal: pd.DataFrame,
    informativeness: pd.DataFrame,
    case_control: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Q1000 mechanism analysis summary")
    lines.append("")
    lines.append(f"- Common valid-time/station subset size: {common_n}")
    lines.append(f"- ERA5 handling: `{args.reference_source}` is used as reference analysis, not truth.")
    lines.append("- Pangu handling: no RH2M is derived from Q_1000; only Q_1000, DP_1000 and Q_1000-Q_925 are analyzed.")
    lines.append("- DP1000 is treated as a monotonic moisture-coordinate consistency check at fixed pressure, not as evidence independent of Q1000.")
    lines.append(f"- Bootstrap iterations: {args.bootstrap_iters}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `q1000_reference_quality_metrics.csv`: paired Q1000/DP1000 metrics against reference analysis.")
    lines.append("- `q1000_reference_tail_metrics.csv`: month-wise P90/P95/P99 tail hit/miss diagnostics.")
    lines.append("- `q1000_reference_tail_summary.csv`: seasonally controlled pooled tail hit, miss, false-extreme and tail-magnitude diagnostics.")
    lines.append("- `q1000_extreme_spatiotemporal_metrics.csv`: same-station/same-valid-time extreme verification using exact-reference and quantile-matched monthly thresholds.")
    lines.append("- `q1000_lowvis_informativeness.csv`: AUC/AP, top-decile enrichment, and odds ratios for observed low-vis labels.")
    lines.append("- `q1000_pangu_case_control.csv`: optional paired mechanism comparison for numerical-hit/Pangu-miss samples.")
    lines.append("")
    lines.append("## Decision logic")
    lines.append("")
    lines.append("- If Pangu has negative high-tail deltas and Pangu misses concentrate under high Q1000/DP1000 conditions, frame the gap as weak near-surface moisture-tail representation.")
    lines.append("- If Pangu Q1000 is close to reference but source-full remains weak, frame Q1000 as insufficient to replace RH2M/DPD and near-saturation coupling.")
    lines.append("- If q_core_no_rh2m performance approaches numerical sources, frame the main source-full gap as a missing-diagnostic-variable issue rather than Q1000 quality itself.")
    lines.append("")
    lines.append("## Method references for manuscript text")
    lines.append("")
    lines.append("- ERA5 reference analysis: Hersbach et al. 2020, QJRMS, doi:10.1002/qj.3803.")
    lines.append("- Distribution distance: Panaretos and Zemel 2019, Annual Review of Statistics and Its Application, doi:10.1146/annurev-statistics-030718-104938.")
    lines.append("- ROC/AP interpretation: Mason and Graham 2002, QJRMS, doi:10.1256/003590002320603584; Saito and Rehmsmeier 2015, PLOS ONE, doi:10.1371/journal.pone.0118432.")
    lines.append("- Model reliance / grouped permutation: Fisher, Rudin and Dominici 2019, JMLR 20:177.")
    lines.append("- Conditional permutation under correlated predictors: Strobl et al. 2008, BMC Bioinformatics 9:307.")
    lines.append("- ALE curves for correlated features: Apley and Zhu 2020, JRSS-B 82:1059-1086.")
    lines.append("- Rare-event SEDI and quantile matching: Ferro and Stephenson 2011, Weather and Forecasting 26:699-713.")
    lines.append("- Extreme-only conditional error caveat: Lerch et al. 2017, Statistical Science 32:106-127.")
    lines.append("- Date-block resampling rationale: Hamill 1999, Weather and Forecasting 14:155-167.")
    if not quality.empty:
        q = quality[(quality["scope"] == "all") & (quality["feature"] == "Q_1000")]
        if not q.empty:
            lines.append("")
            lines.append("## Quick Q1000 tail check")
            for _, row in q.sort_values("p95_delta").iterrows():
                lines.append(
                    f"- {row['source_label']}: MAE={row['mae']:.3g}, P95 delta={row['p95_delta']:.3g} {row['unit']}."
                )
    if not tail_summary.empty:
        q95 = tail_summary[(tail_summary["feature"] == "Q_1000") & (tail_summary["quantile"] == 95)]
        if not q95.empty:
            lines.append("")
            lines.append("## Tail magnitude versus placement")
            for _, row in q95.sort_values("hit_rate", ascending=False).iterrows():
                lines.append(
                    f"- {row['source_label']}: pooled P95 hit={row['hit_rate']:.3g}, "
                    f"false-extreme={row['false_extreme_rate']:.3g}, "
                    f"tail-mean bias={row['source_minus_reference_tail_mean']:.3g}."
                )
            lines.append("- A stronger marginal upper tail is not equivalent to better placement of reference extremes or stronger low-visibility information.")
    if not extreme_spatiotemporal.empty:
        q95 = extreme_spatiotemporal[
            (extreme_spatiotemporal["feature"] == "Q_1000")
            & (extreme_spatiotemporal["quantile"] == 95)
        ]
        if not q95.empty:
            lines.append("")
            lines.append("## Same-place/same-time P95 verification")
            exact = q95[q95["event_definition"] == "exact_reference_threshold"]
            matched = q95[q95["event_definition"] == "quantile_matched"]
            for label in q95["source_label"].drop_duplicates():
                e = exact[exact["source_label"] == label]
                m = matched[matched["source_label"] == label]
                if e.empty or m.empty:
                    continue
                lines.append(
                    f"- {label}: ERA5-extreme conditional MAE={e.iloc[0]['conditional_mae']:.3g} g kg-1, "
                    f"absolute-threshold SEDI={e.iloc[0]['sedi']:.3g}, quantile-matched SEDI={m.iloc[0]['sedi']:.3g}."
                )
            lines.append("- Conditional MAE is descriptive only; primary extreme-event ranking should use the full contingency table and SEDI, not only ERA5-extreme cases.")
    if not informativeness.empty:
        q = informativeness[informativeness["feature"] == "Q_1000"]
        if not q.empty:
            lines.append("")
            lines.append("## Quick informativeness check")
            for _, row in q.sort_values("monthly_enrichment_ratio", ascending=False).iterrows():
                lines.append(
                    f"- {row['source_label']}: AP={row['ap']:.3g}, monthly top-decile enrichment={row['monthly_enrichment_ratio']:.3g}."
                )
    if case_control.empty:
        lines.append("")
        lines.append("Case-control was skipped or empty. Re-run source-full argmax eval with `NO_PER_SAMPLE_CSV=0` to enable it.")
    (out_dir / "q1000_mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_sources(args.sources)
    features = split_features(args.features)
    sources: Dict[str, SourceData] = {}
    for spec in specs:
        print(f"[load] {spec.tag}: {spec.data_dir}", flush=True)
        sources[spec.tag] = load_source(spec, features, args.limit_samples, args.low_vis_threshold_m)
    common_keys = common_key_subset(sources)
    common = {tag: filter_common(src, common_keys) for tag, src in sources.items()}
    labels = {src.spec.tag: src.spec.label for src in sources.values()}
    label_tag = args.label_source or args.reference_source
    if label_tag not in common:
        label_tag = next(iter(common))
        print(f"[warn] label_source not found; using {label_tag}", file=sys.stderr)
    quality, tail = reference_quality_tables(
        common,
        labels,
        args.reference_source,
        features,
        args.min_pairs,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    tail_summary = summarize_reference_tail(tail)
    extreme_spatiotemporal = extreme_spatiotemporal_table(
        common,
        labels,
        args.reference_source,
        features,
        args.min_pairs,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    info = informativeness_table(
        common,
        labels,
        label_tag,
        features,
        args.min_pairs,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    if args.numerical_tags:
        numerical_tags = [x.strip() for x in args.numerical_tags.split(",") if x.strip()]
    else:
        numerical_tags = [tag for tag in common if tag != args.pangu_tag and "pangu" not in tag.lower()]
    case_control = case_control_table(
        common,
        labels,
        args.reference_source,
        args.eval_dir,
        args.pangu_tag,
        numerical_tags,
        features,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    if args.require_case_control and case_control.empty:
        expected_tags = [args.pangu_tag, *numerical_tags]
        expected = [str(Path(args.eval_dir) / f"per_sample_{tag}.csv") for tag in expected_tags]
        existing = sorted(str(path) for path in Path(args.eval_dir).glob("per_sample_*.csv")) if args.eval_dir else []
        raise FileNotFoundError(
            "Case-control is required but no paired rows were produced. "
            f"Expected per-sample files: {expected}. Existing per-sample files: {existing}. "
            "Generate them first with: NO_PER_SAMPLE_CSV=0 bash "
            "submit_static_rnn_source_full_argmax_eval.sh figure1_all_sources"
        )

    quality.to_csv(out_dir / "q1000_reference_quality_metrics.csv", index=False)
    tail.to_csv(out_dir / "q1000_reference_tail_metrics.csv", index=False)
    tail_summary.to_csv(out_dir / "q1000_reference_tail_summary.csv", index=False)
    extreme_spatiotemporal.to_csv(out_dir / "q1000_extreme_spatiotemporal_metrics.csv", index=False)
    info.to_csv(out_dir / "q1000_lowvis_informativeness.csv", index=False)
    case_control.to_csv(out_dir / "q1000_pangu_case_control.csv", index=False)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sources": [
                    {"tag": s.tag, "data_dir": str(s.data_dir), "label": s.label, "group": s.group}
                    for s in specs
                ],
                "reference_source": args.reference_source,
                "label_source": label_tag,
                "pangu_tag": args.pangu_tag,
                "features": features,
                "common_sample_count": len(common_keys),
                "limit_samples": args.limit_samples,
                "low_vis_threshold_m": args.low_vis_threshold_m,
                "bootstrap_iters": args.bootstrap_iters,
                "bootstrap_seed": args.bootstrap_seed,
                "require_case_control": args.require_case_control,
                "method_notes": {
                    "era5": "reference analysis, not truth",
                    "pangu": "no RH2M derived from Q_1000",
                    "comparison_subset": "intersection of valid time, station_id and duplicate index across all sources",
                    "extreme_thresholds": (
                        "calendar-month ERA5 P90/P95/P99; exact-reference thresholds test amplitude plus concurrence, "
                        "quantile-matched thresholds test same-station/same-valid-time rank concurrence"
                    ),
                    "uncertainty": "95% confidence intervals resample valid dates as blocks",
                    "overall_error_uncertainty": "bias/MAE/RMSE use valid-date block bootstrap; marginal percentile displacement is descriptive",
                    "conditional_error_caveat": (
                        "ERA5-extreme conditional bias/MAE/RMSE are descriptive diagnostics and are not used alone for ranking"
                    ),
                },
            },
            f,
            indent=2,
        )

    plot_reference_quality(quality, out_dir)
    plot_extreme_spatiotemporal(extreme_spatiotemporal, out_dir)
    plot_informativeness(info, out_dir)
    plot_case_control(case_control, out_dir)
    write_summary(
        out_dir,
        args,
        len(common_keys),
        quality,
        tail_summary,
        extreme_spatiotemporal,
        info,
        case_control,
    )
    print(f"[OK] wrote Q1000 mechanism outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
