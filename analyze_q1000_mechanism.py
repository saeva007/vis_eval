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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
            "across Tianji, T2ND, IFS, ERA5 reference analysis, and Pangu."
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


def bootstrap_ci(
    rng: np.random.Generator,
    src: np.ndarray,
    ref: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    iters: int,
) -> Tuple[float, float]:
    a, b = finite_pair(src, ref)
    n = len(a)
    if iters <= 0 or n < 5:
        return math.nan, math.nan
    vals = np.empty(int(iters), dtype=np.float64)
    for i in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        vals[i] = metric_fn(a[idx], b[idx])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def add_quality_bootstrap(row: Dict[str, object], rng: np.random.Generator, src: np.ndarray, ref: np.ndarray, iters: int) -> None:
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "bias": lambda a, b: float(np.nanmean(a - b)),
        "mae": lambda a, b: float(np.nanmean(np.abs(a - b))),
        "rmse": lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2))),
        "p95_delta": lambda a, b: float(np.nanpercentile(a, 95) - np.nanpercentile(b, 95)),
        "p99_delta": lambda a, b: float(np.nanpercentile(a, 99) - np.nanpercentile(b, 99)),
    }
    for name, fn in metrics.items():
        lo, hi = bootstrap_ci(rng, src, ref, fn, iters)
        row[f"{name}_ci_low"] = lo
        row[f"{name}_ci_high"] = hi


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
                    add_quality_bootstrap(row, rng, src_v, ref_v, bootstrap_iters)
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
                            "hit_rate": hit / denom_ref,
                            "miss_rate": miss / denom_ref,
                            "false_extreme_rate": false / denom_non,
                            "source_tail_mean": float(np.nanmean(a[src_extreme])) if src_extreme.any() else math.nan,
                            "reference_tail_mean": float(np.nanmean(b[ref_extreme])) if ref_extreme.any() else math.nan,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(tail_rows)


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


def case_control_table(
    common: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    ref_tag: str,
    eval_dir: str,
    pangu_tag: str,
    numerical_tags: Sequence[str],
    features: Sequence[str],
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
        num_eval = num_eval[["_key", "y_cls", "pred"]].rename(columns={"pred": "num_pred", "y_cls": "y_cls_num"})
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
                            float(np.nanmean(pangu_v - num_v))
                            if np.isfinite(pangu_v - num_v).any()
                            else math.nan
                        ),
                        "pangu_minus_reference": (
                            float(np.nanmean(pangu_v - ref_v))
                            if np.isfinite(pangu_v - ref_v).any()
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def savefig_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")


def plot_reference_quality(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["scope"] == "all") & (df["feature"].isin(["Q_1000", "DP_1000", "Q1000_MINUS_Q925"]))].copy()
    if sub.empty:
        return
    labels = sub["source_label"].drop_duplicates().tolist()
    features = [f for f in ["Q_1000", "DP_1000", "Q1000_MINUS_Q925"] if f in set(sub["feature"])]
    x = np.arange(len(labels))
    width = 0.8 / max(len(features), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for j, feat in enumerate(features):
        cur = sub[sub["feature"] == feat].set_index("source_label").reindex(labels)
        axes[0].bar(x + (j - (len(features) - 1) / 2) * width, cur["mae"], width, label=FEATURE_META.get(feat, {}).get("label", feat))
        axes[1].bar(x + (j - (len(features) - 1) / 2) * width, cur["p95_delta"], width, label=FEATURE_META.get(feat, {}).get("label", feat))
    axes[0].set_ylabel("MAE vs ERA5 reference")
    axes[1].set_ylabel("P95(source) - P95(reference)")
    for ax in axes:
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Q1000 reference-analysis quality")
    savefig_all(fig, out_dir, "fig_q1000_reference_quality")
    plt.close(fig)


def plot_informativeness(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["feature"].isin(["Q_1000", "DP_1000", "Q1000_MINUS_Q925"])].copy()
    if sub.empty:
        return
    labels = sub["source_label"].drop_duplicates().tolist()
    features = [f for f in ["Q_1000", "DP_1000", "Q1000_MINUS_Q925"] if f in set(sub["feature"])]
    x = np.arange(len(labels))
    width = 0.8 / max(len(features), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for j, feat in enumerate(features):
        cur = sub[sub["feature"] == feat].set_index("source_label").reindex(labels)
        axes[0].bar(x + (j - (len(features) - 1) / 2) * width, cur["ap"], width, label=FEATURE_META.get(feat, {}).get("label", feat))
        axes[1].bar(x + (j - (len(features) - 1) / 2) * width, cur["monthly_enrichment_ratio"], width, label=FEATURE_META.get(feat, {}).get("label", feat))
    axes[0].set_ylabel("Average precision for low-vis")
    axes[1].set_ylabel("Monthly top-decile enrichment")
    axes[1].axhline(1.0, color="0.3", lw=0.8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Q1000 low-visibility informativeness")
    savefig_all(fig, out_dir, "fig_q1000_lowvis_enrichment")
    plt.close(fig)


def plot_case_control(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    sub = df[
        (df["case"] == "numerical_hit_pangu_miss")
        & (df["feature"].isin(["Q_1000", "DP_1000", "Q1000_MINUS_Q925"]))
    ].copy()
    if sub.empty:
        return
    labels = sub["numerical_label"].drop_duplicates().tolist()
    features = [f for f in ["Q_1000", "DP_1000", "Q1000_MINUS_Q925"] if f in set(sub["feature"])]
    x = np.arange(len(labels))
    width = 0.8 / max(len(features), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for j, feat in enumerate(features):
        cur = sub[sub["feature"] == feat].set_index("numerical_label").reindex(labels)
        ax.bar(
            x + (j - (len(features) - 1) / 2) * width,
            cur["pangu_minus_numerical"],
            width,
            label=FEATURE_META.get(feat, {}).get("label", feat),
        )
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylabel("Pangu - numerical source")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Pangu misses where numerical-source model hits")
    savefig_all(fig, out_dir, "fig_q1000_pangu_case_control")
    plt.close(fig)


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    common_n: int,
    quality: pd.DataFrame,
    informativeness: pd.DataFrame,
    case_control: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Q1000 mechanism analysis summary")
    lines.append("")
    lines.append(f"- Common valid-time/station subset size: {common_n}")
    lines.append(f"- ERA5 handling: `{args.reference_source}` is used as reference analysis, not truth.")
    lines.append("- Pangu handling: no RH2M is derived from Q_1000; only Q_1000, DP_1000 and Q_1000-Q_925 are analyzed.")
    lines.append(f"- Bootstrap iterations: {args.bootstrap_iters}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `q1000_reference_quality_metrics.csv`: paired Q1000/DP1000 metrics against reference analysis.")
    lines.append("- `q1000_reference_tail_metrics.csv`: month-wise P90/P95/P99 tail hit/miss diagnostics.")
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
    lines.append("- Model reliance / grouped permutation: Fisher, Rudin and Dominici 2019, JMLR 20:177.")
    lines.append("- Conditional permutation under correlated predictors: Strobl et al. 2008, BMC Bioinformatics 9:307.")
    lines.append("- ALE curves for correlated features: Apley and Zhu 2020, JRSS-B 82:1059-1086.")
    if not quality.empty:
        q = quality[(quality["scope"] == "all") & (quality["feature"] == "Q_1000")]
        if not q.empty:
            lines.append("")
            lines.append("## Quick Q1000 tail check")
            for _, row in q.sort_values("p95_delta").iterrows():
                lines.append(
                    f"- {row['source_label']}: MAE={row['mae']:.3g}, P95 delta={row['p95_delta']:.3g} {row['unit']}."
                )
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
    )

    quality.to_csv(out_dir / "q1000_reference_quality_metrics.csv", index=False)
    tail.to_csv(out_dir / "q1000_reference_tail_metrics.csv", index=False)
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
                "method_notes": {
                    "era5": "reference analysis, not truth",
                    "pangu": "no RH2M derived from Q_1000",
                    "comparison_subset": "intersection of valid time, station_id and duplicate index across all sources",
                },
            },
            f,
            indent=2,
        )

    plot_reference_quality(quality, out_dir)
    plot_informativeness(info, out_dir)
    plot_case_control(case_control, out_dir)
    write_summary(out_dir, args, len(common_keys), quality, info, case_control)
    print(f"[OK] wrote Q1000 mechanism outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
