#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation-anchored RH2M verification across forecast sources.

The comparison is restricted to one common valid-time/station sample across
all sources and station observations. Pangu RH2M proxies are rejected by
default because they are not comparable 2-m humidity products.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
if str(VIS_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_EVAL_DIR))

from analyze_key_variable_quality import (  # noqa: E402
    choose_obs_time_shift,
    clean_physical_values,
    convert_forecast_units,
    read_build_config,
    read_meta,
)
from feature_catalog_pm10_pm25 import dynamic_features_for_count  # noqa: E402


SOURCE_STYLE = {
    "tianji": {"color": "#2878A5", "marker": "o", "label": "Tianji product"},
    "t2nd": {"color": "#8EC9E2", "marker": "s", "label": "Tianji T2ND RH2M"},
    "ifs": {"color": "#C47A1D", "marker": "^", "label": "IFS"},
    "era5": {"color": "#6C7280", "marker": "D", "label": "ERA5 analysis"},
    "pangu": {"color": "#7651A8", "marker": "P", "label": "Pangu"},
    "observed": {"color": "#202124", "marker": "", "label": "Observed"},
}
THRESHOLD_COLORS = {90.0: "#BFD7E5", 95.0: "#5A94B8", 98.0: "#C96B4B", 99.0: "#8E2F2F"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Observation-anchored multi-source RH2M verification.")
    ap.add_argument("--sources", required=True, help="Semicolon-separated tag=dataset_dir|label|group specs.")
    ap.add_argument("--obs_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=12, help="Fallback only when dataset config lacks window.")
    ap.add_argument("--dyn_vars_count", type=int, default=27, help="Fallback only when dataset config lacks layout metadata.")
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--timezone_probe_files", type=int, default=96)
    ap.add_argument("--thresholds", default="90,95,98,99", help="Physical RH2M thresholds in percent.")
    ap.add_argument("--quantiles", default="50,75,90,95,98,99")
    ap.add_argument("--min_pairs", type=int, default=100)
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260628)
    ap.add_argument(
        "--allow_pangu_rh2m_proxy",
        action="store_true",
        help="Explicitly allow a Pangu RH2M proxy. Off by default for the manuscript comparison.",
    )
    ap.add_argument(
        "--allow_non_rh2m_mismatch",
        action="store_true",
        help="Continue if Tianji and T2ND contain differences outside RH2M; the audit is always written.",
    )
    return ap.parse_args()


def split_numbers(value: str, fallback: Sequence[float]) -> List[float]:
    out = [float(v.strip()) for v in str(value or "").replace(";", ",").split(",") if v.strip()]
    return out or list(fallback)


def parse_sources(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Source entry must be tag=path: {chunk!r}")
        tag, rest = chunk.split("=", 1)
        parts = [p.strip() for p in rest.split("|")]
        rows.append(
            {
                "tag": tag.strip(),
                "path": parts[0],
                "label": parts[1] if len(parts) > 1 and parts[1] else tag.strip(),
                "group": parts[2] if len(parts) > 2 else "",
            }
        )
    if not rows:
        raise ValueError("No sources parsed.")
    tags = [row["tag"] for row in rows]
    if len(tags) != len(set(tags)):
        raise ValueError("Duplicate source tags are not allowed.")
    return rows


def source_family(tag: str, label: str = "") -> str:
    text = f"{tag} {label}".lower()
    if "t2nd" in text:
        return "t2nd"
    if "tianji" in text:
        return "tianji"
    if "era5" in text:
        return "era5"
    if "pangu" in text:
        return "pangu"
    if "ifs" in text:
        return "ifs"
    return "era5"


def style_for(tag: str, label: str = "") -> Dict[str, str]:
    return SOURCE_STYLE[source_family(tag, label)]


def normalize_station_ids(values: Iterable[object]) -> pd.Series:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    out = series.astype(str)
    mask = numeric.notna()
    out.loc[mask] = numeric.loc[mask].astype(np.int64).astype(str)
    return out


def ensure_keys(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    if "station_key" not in out:
        out["station_key"] = normalize_station_ids(out["station_id"].values).values
    if "dup" not in out:
        out["dup"] = out.groupby(["time", "station_key"]).cumcount()
    if "row_idx" not in out:
        out["row_idx"] = np.arange(len(out), dtype=np.int64)
    out["_key"] = (
        out["time"].astype("int64").astype(str)
        + "|"
        + out["station_key"].astype(str)
        + "|"
        + out["dup"].astype(str)
    )
    return out


def infer_layout(data_dir: Path, cfg: Dict[str, object], x: np.ndarray, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
    window = int(cfg.get("window") or args.window)
    dyn_vars = int(cfg.get("dyn_vars") or cfg.get("dyn_vars_count") or 0)
    if dyn_vars <= 0:
        if x.ndim == 3:
            dyn_vars = int(x.shape[-1])
        else:
            fe_dim = int(cfg.get("fe_dim") or 0)
            dyn_vars = int((x.shape[1] - fe_dim) // window)
    order = cfg.get("dynamic_feature_order")
    if isinstance(order, list) and len(order) == dyn_vars:
        dynamic_order = [str(v) for v in order]
    else:
        dynamic_order = [str(item["feature"]) for item in dynamic_features_for_count(dyn_vars)]
    if "RH2M" not in {v.upper().replace("-", "_") for v in dynamic_order}:
        raise KeyError(f"RH2M is absent from dynamic_feature_order for {data_dir}")
    return window, dyn_vars, dynamic_order


def read_last_hour_rh2m(data_dir: Path, cfg: Dict[str, object], meta: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    path = data_dir / "X_test.npy"
    x = np.load(path, mmap_mode="r")
    window, dyn_vars, order = infer_layout(data_dir, cfg, x, args)
    lookup = {str(v).upper().replace("-", "_"): i for i, v in enumerate(order)}
    idx = lookup["RH2M"]
    rows = meta["row_idx"].to_numpy(dtype=np.int64)
    if x.ndim == 3:
        values = np.asarray(x[rows, window - 1, idx], dtype=np.float64)
    else:
        col = (window - 1) * dyn_vars + idx
        values = np.asarray(x[rows, col], dtype=np.float64)
    return convert_forecast_units("RH2M", values)


def load_source(entry: Dict[str, str], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    data_dir = Path(entry["path"]).expanduser()
    cfg = read_build_config(data_dir)
    meta = ensure_keys(read_meta(data_dir, args.limit_samples))
    forecast = read_last_hour_rh2m(data_dir, cfg, meta, args)
    best_shift, obs, tz_diag = choose_obs_time_shift(Path(args.obs_root), meta, args.timezone_probe_files)
    merged = meta.merge(obs, on=["time", "station_key"], how="left")
    if "rhu" not in merged:
        raise KeyError(f"Station observation files under {args.obs_root!r} do not provide the required 'rhu' column.")
    observed = clean_physical_values(
        "RH2M", pd.to_numeric(merged["rhu"], errors="coerce").to_numpy(dtype=np.float64)
    )
    sample = merged[["_key", "time", "station_key", "dup", "row_idx"]].copy()
    sample["forecast_rh2m"] = forecast[: len(sample)]
    sample["obs_rh2m"] = observed
    sample["date"] = pd.to_datetime(sample["time"]).dt.floor("D")
    sample["month"] = pd.to_datetime(sample["time"]).dt.month
    diag = {
        "tag": entry["tag"],
        "source_label": entry["label"],
        "data_dir": str(data_dir),
        "feature_set": cfg.get("feature_set", ""),
        "rh2m_source": cfg.get("rh2m_source", ""),
        "obs_time_shift_to_utc_hours": float(best_shift),
        "obs_time_interpretation": "raw_obs_time_is_bjt" if best_shift == -8.0 else "raw_obs_time_is_utc",
    }
    tz_diag = tz_diag.assign(tag=entry["tag"], source_label=entry["label"])
    return sample, tz_diag, diag


def build_common_sample(samples: Dict[str, pd.DataFrame], entries: Sequence[Dict[str, str]], min_pairs: int) -> pd.DataFrame:
    common_keys = set.intersection(*(set(df["_key"]) for df in samples.values()))
    if not common_keys:
        raise RuntimeError("No common valid-time/station rows across RH2M sources.")
    key_order = pd.DataFrame({"_key": sorted(common_keys)})
    first_tag = entries[0]["tag"]
    base = key_order.merge(
        samples[first_tag][["_key", "time", "station_key", "dup", "date", "month", "obs_rh2m"]],
        on="_key",
        how="left",
    )
    for entry in entries:
        tag = entry["tag"]
        base = base.merge(
            samples[tag][["_key", "row_idx", "forecast_rh2m"]].rename(
                columns={"row_idx": f"row_idx_{tag}", "forecast_rh2m": f"forecast_{tag}"}
            ),
            on="_key",
            how="left",
        )
    required = ["obs_rh2m", *[f"forecast_{entry['tag']}" for entry in entries]]
    mask = np.ones(len(base), dtype=bool)
    for col in required:
        mask &= np.isfinite(pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float))
    common = base.loc[mask].reset_index(drop=True)
    if len(common) < int(min_pairs):
        raise RuntimeError(f"Only {len(common)} common finite RH2M-observation pairs; min_pairs={min_pairs}.")
    return common


def audit_tianji_t2nd_identity(
    common: pd.DataFrame,
    entries: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    tianji = next((entry for entry in entries if source_family(entry["tag"], entry["label"]) == "tianji"), None)
    t2nd = next((entry for entry in entries if source_family(entry["tag"], entry["label"]) == "t2nd"), None)
    if tianji is None or t2nd is None:
        return {"status": "not_applicable", "reason": "Tianji/T2ND pair not both present"}
    arrays = []
    layouts = []
    for entry in (tianji, t2nd):
        data_dir = Path(entry["path"]).expanduser()
        cfg = read_build_config(data_dir)
        x = np.load(data_dir / "X_test.npy", mmap_mode="r")
        layouts.append(infer_layout(data_dir, cfg, x, args))
        arrays.append(x)
    xa, xb = arrays
    window_a, dyn_a, order_a = layouts[0]
    window_b, dyn_b, order_b = layouts[1]
    result: Dict[str, object] = {
        "source_a": tianji["tag"],
        "source_b": t2nd["tag"],
        "n_common_rows": len(common),
        "shape_a": list(xa.shape),
        "shape_b": list(xb.shape),
        "window_a": window_a,
        "window_b": window_b,
        "dynamic_order_equal": order_a == order_b,
        "rtol": 1e-6,
        "atol": 1e-6,
    }
    if xa.ndim != xb.ndim or xa.shape[1:] != xb.shape[1:] or window_a != window_b or dyn_a != dyn_b or order_a != order_b:
        result.update({"status": "failed_layout_mismatch", "non_rh2m_identical": False})
        return result
    rh_idx = [i for i, name in enumerate(order_a) if str(name).upper().replace("-", "_") == "RH2M"]
    if len(rh_idx) != 1:
        result.update({"status": "failed_rh2m_index", "non_rh2m_identical": False, "rh2m_indices": rh_idx})
        return result
    rh_idx = rh_idx[0]
    if xa.ndim == 3:
        keep = np.asarray([i for i in range(dyn_a) if i != rh_idx], dtype=np.int64)
    else:
        excluded = {hour * dyn_a + rh_idx for hour in range(window_a)}
        keep = np.asarray([i for i in range(xa.shape[1]) if i not in excluded], dtype=np.int64)
    rows_a = common[f"row_idx_{tianji['tag']}"] .to_numpy(dtype=np.int64)
    rows_b = common[f"row_idx_{t2nd['tag']}"] .to_numpy(dtype=np.int64)
    mismatch_count = 0
    compared_count = 0
    max_abs_diff = 0.0
    for start in range(0, len(common), 2048):
        ia = rows_a[start : start + 2048]
        ib = rows_b[start : start + 2048]
        if xa.ndim == 3:
            a = np.asarray(xa[ia][:, :, keep], dtype=np.float64)
            b = np.asarray(xb[ib][:, :, keep], dtype=np.float64)
        else:
            a = np.asarray(xa[ia][:, keep], dtype=np.float64)
            b = np.asarray(xb[ib][:, keep], dtype=np.float64)
        equal = np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
        mismatch_count += int(equal.size - equal.sum())
        compared_count += int(equal.size)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a[finite] - b[finite]))))
    non_rh_identical: Optional[bool] = mismatch_count == 0 if compared_count > 0 else None
    result.update(
        {
            "status": (
                "not_testable_no_non_rh2m_values"
                if compared_count == 0
                else "passed"
                if mismatch_count == 0
                else "failed_value_mismatch"
            ),
            "non_rh2m_identical": non_rh_identical,
            "compared_value_count": compared_count,
            "mismatch_count": mismatch_count,
            "max_abs_diff": max_abs_diff,
            "excluded_feature": "RH2M at every input hour",
        }
    )
    return result


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) <= 0 or np.nanstd(b) <= 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    precision = np.cumsum(ys) / (np.arange(len(ys)) + 1.0)
    return float(np.sum(precision[ys]) / n_pos)


def continuous_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)
    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": corr_safe(pred, obs),
    }


def event_metrics(pred: np.ndarray, obs: np.ndarray, threshold: float) -> Dict[str, float]:
    observed = np.asarray(obs) >= float(threshold)
    forecast = np.asarray(pred) >= float(threshold)
    h = int(np.sum(observed & forecast))
    m = int(np.sum(observed & ~forecast))
    f = int(np.sum(~observed & forecast))
    c = int(np.sum(~observed & ~forecast))
    n = h + m + f + c
    pod = h / (h + m) if h + m else math.nan
    far = f / (h + f) if h + f else math.nan
    pofd = f / (f + c) if f + c else math.nan
    csi = h / (h + m + f) if h + m + f else math.nan
    random_hits = (h + m) * (h + f) / n if n else math.nan
    ets_denom = h + m + f - random_hits if n else math.nan
    ets = (h - random_hits) / ets_denom if np.isfinite(ets_denom) and ets_denom != 0 else math.nan
    frequency_bias = (h + f) / (h + m) if h + m else math.nan
    hit_rate = (h + 0.5) / (h + m + 1.0)
    false_rate = (f + 0.5) / (f + c + 1.0)
    sedi_num = math.log(false_rate) - math.log(hit_rate) - math.log(1 - false_rate) + math.log(1 - hit_rate)
    sedi_den = math.log(false_rate) + math.log(hit_rate) + math.log(1 - false_rate) + math.log(1 - hit_rate)
    sedi = sedi_num / sedi_den if sedi_den != 0 and (h + m) > 0 and (f + c) > 0 else math.nan
    return {
        "hits": h,
        "misses": m,
        "false_alarms": f,
        "correct_negatives": c,
        "event_count": h + m,
        "event_rate": (h + m) / n if n else math.nan,
        "pod": pod,
        "far": far,
        "pofd": pofd,
        "csi": csi,
        "ets": ets,
        "frequency_bias": frequency_bias,
        "sedi": sedi,
        "roc_auc": auc_rank(observed, pred),
        "average_precision": average_precision(observed, pred),
    }


def bootstrap_day_draws(dates: np.ndarray, iters: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, day_code = np.unique(dates, return_inverse=True)
    if iters <= 0 or len(unique) < 3:
        return unique, day_code, np.empty((0, len(unique)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(int(iters), len(unique)), dtype=np.int64)
    return unique, day_code, draws


def continuous_bootstrap_values(pred: np.ndarray, obs: np.ndarray, day_code: np.ndarray, draws: np.ndarray) -> Dict[str, np.ndarray]:
    if not draws.size:
        return {}
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    err = pred - obs
    n_days = int(day_code.max()) + 1
    daily = np.column_stack(
        [
            np.bincount(day_code, minlength=n_days),
            np.bincount(day_code, weights=err, minlength=n_days),
            np.bincount(day_code, weights=np.abs(err), minlength=n_days),
            np.bincount(day_code, weights=err**2, minlength=n_days),
            np.bincount(day_code, weights=pred, minlength=n_days),
            np.bincount(day_code, weights=obs, minlength=n_days),
            np.bincount(day_code, weights=pred**2, minlength=n_days),
            np.bincount(day_code, weights=obs**2, minlength=n_days),
            np.bincount(day_code, weights=pred * obs, minlength=n_days),
        ]
    )
    sums = daily[draws].sum(axis=1)
    n, sum_err, sum_abs, sum_sq, sum_p, sum_o, sum_p2, sum_o2, sum_po = sums.T
    cov = sum_po - sum_p * sum_o / n
    var_p = np.maximum(sum_p2 - sum_p**2 / n, 0.0)
    var_o = np.maximum(sum_o2 - sum_o**2 / n, 0.0)
    denom = np.sqrt(var_p * var_o)
    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
    return {
        "bias": sum_err / n,
        "mae": sum_abs / n,
        "rmse": np.sqrt(sum_sq / n),
        "corr": corr,
    }


def event_bootstrap_values(
    pred: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    day_code: np.ndarray,
    draws: np.ndarray,
) -> Dict[str, np.ndarray]:
    if not draws.size:
        return {}
    observed = np.asarray(obs) >= float(threshold)
    forecast = np.asarray(pred) >= float(threshold)
    n_days = int(day_code.max()) + 1
    daily = np.column_stack(
        [
            np.bincount(day_code, weights=(observed & forecast), minlength=n_days),
            np.bincount(day_code, weights=(observed & ~forecast), minlength=n_days),
            np.bincount(day_code, weights=(~observed & forecast), minlength=n_days),
            np.bincount(day_code, weights=(~observed & ~forecast), minlength=n_days),
        ]
    )
    h, m, f, c = daily[draws].sum(axis=1).T
    n = h + m + f + c
    pod = np.divide(h, h + m, out=np.full_like(h, np.nan), where=(h + m) > 0)
    far = np.divide(f, h + f, out=np.full_like(h, np.nan), where=(h + f) > 0)
    csi = np.divide(h, h + m + f, out=np.full_like(h, np.nan), where=(h + m + f) > 0)
    random_hits = np.divide((h + m) * (h + f), n, out=np.full_like(h, np.nan), where=n > 0)
    ets_denom = h + m + f - random_hits
    ets = np.divide(h - random_hits, ets_denom, out=np.full_like(h, np.nan), where=ets_denom != 0)
    frequency_bias = np.divide(h + f, h + m, out=np.full_like(h, np.nan), where=(h + m) > 0)
    hit_rate = (h + 0.5) / (h + m + 1.0)
    false_rate = (f + 0.5) / (f + c + 1.0)
    sedi_num = np.log(false_rate) - np.log(hit_rate) - np.log1p(-false_rate) + np.log1p(-hit_rate)
    sedi_den = np.log(false_rate) + np.log(hit_rate) + np.log1p(-false_rate) + np.log1p(-hit_rate)
    valid_sedi = (sedi_den != 0) & ((h + m) > 0) & ((f + c) > 0)
    sedi = np.divide(sedi_num, sedi_den, out=np.full_like(sedi_num, np.nan), where=valid_sedi)
    return {"pod": pod, "far": far, "csi": csi, "ets": ets, "frequency_bias": frequency_bias, "sedi": sedi}


def array_ci(values: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def metric_tables(
    common: pd.DataFrame,
    entries: Sequence[Dict[str, str]],
    thresholds: Sequence[float],
    quantiles: Sequence[float],
    bootstrap_iters: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = common["obs_rh2m"].to_numpy(dtype=float)
    dates = common["date"].to_numpy()
    quality_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    quantile_rows: List[Dict[str, object]] = []
    _, day_code, draws = bootstrap_day_draws(dates, bootstrap_iters, seed)
    for entry in entries:
        tag = entry["tag"]
        pred = common[f"forecast_{tag}"].to_numpy(dtype=float)
        row: Dict[str, object] = {
            "source": tag,
            "source_label": entry["label"],
            "n": len(common),
            "n_dates": int(pd.Series(dates).nunique()),
            **continuous_metrics(pred, obs),
        }
        continuous_boot = continuous_bootstrap_values(pred, obs, day_code, draws)
        for name in ("bias", "mae", "rmse", "corr"):
            lo, hi = array_ci(continuous_boot.get(name, np.array([])))
            row[f"{name}_ci_low"] = lo
            row[f"{name}_ci_high"] = hi
        quality_rows.append(row)
        for q in quantiles:
            pred_q = float(np.percentile(pred, q))
            obs_q = float(np.percentile(obs, q))
            quantile_rows.append(
                {
                    "source": tag,
                    "source_label": entry["label"],
                    "quantile": q,
                    "forecast_quantile": pred_q,
                    "observed_quantile": obs_q,
                    "quantile_bias": pred_q - obs_q,
                    "unit": "%",
                }
            )
        for threshold in thresholds:
            metrics = event_metrics(pred, obs, threshold)
            event_row: Dict[str, object] = {
                "source": tag,
                "source_label": entry["label"],
                "threshold": threshold,
                "unit": "%",
                "n": len(common),
                **metrics,
            }
            event_boot = event_bootstrap_values(pred, obs, threshold, day_code, draws)
            for name in ("pod", "far", "csi", "ets", "frequency_bias", "sedi"):
                lo, hi = array_ci(event_boot.get(name, np.array([])))
                event_row[f"{name}_ci_low"] = lo
                event_row[f"{name}_ci_high"] = hi
            event_row["roc_auc_ci_low"] = math.nan
            event_row["roc_auc_ci_high"] = math.nan
            event_row["average_precision_ci_low"] = math.nan
            event_row["average_precision_ci_high"] = math.nan
            event_rows.append(event_row)
    return pd.DataFrame(quality_rows), pd.DataFrame(event_rows), pd.DataFrame(quantile_rows)


def pairwise_skill_table(
    common: pd.DataFrame,
    entries: Sequence[Dict[str, str]],
    thresholds: Sequence[float],
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    obs = common["obs_rh2m"].to_numpy(dtype=float)
    dates = common["date"].to_numpy()
    _, day_code, draws = bootstrap_day_draws(dates, bootstrap_iters, seed + 91)
    rows: List[Dict[str, object]] = []
    metrics: List[Tuple[str, Optional[float], Callable[[np.ndarray, np.ndarray], float], str]] = [
        ("mae", None, lambda p, o: float(np.mean(np.abs(p - o))), "lower_better"),
        ("rmse", None, lambda p, o: float(np.sqrt(np.mean((p - o) ** 2))), "lower_better"),
        ("corr", None, corr_safe, "higher_better"),
    ]
    for threshold in thresholds:
        for name in ("csi", "ets", "sedi"):
            metrics.append(
                (name, threshold, lambda p, o, n=name, t=threshold: float(event_metrics(p, o, t)[n]), "higher_better")
            )
    for a, b in combinations(entries, 2):
        pa = common[f"forecast_{a['tag']}"] .to_numpy(dtype=float)
        pb = common[f"forecast_{b['tag']}"] .to_numpy(dtype=float)
        cont_a = continuous_bootstrap_values(pa, obs, day_code, draws)
        cont_b = continuous_bootstrap_values(pb, obs, day_code, draws)
        event_a = {t: event_bootstrap_values(pa, obs, t, day_code, draws) for t in thresholds}
        event_b = {t: event_bootstrap_values(pb, obs, t, day_code, draws) for t in thresholds}
        for metric, threshold, fn, direction in metrics:
            estimate = fn(pb, obs) - fn(pa, obs)
            if threshold is None:
                values = cont_b.get(metric, np.array([])) - cont_a.get(metric, np.array([]))
            else:
                values = event_b[threshold].get(metric, np.array([])) - event_a[threshold].get(metric, np.array([]))
            lo, hi = array_ci(values)
            rows.append(
                {
                    "source_a": a["tag"],
                    "label_a": a["label"],
                    "source_b": b["tag"],
                    "label_b": b["label"],
                    "metric": metric,
                    "threshold": threshold,
                    "direction": direction,
                    "difference_b_minus_a": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
                }
            )
    return pd.DataFrame(rows)


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


def savefig_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def plot_observation_quality(
    quality: pd.DataFrame,
    events: pd.DataFrame,
    quantiles: pd.DataFrame,
    entries: Sequence[Dict[str, str]],
    out_dir: Path,
) -> None:
    set_plot_style()
    labels = [entry["label"] for entry in entries]
    y = np.arange(len(entries))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for idx, entry in enumerate(entries):
        tag = entry["tag"]
        style = style_for(tag, entry["label"])
        row = quality[quality["source"] == tag].iloc[0]
        for ax, metric in ((axes[0, 0], "mae"), (axes[0, 1], "corr")):
            value = float(row[metric])
            lo = float(row.get(f"{metric}_ci_low", math.nan))
            hi = float(row.get(f"{metric}_ci_high", math.nan))
            xerr = np.array([[value - lo], [hi - value]]) if np.isfinite(lo) and np.isfinite(hi) else None
            ax.errorbar(value, idx, xerr=xerr, fmt=style["marker"], ms=5.5, color=style["color"], capsize=2.5, lw=1.1)
        p99 = quantiles[(quantiles["source"] == tag) & (quantiles["quantile"] == 99)]["quantile_bias"]
        if not p99.empty:
            axes[1, 0].barh(idx, float(p99.iloc[0]), color=style["color"], height=0.58)
    threshold_counts = events.groupby("threshold", as_index=False)["event_count"].max()
    eligible = threshold_counts[threshold_counts["event_count"] >= 20]
    if eligible.empty:
        eligible = threshold_counts[threshold_counts["event_count"] > 0]
    threshold = float(eligible["threshold"].max() if not eligible.empty else events["threshold"].min())
    extreme = events[events["threshold"] == threshold]
    width = 0.34
    for idx, entry in enumerate(entries):
        row = extreme[extreme["source"] == entry["tag"]].iloc[0]
        style = style_for(entry["tag"], entry["label"])
        for xpos, metric, alpha, hatch in (
            (idx - width / 2, "csi", 1.0, ""),
            (idx + width / 2, "sedi", 0.48, "//"),
        ):
            value = float(row[metric])
            lo = float(row.get(f"{metric}_ci_low", math.nan))
            hi = float(row.get(f"{metric}_ci_high", math.nan))
            yerr = np.array([[value - lo], [hi - value]]) if np.isfinite(lo) and np.isfinite(hi) else None
            axes[1, 1].bar(
                xpos,
                value,
                width,
                color=style["color"],
                alpha=alpha,
                hatch=hatch,
                edgecolor=style["color"],
                yerr=yerr,
                error_kw={"ecolor": "#202124", "elinewidth": 0.8, "capsize": 2},
            )
    axes[0, 0].set_title("a  Overall absolute error")
    axes[0, 0].set_xlabel("MAE (percentage points; lower is better)")
    axes[0, 1].set_title("b  Linear association")
    axes[0, 1].set_xlabel("Pearson correlation (higher is better)")
    axes[1, 0].set_title("c  Near-saturation tail displacement")
    axes[1, 0].set_xlabel("P99 forecast - observed (percentage points)")
    axes[1, 0].axvline(0, color="#202124", lw=0.8)
    axes[1, 1].set_title(f"d  Rare-event skill at RH2M >= {threshold:g}%")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_xticks(np.arange(len(entries)))
    axes[1, 1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1, 1].legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="#777777", label="CSI"),
            Patch(facecolor="#D0D0D0", edgecolor="#777777", hatch="//", label="SEDI"),
        ],
        loc="lower left",
    )
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    axes[1, 1].axhline(0, color="#202124", lw=0.8)
    axes[1, 1].set_ylim(-1.0, 1.02)
    fig.suptitle("Observation-anchored RH2M forecast quality", fontsize=10, y=1.01)
    fig.tight_layout()
    savefig_all(fig, out_dir, "fig_rh2m_observation_quality")
    plt.close(fig)


def plot_tail_curves(common: pd.DataFrame, entries: Sequence[Dict[str, str]], out_dir: Path) -> None:
    set_plot_style()
    thresholds = np.linspace(70, 100, 121)
    fig, ax = plt.subplots(figsize=(5.1, 3.55))
    obs = common["obs_rh2m"].to_numpy(dtype=float)
    rows = {"threshold": thresholds, "observed": [100 * np.mean(obs >= t) for t in thresholds]}
    ax.plot(thresholds, rows["observed"], color=SOURCE_STYLE["observed"]["color"], lw=2.2, label="Observed")
    for entry in entries:
        pred = common[f"forecast_{entry['tag']}"] .to_numpy(dtype=float)
        curve = np.asarray([100 * np.mean(pred >= t) for t in thresholds])
        rows[entry["tag"]] = curve
        style = style_for(entry["tag"], entry["label"])
        ax.plot(thresholds, curve, color=style["color"], lw=1.8, label=entry["label"])
    pd.DataFrame(rows).to_csv(out_dir / "rh2m_tail_curve_common_sample.csv", index=False)
    ax.axvspan(95, 100, color="#C96B4B", alpha=0.08, lw=0)
    ax.set_xlabel("RH2M threshold (%)")
    ax.set_ylabel("Frequency above threshold (%)")
    ax.set_title("Near-saturation tail frequency")
    ax.set_xlim(70, 100)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2)
    fig.tight_layout()
    savefig_all(fig, out_dir, "fig_rh2m_tail_multi_source_2025")
    plt.close(fig)


def plot_quantile_bias(quantiles: pd.DataFrame, entries: Sequence[Dict[str, str]], out_dir: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.1, 3.55))
    for entry in entries:
        cur = quantiles[quantiles["source"] == entry["tag"]].sort_values("quantile")
        style = style_for(entry["tag"], entry["label"])
        ax.plot(cur["quantile"], cur["quantile_bias"], color=style["color"], marker=style["marker"], ms=4, lw=1.7, label=entry["label"])
    ax.axhline(0, color="#202124", lw=0.8)
    ax.axvspan(95, 99.5, color="#C96B4B", alpha=0.08, lw=0)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Forecast - observed RH2M (percentage points)")
    ax.set_title("Quantile-dependent RH2M bias")
    ax.legend(ncol=2)
    fig.tight_layout()
    savefig_all(fig, out_dir, "fig_rh2m_quantile_bias_multi_source_2025")
    plt.close(fig)


def write_method_summary(out_dir: Path, common: pd.DataFrame, entries: Sequence[Dict[str, str]], args: argparse.Namespace) -> None:
    lines = [
        "# RH2M verification summary",
        "",
        f"- Common station/valid-time pairs: {len(common)} across {common['date'].nunique()} dates.",
        "- Sources: " + ", ".join(entry["label"] for entry in entries) + ".",
        "- Pangu RH2M proxies are excluded unless --allow_pangu_rh2m_proxy is explicitly set.",
        f"- Uncertainty: paired date-block bootstrap, {args.bootstrap_iters} iterations.",
        "- Overall quality: bias, MAE, RMSE and Pearson correlation.",
        "- Tail magnitude: P50/P75/P90/P95/P98/P99 forecast-minus-observation quantile bias.",
        "- Near-saturation events: POD, FAR, POFD, CSI, ETS, frequency bias, SEDI, ROC-AUC and average precision.",
        "",
        "## Interpretation",
        "",
        "A source is not judged from one score. Overall accuracy requires low MAE/RMSE and small bias; tail fidelity requires near-zero upper-quantile bias; rare-event usefulness requires high CSI/ETS/SEDI without excessive FAR or frequency bias.",
        "",
        "## Method references",
        "",
        "- Gneiting (2011), JASA, doi:10.1198/jasa.2011.r10138: point forecasts and consistent error scores.",
        "- Mason and Graham (2002), QJRMS, doi:10.1256/003590002320603584: ROC-area interpretation in atmospheric forecast verification.",
        "- Saito and Rehmsmeier (2015), PLOS ONE, doi:10.1371/journal.pone.0118432: precision-recall metrics for imbalanced events.",
        "- Ferro and Stephenson (2011), Weather and Forecasting, doi:10.1175/WAF-D-10-05030.1: SEDI for rare binary events.",
        "- Hamill (1999), Weather and Forecasting, doi:10.1175/1520-0434(1999)014<0155:HTFENP>2.0.CO;2: paired resampling while respecting spatial dependence.",
    ]
    (out_dir / "rh2m_verification_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_sources(args.sources)
    if not args.allow_pangu_rh2m_proxy:
        pangu = [entry["tag"] for entry in entries if source_family(entry["tag"], entry["label"]) == "pangu"]
        if pangu:
            raise ValueError(
                "Pangu RH2M proxy is excluded from the observation-anchored RH2M comparison. "
                f"Remove {pangu} or explicitly pass --allow_pangu_rh2m_proxy for a sensitivity analysis."
            )
    thresholds = split_numbers(args.thresholds, [90, 95, 98, 99])
    quantiles = split_numbers(args.quantiles, [50, 75, 90, 95, 98, 99])
    samples: Dict[str, pd.DataFrame] = {}
    tz_rows: List[pd.DataFrame] = []
    config_rows: List[Dict[str, object]] = []
    for entry in entries:
        print(f"[load] {entry['tag']}: {entry['path']}", flush=True)
        sample, tz_diag, config = load_source(entry, args)
        samples[entry["tag"]] = sample
        tz_rows.append(tz_diag)
        config_rows.append(config)
    shifts = {row["obs_time_shift_to_utc_hours"] for row in config_rows}
    if len(shifts) != 1:
        raise RuntimeError(f"Observation-time interpretation differs across sources: shifts={sorted(shifts)}")
    common = build_common_sample(samples, entries, args.min_pairs)
    identity_audit = audit_tianji_t2nd_identity(common, entries, args)
    (out_dir / "rh2m_tianji_t2nd_non_rh2m_identity_audit.json").write_text(
        json.dumps(identity_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if identity_audit.get("non_rh2m_identical") is False and not args.allow_non_rh2m_mismatch:
        raise RuntimeError(
            "Tianji and T2ND differ outside RH2M; controlled-replacement interpretation is invalid. "
            "Inspect rh2m_tianji_t2nd_non_rh2m_identity_audit.json or pass --allow_non_rh2m_mismatch for diagnostics only."
        )
    quality, events, quantiles_df = metric_tables(
        common, entries, thresholds, quantiles, args.bootstrap_iters, args.bootstrap_seed
    )
    pairwise = pairwise_skill_table(
        common, entries, thresholds, args.bootstrap_iters, args.bootstrap_seed
    )
    quality.to_csv(out_dir / "rh2m_source_quality_metrics.csv", index=False)
    events.to_csv(out_dir / "rh2m_extreme_event_metrics.csv", index=False)
    quantiles_df.to_csv(out_dir / "rh2m_quantile_metrics.csv", index=False)
    pairwise.to_csv(out_dir / "rh2m_pairwise_skill_differences.csv", index=False)
    common.to_csv(out_dir / "rh2m_common_sample.csv", index=False, float_format="%.6f")
    pd.concat(tz_rows, ignore_index=True).to_csv(out_dir / "rh2m_observation_time_alignment_diagnostics.csv", index=False)
    pd.DataFrame(config_rows).to_csv(out_dir / "rh2m_source_dataset_configs.csv", index=False)
    plot_observation_quality(quality, events, quantiles_df, entries, out_dir)
    plot_tail_curves(common, entries, out_dir)
    plot_quantile_bias(quantiles_df, entries, out_dir)
    write_method_summary(out_dir, common, entries, args)
    run_config = {
        "sources": entries,
        "obs_root": args.obs_root,
        "thresholds": thresholds,
        "quantiles": quantiles,
        "common_sample_n": len(common),
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_unit": "valid_date",
        "pangu_proxy_allowed": args.allow_pangu_rh2m_proxy,
        "tianji_t2nd_non_rh2m_identity_audit": identity_audit,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote RH2M verification outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
