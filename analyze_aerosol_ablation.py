#!/usr/bin/env python3
"""Analyze a validation-frozen Full-versus-No-PM low-visibility experiment.

The script keeps three questions separate:

1. Does a retrained model with PM10/PM2.5 inputs outperform the same training
   protocol without those inputs?
2. Is any skill difference conditional on humidity and the relative aerosol
   loading for the month?
3. Do the three manuscript events occupy contrasting compound environments?

PM values are used only as within-month validation-referenced ranks.  This is
intentional: historical mainline datasets use a legacy PM numeric scale, so
this analysis must not report absolute PM concentrations or thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = ("pmst_p_fog", "pmst_p_mist", "pmst_p_clear")
DEFAULT_EVENTS = (
    "2025-02-27 22:00:00",
    "2025-09-28 23:00:00",
    "2025-10-30 22:00:00",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-frozen aerosol ablation and event-regime analysis."
    )
    parser.add_argument("--full-val-dir", required=True)
    parser.add_argument("--no-pm-val-dir", required=True)
    parser.add_argument("--full-test-dir", required=True)
    parser.add_argument("--no-pm-test-dir", required=True)
    parser.add_argument("--full-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-fpr", type=float, default=0.04)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--event-window-hours", type=int, default=3)
    parser.add_argument(
        "--event-times",
        default=";".join(DEFAULT_EVENTS),
        help="Semicolon-separated UTC peak times.",
    )
    parser.add_argument("--min-cell-lowvis", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=250000)
    return parser.parse_args()


def read_ensemble_frame(directory: Path) -> pd.DataFrame:
    path = directory / "per_sample_eval.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"station_id", "y_true", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    time_name = "time_utc" if "time_utc" in frame.columns else "time"
    if time_name not in frame.columns:
        raise KeyError(f"{path}: expected time_utc or time")
    frame["__time"] = pd.to_datetime(frame[time_name], errors="raise")
    return frame


def assert_aligned(left: pd.DataFrame, right: pd.DataFrame, scope: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{scope}: row mismatch {len(left)} != {len(right)}")
    left_sid = left["station_id"].astype(str).to_numpy()
    right_sid = right["station_id"].astype(str).to_numpy()
    if not np.array_equal(left_sid, right_sid):
        raise ValueError(f"{scope}: station_id order differs between Full and No-PM")
    if not np.array_equal(left["__time"].to_numpy(), right["__time"].to_numpy()):
        raise ValueError(f"{scope}: time order differs between Full and No-PM")
    if not np.array_equal(
        left["y_true"].to_numpy(dtype=np.int16),
        right["y_true"].to_numpy(dtype=np.int16),
    ):
        raise ValueError(f"{scope}: y_true differs between Full and No-PM")


def lowvis_score(frame: pd.DataFrame) -> np.ndarray:
    probs = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite ensemble probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-4):
        raise ValueError("Ensemble probability rows do not sum to one")
    return probs[:, 0] + probs[:, 1]


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with tied scores evaluated as one threshold group."""
    y = np.asarray(y_true, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y))
    if positives == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    change = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    ends = np.flatnonzero(change)
    tp = np.cumsum(sorted_y, dtype=np.int64)[ends]
    fp = np.cumsum(~sorted_y, dtype=np.int64)[ends]
    recall = tp / float(positives)
    precision = tp / (tp + fp)
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def binary_counts(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(pred, dtype=bool)
    return {
        "tp": int(np.sum(y & p)),
        "fp": int(np.sum(~y & p)),
        "fn": int(np.sum(y & ~p)),
        "tn": int(np.sum(~y & ~p)),
    }


def metrics_from_counts(counts: Mapping[str, float]) -> Dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    tn = float(counts["tn"])
    return {
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "csi": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
    }


def binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    counts = binary_counts(y_true, pred)
    return {**{key: float(value) for key, value in counts.items()}, **metrics_from_counts(counts)}


def select_threshold_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float
) -> Tuple[float, Dict[str, float]]:
    """Select the highest-recall validation threshold under a fixed FPR cap."""
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr}")
    y = np.asarray(y_true, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    if len(y) != len(score) or len(y) == 0:
        raise ValueError("Empty or misaligned validation arrays")
    if not np.isfinite(score).all():
        raise ValueError("Non-finite validation scores")

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    change = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    ends = np.flatnonzero(change)
    cum_tp = np.cumsum(sorted_y, dtype=np.int64)[ends]
    cum_fp = np.cumsum(~sorted_y, dtype=np.int64)[ends]
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    fn = positives - cum_tp
    tn = negatives - cum_fp
    recall = np.divide(cum_tp, positives, dtype=float) if positives else np.full(len(ends), np.nan)
    fpr = np.divide(cum_fp, negatives, dtype=float) if negatives else np.full(len(ends), np.nan)
    csi_den = cum_tp + cum_fp + fn
    csi = np.divide(cum_tp, csi_den, out=np.zeros_like(cum_tp, dtype=float), where=csi_den > 0)

    eligible = np.flatnonzero(fpr <= target_fpr + 1.0e-12)
    if len(eligible) == 0:
        threshold = float(np.nextafter(np.max(score), np.inf))
        metrics = binary_metrics(y, score >= threshold)
        return threshold, metrics
    best = max(eligible.tolist(), key=lambda idx: (recall[idx], csi[idx], fpr[idx]))
    threshold = float(sorted_score[ends[best]])
    metrics = binary_metrics(y, score >= threshold)
    return threshold, metrics


def daily_counts(y_true: np.ndarray, pred: np.ndarray, times: pd.Series) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(pred, dtype=bool)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(times).dt.floor("D"),
            "tp": (y & p).astype(np.int64),
            "fp": (~y & p).astype(np.int64),
            "fn": (y & ~p).astype(np.int64),
            "tn": (~y & ~p).astype(np.int64),
        }
    )
    return frame.groupby("date", sort=True)[["tp", "fp", "fn", "tn"]].sum()


def bootstrap_metric_differences(
    full_daily: pd.DataFrame,
    no_pm_daily: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    if not full_daily.index.equals(no_pm_daily.index):
        raise ValueError("Full and No-PM daily blocks do not align")
    if iterations <= 0:
        return pd.DataFrame()
    full = full_daily[["tp", "fp", "fn", "tn"]].to_numpy(dtype=np.float64)
    no_pm = no_pm_daily[["tp", "fp", "fn", "tn"]].to_numpy(dtype=np.float64)
    n_days = len(full)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_days, size=(iterations, n_days))
    full_sum = full[draws].sum(axis=1)
    no_pm_sum = no_pm[draws].sum(axis=1)

    def vector_metrics(counts: np.ndarray) -> Dict[str, np.ndarray]:
        tp, fp, fn, tn = (counts[:, i] for i in range(4))
        return {
            "recall": np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0),
            "precision": np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0),
            "csi": np.divide(tp, tp + fp + fn, out=np.full_like(tp, np.nan), where=(tp + fp + fn) > 0),
            "fpr": np.divide(fp, fp + tn, out=np.full_like(tp, np.nan), where=(fp + tn) > 0),
        }

    full_metrics = vector_metrics(full_sum)
    no_pm_metrics = vector_metrics(no_pm_sum)
    rows: List[Dict[str, float]] = []
    for metric in ("recall", "precision", "csi", "fpr"):
        delta = full_metrics[metric] - no_pm_metrics[metric]
        rows.append(
            {
                "metric": metric,
                "delta_definition": "full_minus_no_pm",
                "bootstrap_mean": float(np.nanmean(delta)),
                "ci_low": float(np.nanquantile(delta, 0.025)),
                "ci_high": float(np.nanquantile(delta, 0.975)),
                "iterations": int(iterations),
                "date_blocks": int(n_days),
            }
        )
    return pd.DataFrame(rows)


def normalize_feature_name(name: str) -> str:
    compact = str(name).upper().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "PM10UGM3": "PM10",
        "PM10": "PM10",
        "PM25UGM3": "PM25",
        "PM2P5": "PM25",
        "PM25": "PM25",
        "RH2M": "RH2M",
        "WSPD10": "WSPD10",
        "PRECIP": "PRECIP",
    }
    return aliases.get(compact, compact)


def dynamic_layout(data_dir: Path) -> Tuple[int, int, List[str], Dict[str, object]]:
    cfg_path = data_dir / "dataset_build_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    order = cfg.get("dynamic_feature_order")
    if not isinstance(order, list) or not order:
        raise ValueError(f"{cfg_path}: dynamic_feature_order is missing")
    dyn_vars = int(cfg.get("dyn_vars", len(order)))
    window_size = int(cfg.get("window_size", 12))
    if len(order) != dyn_vars:
        raise ValueError(f"{cfg_path}: order length {len(order)} != dyn_vars {dyn_vars}")
    return window_size, dyn_vars, [str(item) for item in order], cfg


def feature_index(order: Sequence[str], name: str) -> int:
    target = normalize_feature_name(name)
    for index, feature in enumerate(order):
        if normalize_feature_name(feature) == target:
            return index
    raise KeyError(f"Dynamic feature {name} is missing from {list(order)}")


def sequence_stat(
    x_path: Path,
    window_size: int,
    dyn_vars: int,
    feature: int,
    statistic: str,
    chunk_size: int,
) -> np.ndarray:
    data = np.load(x_path, mmap_mode="r")
    indices = np.asarray([step * dyn_vars + feature for step in range(window_size)], dtype=np.int64)
    result = np.empty(len(data), dtype=np.float32)
    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        block = np.asarray(data[start:stop, indices], dtype=np.float32)
        if statistic == "last":
            values = block[:, -1]
        elif statistic == "mean":
            values = np.nanmean(block, axis=1)
        elif statistic == "max":
            values = np.nanmax(block, axis=1)
        else:
            raise ValueError(statistic)
        result[start:stop] = values.astype(np.float32)
    return result


def meta_times(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"meta_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    time_name = "time_utc" if "time_utc" in frame.columns else "time"
    if time_name not in frame.columns:
        raise KeyError(f"{path}: expected time_utc or time")
    frame["__time"] = pd.to_datetime(frame[time_name], errors="raise")
    return frame


def reference_percentile(
    reference: np.ndarray,
    reference_month: np.ndarray,
    values: np.ndarray,
    value_month: np.ndarray,
) -> np.ndarray:
    ranks = np.full(len(values), np.nan, dtype=np.float32)
    for month in range(1, 13):
        ref = np.asarray(reference[reference_month == month], dtype=np.float64)
        ref = np.sort(ref[np.isfinite(ref)])
        mask = (value_month == month) & np.isfinite(values)
        if len(ref) == 0 or not np.any(mask):
            continue
        ranks[mask] = np.searchsorted(ref, values[mask], side="right") / float(len(ref))
    return ranks


def conditional_metrics(
    y_true: np.ndarray,
    pred_full: np.ndarray,
    pred_no_pm: np.ndarray,
    rh: np.ndarray,
    pm_rank: np.ndarray,
    min_lowvis: int,
) -> pd.DataFrame:
    rh_bin = pd.cut(
        rh,
        bins=[-np.inf, 70.0, 90.0, np.inf],
        labels=["RH<70", "70<=RH<90", "RH>=90"],
        right=False,
    ).astype(str)
    pm_bin = pd.cut(
        pm_rank,
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=["PM_low", "PM_middle", "PM_high"],
        right=False,
    ).astype(str)
    rows: List[Dict[str, object]] = []
    for rh_label in ("RH<70", "70<=RH<90", "RH>=90"):
        for pm_label in ("PM_low", "PM_middle", "PM_high"):
            mask = (rh_bin == rh_label) & (pm_bin == pm_label)
            support = int(np.sum(y_true[mask]))
            if not np.any(mask):
                continue
            full = binary_metrics(y_true[mask], pred_full[mask])
            no_pm = binary_metrics(y_true[mask], pred_no_pm[mask])
            row: Dict[str, object] = {
                "rh_bin": rh_label,
                "pm_rank_bin": pm_label,
                "n": int(np.sum(mask)),
                "lowvis_support": support,
                "event_rate": float(np.mean(y_true[mask])),
                "reportable": bool(support >= min_lowvis),
            }
            for metric in ("recall", "precision", "csi", "fpr"):
                row[f"full_{metric}"] = full[metric]
                row[f"no_pm_{metric}"] = no_pm[metric]
                row[f"delta_{metric}"] = full[metric] - no_pm[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def environment_signature(
    rh_median: float,
    pm_rank_median: float,
    weak_wind_fraction: float,
    precip_fraction: float,
) -> str:
    if np.isfinite(precip_fraction) and precip_fraction >= 0.35:
        return "precipitation-associated compound environment"
    if rh_median >= 90.0 and pm_rank_median >= 2.0 / 3.0 and weak_wind_fraction >= 0.5:
        return "humid aerosol-rich weak-ventilation environment"
    if rh_median >= 90.0 and weak_wind_fraction >= 0.5:
        return "humid weak-ventilation environment"
    if pm_rank_median >= 2.0 / 3.0:
        return "aerosol-rich environment"
    return "mixed environment"


def event_summary(
    times: pd.Series,
    event_times: Iterable[pd.Timestamp],
    window_hours: int,
    y_true: np.ndarray,
    pred_full: np.ndarray,
    pred_no_pm: np.ndarray,
    rh: np.ndarray,
    pm_rank: np.ndarray,
    wind: np.ndarray,
    precip: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    time_values = pd.to_datetime(times)
    for rank, peak in enumerate(event_times, start=1):
        delta = (time_values - peak).abs()
        mask = delta <= pd.Timedelta(hours=window_hours)
        lowvis = mask & y_true
        if not np.any(mask):
            raise ValueError(f"No test rows found for event peak {peak}")
        full = binary_metrics(y_true[mask], pred_full[mask])
        no_pm = binary_metrics(y_true[mask], pred_no_pm[mask])
        event_rh = float(np.nanmedian(rh[lowvis])) if np.any(lowvis) else np.nan
        event_pm = float(np.nanmedian(pm_rank[lowvis])) if np.any(lowvis) else np.nan
        weak_wind = float(np.nanmean(wind[lowvis] < 3.0)) if np.any(lowvis) else np.nan
        wet = float(np.nanmean(precip[lowvis] > 0.0)) if np.any(lowvis) else np.nan
        rows.append(
            {
                "event_rank": rank,
                "peak_time_utc": peak,
                "window_hours_each_side": int(window_hours),
                "station_times": int(np.sum(mask)),
                "observed_lowvis_station_times": int(np.sum(lowvis)),
                "lowvis_rh2m_median": event_rh,
                "lowvis_month_relative_pm_rank_median": event_pm,
                "lowvis_weak_wind_fraction": weak_wind,
                "lowvis_precip_positive_fraction": wet,
                "descriptive_environment_signature": environment_signature(
                    event_rh, event_pm, weak_wind, wet
                ),
                "full_recall": full["recall"],
                "no_pm_recall": no_pm["recall"],
                "delta_recall": full["recall"] - no_pm["recall"],
                "full_csi": full["csi"],
                "no_pm_csi": no_pm["csi"],
                "delta_csi": full["csi"] - no_pm["csi"],
                "full_fpr": full["fpr"],
                "no_pm_fpr": no_pm["fpr"],
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    if selected.empty:
        return "_No rows._"

    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if not np.isfinite(value) else f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_val = read_ensemble_frame(Path(args.full_val_dir))
    no_pm_val = read_ensemble_frame(Path(args.no_pm_val_dir))
    full_test = read_ensemble_frame(Path(args.full_test_dir))
    no_pm_test = read_ensemble_frame(Path(args.no_pm_test_dir))
    assert_aligned(full_val, no_pm_val, "validation")
    assert_aligned(full_test, no_pm_test, "test")

    y_val = full_val["y_true"].to_numpy(dtype=np.int16) < 2
    y_test = full_test["y_true"].to_numpy(dtype=np.int16) < 2
    scores = {
        "full_val": lowvis_score(full_val),
        "no_pm_val": lowvis_score(no_pm_val),
        "full_test": lowvis_score(full_test),
        "no_pm_test": lowvis_score(no_pm_test),
    }
    full_threshold, full_val_metrics = select_threshold_at_fpr(
        y_val, scores["full_val"], args.target_fpr
    )
    no_pm_threshold, no_pm_val_metrics = select_threshold_at_fpr(
        y_val, scores["no_pm_val"], args.target_fpr
    )
    pred_full = scores["full_test"] >= full_threshold
    pred_no_pm = scores["no_pm_test"] >= no_pm_threshold
    full_test_metrics = binary_metrics(y_test, pred_full)
    no_pm_test_metrics = binary_metrics(y_test, pred_no_pm)

    overall_rows: List[Dict[str, object]] = []
    for arm, threshold, val_metrics, test_metrics, test_score in (
        ("full", full_threshold, full_val_metrics, full_test_metrics, scores["full_test"]),
        ("no_pm", no_pm_threshold, no_pm_val_metrics, no_pm_test_metrics, scores["no_pm_test"]),
    ):
        overall_rows.append(
            {
                "arm": arm,
                "validation_selected_threshold": threshold,
                "target_validation_fpr": float(args.target_fpr),
                "validation_recall": val_metrics["recall"],
                "validation_precision": val_metrics["precision"],
                "validation_csi": val_metrics["csi"],
                "validation_fpr": val_metrics["fpr"],
                "test_n": int(len(y_test)),
                "test_lowvis_support": int(np.sum(y_test)),
                "test_average_precision": average_precision(y_test, test_score),
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out_dir / "aerosol_ablation_overall_metrics.csv", index=False)

    difference_rows = []
    full_row = overall.set_index("arm").loc["full"]
    no_pm_row = overall.set_index("arm").loc["no_pm"]
    for metric in ("average_precision", "recall", "precision", "csi", "fpr"):
        column = f"test_{metric}"
        difference_rows.append(
            {
                "metric": metric,
                "full": float(full_row[column]),
                "no_pm": float(no_pm_row[column]),
                "delta_full_minus_no_pm": float(full_row[column] - no_pm_row[column]),
            }
        )
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(out_dir / "aerosol_ablation_metric_differences.csv", index=False)

    full_daily = daily_counts(y_test, pred_full, full_test["__time"])
    no_pm_daily = daily_counts(y_test, pred_no_pm, full_test["__time"])
    bootstrap = bootstrap_metric_differences(
        full_daily,
        no_pm_daily,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    bootstrap.to_csv(out_dir / "aerosol_ablation_date_block_bootstrap.csv", index=False)

    data_dir = Path(args.full_data_dir).expanduser().resolve()
    window_size, dyn_vars, order, dataset_cfg = dynamic_layout(data_dir)
    val_meta = meta_times(data_dir, "val")
    test_meta = meta_times(data_dir, "test")
    if len(val_meta) != len(full_val) or len(test_meta) != len(full_test):
        raise ValueError("Full ensemble rows do not match full dataset metadata")
    if not np.array_equal(test_meta["__time"].to_numpy(), full_test["__time"].to_numpy()):
        raise ValueError("Full test ensemble time order differs from full dataset metadata")

    feature_specs = {
        "rh": ("RH2M", "last"),
        "pm10": ("PM10", "mean"),
        "pm25": ("PM25", "mean"),
        "wind": ("WSPD10", "last"),
        "precip": ("PRECIP", "max"),
    }
    val_features: Dict[str, np.ndarray] = {}
    test_features: Dict[str, np.ndarray] = {}
    for key, (name, statistic) in feature_specs.items():
        index = feature_index(order, name)
        val_features[key] = sequence_stat(
            data_dir / "X_val.npy",
            window_size,
            dyn_vars,
            index,
            statistic,
            args.chunk_size,
        )
        test_features[key] = sequence_stat(
            data_dir / "X_test.npy",
            window_size,
            dyn_vars,
            index,
            statistic,
            args.chunk_size,
        )

    val_month = val_meta["__time"].dt.month.to_numpy(dtype=np.int8)
    test_month = test_meta["__time"].dt.month.to_numpy(dtype=np.int8)
    pm10_rank = reference_percentile(
        val_features["pm10"], val_month, test_features["pm10"], test_month
    )
    pm25_rank = reference_percentile(
        val_features["pm25"], val_month, test_features["pm25"], test_month
    )
    pm_rank = np.nanmean(np.column_stack([pm10_rank, pm25_rank]), axis=1).astype(np.float32)

    conditional = conditional_metrics(
        y_test,
        pred_full,
        pred_no_pm,
        test_features["rh"],
        pm_rank,
        args.min_cell_lowvis,
    )
    conditional.to_csv(out_dir / "rh_pm_conditional_skill.csv", index=False)

    event_times = [
        pd.Timestamp(value.strip())
        for value in str(args.event_times).split(";")
        if value.strip()
    ]
    events = event_summary(
        full_test["__time"],
        event_times,
        args.event_window_hours,
        y_test,
        pred_full,
        pred_no_pm,
        test_features["rh"],
        pm_rank,
        test_features["wind"],
        test_features["precip"],
    )
    events.to_csv(out_dir / "three_event_aerosol_environment_summary.csv", index=False)

    config = {
        "analysis": "validation_frozen_full_vs_no_pm_aerosol_ablation",
        "claim_scope": "predictive aerosol information; not causal aerosol attribution",
        "full_val_dir": str(Path(args.full_val_dir).expanduser().resolve()),
        "no_pm_val_dir": str(Path(args.no_pm_val_dir).expanduser().resolve()),
        "full_test_dir": str(Path(args.full_test_dir).expanduser().resolve()),
        "no_pm_test_dir": str(Path(args.no_pm_test_dir).expanduser().resolve()),
        "full_data_dir": str(data_dir),
        "dataset_protocol": dataset_cfg.get("protocol", ""),
        "dataset_pm_unit_note": (
            "PM is analyzed only by month-relative validation ranks; absolute PM values are not reported."
        ),
        "threshold_selection": {
            "split": "validation",
            "target_fpr": float(args.target_fpr),
            "full_threshold": full_threshold,
            "no_pm_threshold": no_pm_threshold,
        },
        "bootstrap": {
            "unit": "UTC date",
            "iterations": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
        },
        "events": [str(value) for value in event_times],
        "event_window_hours_each_side": int(args.event_window_hours),
    }
    (out_dir / "aerosol_ablation_run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Aerosol-informed low-visibility ablation report",
        "",
        "## Full versus No-PM test metrics",
        "",
        markdown_table(
            overall,
            [
                "arm",
                "validation_selected_threshold",
                "validation_fpr",
                "test_average_precision",
                "test_recall",
                "test_precision",
                "test_csi",
                "test_fpr",
            ],
        ),
        "",
        "## Paired UTC-date bootstrap",
        "",
        markdown_table(bootstrap, ["metric", "bootstrap_mean", "ci_low", "ci_high"]),
        "",
        "## Three event environments",
        "",
        markdown_table(
            events,
            [
                "event_rank",
                "peak_time_utc",
                "observed_lowvis_station_times",
                "lowvis_rh2m_median",
                "lowvis_month_relative_pm_rank_median",
                "lowvis_weak_wind_fraction",
                "lowvis_precip_positive_fraction",
                "descriptive_environment_signature",
                "delta_recall",
                "delta_csi",
            ],
        ),
        "",
        "## Interpretation boundary",
        "",
        "- Full-minus-No-PM differences test the predictive value of the PM channels under a matched retraining protocol.",
        "- RH-PM cells use validation-referenced, month-relative PM ranks; they do not report absolute aerosol concentration.",
        "- Event signatures describe co-occurring environments and are not fog, haze, precipitation, anthropogenic, or natural-source causal labels.",
        "- A causal aerosol claim requires source attribution and a physical intervention or stronger quasi-experimental design.",
    ]
    (out_dir / "aerosol_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[overall] {out_dir / 'aerosol_ablation_overall_metrics.csv'}", flush=True)
    print(f"[conditional] {out_dir / 'rh_pm_conditional_skill.csv'}", flush=True)
    print(f"[events] {out_dir / 'three_event_aerosol_environment_summary.csv'}", flush=True)
    print(f"[report] {out_dir / 'aerosol_ablation_report.md'}", flush=True)


if __name__ == "__main__":
    main()
