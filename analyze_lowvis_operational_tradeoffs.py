#!/usr/bin/env python3
"""Post-hoc operational audit for an existing low-visibility evaluation.

The audit reuses ``per_sample_eval.csv`` and the existing event-hourly CSVs. It
does not rerun inference, change the trained model, or tune a threshold on the
test set. When ``--selection-dir`` is supplied, thresholds are selected only
from that validation evaluation and then frozen before application to
``--eval-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "y_true",
    "pmst_pred",
    "pmst_p_fog",
    "pmst_p_mist",
    "pmst_p_clear",
)
OPTIONAL_COLUMNS = (
    "ifs_diagnostic_valid",
    "ifs_diagnostic_pred",
    "lead_hour",
    "init_time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit low-visibility precision/recall/FPR, calibration, and event footprint dynamics."
    )
    parser.add_argument("--eval-dir", required=True, help="Frozen test paper-evaluation directory.")
    parser.add_argument(
        "--selection-dir",
        default="",
        help="Optional validation evaluation used to select operating thresholds. Must differ from --eval-dir.",
    )
    parser.add_argument("--out-dir", default="", help="Default: <eval-dir>/operational_tradeoff_audit.")
    parser.add_argument("--threshold-points", type=int, default=1001)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--activity-fraction", type=float, default=0.20)
    parser.add_argument("--precision-targets", default="0.20,0.25,0.30,0.40")
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}_r{index:02d}")
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        index += 1


def _available_columns(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def load_per_sample(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "per_sample_eval.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-sample evaluation: {path}")
    columns = _available_columns(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    usecols = list(REQUIRED_COLUMNS) + [name for name in OPTIONAL_COLUMNS if name in columns]
    frame = pd.read_csv(path, usecols=usecols)
    for name in REQUIRED_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    valid = frame[list(REQUIRED_COLUMNS)].notna().all(axis=1)
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No finite evaluation rows in {path}")
    y = frame["y_true"].to_numpy(dtype=np.int16)
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("y_true must contain only Fog=0, Mist=1, Clear=2")
    probabilities = frame[["pmst_p_fog", "pmst_p_mist", "pmst_p_clear"]].to_numpy(dtype=float)
    if np.any(probabilities < -1e-6) or np.any(probabilities > 1.0 + 1e-6):
        raise ValueError("Class probabilities fall outside [0, 1]")
    sums = probabilities.sum(axis=1)
    if float(np.nanmax(np.abs(sums - 1.0))) > 5e-3:
        raise ValueError("Class probabilities do not sum to one within tolerance")
    return frame


def binary_counts(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, int]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    return {
        "tp": int(np.count_nonzero(truth & prediction)),
        "fp": int(np.count_nonzero(~truth & prediction)),
        "fn": int(np.count_nonzero(truth & ~prediction)),
        "tn": int(np.count_nonzero(~truth & ~prediction)),
    }


def metrics_from_counts(counts: Mapping[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = (int(counts[name]) for name in ("tp", "fp", "fn", "tn"))

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        **{name: int(counts[name]) for name in ("tp", "fp", "fn", "tn")},
        "precision": precision,
        "recall": recall,
        "csi": ratio(tp, tp + fp + fn),
        "fpr": ratio(fp, fp + tn),
        "far": ratio(fp, tp + fp),
        "predicted_positive_rate": ratio(tp + fp, tp + fp + fn + tn),
    }


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return metrics_from_counts(binary_counts(truth, prediction))


def threshold_curve(truth: np.ndarray, score: np.ndarray, points: int = 1001) -> pd.DataFrame:
    if points < 3:
        raise ValueError("threshold-points must be at least 3")
    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=float)
    if truth.shape != score.shape:
        raise ValueError("truth and score shapes differ")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    thresholds = np.linspace(0.0, 1.0, points, dtype=float)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_truth = truth[order].astype(np.int64)
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_truth, dtype=np.int64)))
    total_positive = int(positive_prefix[-1])
    total_negative = int(len(truth) - total_positive)
    starts = np.searchsorted(sorted_score, thresholds, side="left")
    rows = []
    for threshold, start in zip(thresholds, starts):
        predicted_count = int(len(truth) - start)
        tp = int(total_positive - positive_prefix[int(start)])
        fp = int(predicted_count - tp)
        counts = {"tp": tp, "fp": fp, "fn": total_positive - tp, "tn": total_negative - fp}
        row = {"threshold": float(threshold)}
        row.update(metrics_from_counts(counts))
        rows.append(row)
    return pd.DataFrame(rows)


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked_truth = truth[order]
    precision_at_rank = np.cumsum(ranked_truth) / np.arange(1, len(ranked_truth) + 1)
    return float(np.sum(precision_at_rank[ranked_truth]) / positives)


def closest_curve_row(curve: pd.DataFrame, column: str, target: float) -> pd.Series:
    values = pd.to_numeric(curve[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError(f"No finite {column} values in threshold curve")
    positions = np.flatnonzero(valid)
    distances = np.abs(values[valid] - float(target))
    candidates = positions[np.flatnonzero(distances == np.nanmin(distances))]
    if len(candidates) > 1:
        sub = curve.iloc[candidates]
        candidates = np.asarray([int(sub.sort_values(["csi", "threshold"], ascending=[False, False]).index[0])])
    return curve.loc[int(candidates[0])]


def select_thresholds(
    curve: pd.DataFrame,
    precision_targets: Sequence[float],
    ifs_fpr: Optional[float],
) -> pd.DataFrame:
    selected: List[Dict[str, float]] = []
    finite_csi = curve[np.isfinite(pd.to_numeric(curve["csi"], errors="coerce"))]
    if not finite_csi.empty:
        row = finite_csi.sort_values(["csi", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
        selected.append({"selection_rule": "max_csi", **row.to_dict()})
    for target in precision_targets:
        eligible = curve[np.isfinite(curve["precision"]) & (curve["precision"] >= float(target))]
        if eligible.empty:
            continue
        row = eligible.sort_values(["recall", "csi", "threshold"], ascending=[False, False, False]).iloc[0]
        selected.append({"selection_rule": f"precision_at_least_{target:.2f}", **row.to_dict()})
    if ifs_fpr is not None and np.isfinite(ifs_fpr):
        row = closest_curve_row(curve, "fpr", float(ifs_fpr))
        selected.append({"selection_rule": "matched_ifs_fpr", **row.to_dict()})
    return pd.DataFrame(selected).drop_duplicates(subset=["selection_rule"], keep="first")


def calibration_table(truth: np.ndarray, score: np.ndarray, bins: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if bins < 2:
        raise ValueError("calibration-bins must be at least 2")
    truth = np.asarray(truth, dtype=float)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.searchsorted(edges, score, side="right") - 1, bins - 1)
    indices = np.maximum(indices, 0)
    rows = []
    ece = 0.0
    total = len(score)
    for index in range(bins):
        mask = indices == index
        count = int(np.count_nonzero(mask))
        mean_score = float(np.mean(score[mask])) if count else float("nan")
        observed = float(np.mean(truth[mask])) if count else float("nan")
        gap = abs(mean_score - observed) if count else float("nan")
        if count:
            ece += count / total * gap
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": mean_score,
                "observed_frequency": observed,
                "absolute_gap": gap,
            }
        )
    summary = {
        "n": int(total),
        "prevalence": float(np.mean(truth)),
        "brier": float(np.mean((score - truth) ** 2)),
        "ece": float(ece),
    }
    return pd.DataFrame(rows), summary


def _first_last_active(offsets: np.ndarray, counts: np.ndarray, threshold: float) -> Tuple[float, float, bool, bool]:
    active = counts >= threshold
    if not np.any(active):
        return float("nan"), float("nan"), False, False
    active_positions = np.flatnonzero(active)
    return (
        float(offsets[active_positions[0]]),
        float(offsets[active_positions[-1]]),
        bool(active_positions[0] == 0),
        bool(active_positions[-1] == len(offsets) - 1),
    )


def _relative_bias(value: float, reference: float) -> float:
    return float((value - reference) / reference) if reference else float("nan")


def event_dynamics_frame(
    frame: pd.DataFrame,
    event_name: str,
    activity_fraction: float,
) -> List[Dict[str, object]]:
    if "hour_offset" not in frame:
        raise ValueError(f"Missing hour_offset for {event_name}")
    frame = frame.sort_values("hour_offset").reset_index(drop=True)
    offsets = pd.to_numeric(frame["hour_offset"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError(f"Non-finite hour_offset for {event_name}")
    rows: List[Dict[str, object]] = []
    for event_class, suffix in (("low_vis", "low_vis_count"), ("fog", "fog_count")):
        obs_col = f"obs_{suffix}"
        if obs_col not in frame:
            continue
        obs = pd.to_numeric(frame[obs_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        obs_peak = float(np.max(obs)) if len(obs) else 0.0
        activity_threshold = max(1.0, float(math.ceil(obs_peak * activity_fraction)))
        obs_onset, obs_diss, obs_left, obs_right = _first_last_active(offsets, obs, activity_threshold)
        obs_peak_offset = float(offsets[int(np.argmax(obs))]) if len(obs) else float("nan")
        obs_integral = float(np.sum(obs))
        for source, prefix in (("model", "pmst"), ("ifs", "ifs")):
            column = f"{prefix}_{suffix}"
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=float)
            onset, diss, left, right = _first_last_active(offsets, values, activity_threshold)
            peak = float(np.max(values)) if len(values) else 0.0
            peak_offset = float(offsets[int(np.argmax(values))]) if len(values) else float("nan")
            integral = float(np.sum(values))
            onset_error = onset - obs_onset if np.isfinite(onset) and np.isfinite(obs_onset) and not left and not obs_left else float("nan")
            diss_error = diss - obs_diss if np.isfinite(diss) and np.isfinite(obs_diss) and not right and not obs_right else float("nan")
            rows.append(
                {
                    "event": event_name,
                    "event_class": event_class,
                    "source": source,
                    "activity_fraction": float(activity_fraction),
                    "activity_count_threshold": activity_threshold,
                    "window_start_offset_h": float(offsets[0]),
                    "window_end_offset_h": float(offsets[-1]),
                    "obs_peak_count": obs_peak,
                    "pred_peak_count": peak,
                    "peak_count_relative_bias": _relative_bias(peak, obs_peak),
                    "obs_peak_offset_h": obs_peak_offset,
                    "pred_peak_offset_h": peak_offset,
                    "peak_timing_error_h": peak_offset - obs_peak_offset,
                    "obs_station_hours": obs_integral,
                    "pred_station_hours": integral,
                    "station_hours_relative_bias": _relative_bias(integral, obs_integral),
                    "obs_onset_offset_h": obs_onset,
                    "pred_onset_offset_h": onset,
                    "onset_error_h": onset_error,
                    "obs_dissipation_offset_h": obs_diss,
                    "pred_dissipation_offset_h": diss,
                    "dissipation_error_h": diss_error,
                    "obs_onset_left_censored": obs_left,
                    "pred_onset_left_censored": left,
                    "obs_dissipation_right_censored": obs_right,
                    "pred_dissipation_right_censored": right,
                }
            )
    return rows


def event_dynamics_rows(path: Path, activity_fraction: float) -> List[Dict[str, object]]:
    frame = pd.read_csv(path)
    event_name = path.stem.replace("fig9_", "").replace("_hourly_metrics", "")
    return event_dynamics_frame(frame, event_name, activity_fraction)


def _ifs_metrics(frame: pd.DataFrame) -> Optional[Dict[str, float]]:
    if "ifs_diagnostic_valid" not in frame or "ifs_diagnostic_pred" not in frame:
        return None
    valid = frame["ifs_diagnostic_valid"].astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()
    pred_values = pd.to_numeric(frame["ifs_diagnostic_pred"], errors="coerce").to_numpy(dtype=float)
    valid &= np.isfinite(pred_values)
    if not np.any(valid):
        return None
    truth = frame["y_true"].to_numpy(dtype=np.int16)[valid] < 2
    prediction = pred_values[valid].astype(np.int16) < 2
    metrics = binary_metrics(truth, prediction)
    metrics["n"] = int(np.count_nonzero(valid))
    return metrics


def _frame_arrays(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = frame["y_true"].to_numpy(dtype=np.int16) < 2
    score = frame["pmst_p_fog"].to_numpy(dtype=float) + frame["pmst_p_mist"].to_numpy(dtype=float)
    argmax_prediction = frame["pmst_pred"].to_numpy(dtype=np.int16) < 2
    return truth, score, argmax_prediction


def _write_plots(curve: pd.DataFrame, calibration: pd.DataFrame, out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime fallback
        print(f"[WARN] matplotlib unavailable; CSV outputs were still written: {exc}", flush=True)
        return []

    generated = []
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    axes[0].plot(curve["recall"], curve["precision"], color="#0F766E", lw=2)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Low-visibility precision–recall")
    axes[0].grid(alpha=0.25)
    axes[1].plot(curve["fpr"], curve["recall"], color="#B45309", lw=2)
    axes[1].set(xlabel="False-positive rate", ylabel="Recall", title="Recall at controlled false alarms")
    axes[1].grid(alpha=0.25)
    tradeoff_path = out_dir / "fig_lowvis_operating_tradeoffs.png"
    fig.savefig(tradeoff_path, dpi=220)
    plt.close(fig)
    generated.append(tradeoff_path.name)

    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    plotted = calibration[calibration["count"] > 0]
    ax.plot([0, 1], [0, 1], "--", color="#6B7280", lw=1.2, label="Perfect calibration")
    ax.plot(plotted["mean_probability"], plotted["observed_frequency"], "o-", color="#1D4ED8", lw=2, label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Low-visibility reliability", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    reliability_path = out_dir / "fig_lowvis_reliability.png"
    fig.savefig(reliability_path, dpi=220)
    plt.close(fig)
    generated.append(reliability_path.name)
    return generated


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    selection_dir = Path(args.selection_dir).expanduser().resolve() if args.selection_dir else None
    if selection_dir is not None and selection_dir == eval_dir:
        raise ValueError("--selection-dir must be an independent validation evaluation, not --eval-dir")
    requested_out = Path(args.out_dir).expanduser() if args.out_dir else eval_dir / "operational_tradeoff_audit"
    out_dir = resolve_output_dir(requested_out.resolve())
    precision_targets = [float(value.strip()) for value in args.precision_targets.split(",") if value.strip()]
    if any(value <= 0 or value >= 1 for value in precision_targets):
        raise ValueError("precision targets must fall strictly between 0 and 1")
    if not 0 < args.activity_fraction < 1:
        raise ValueError("activity-fraction must fall strictly between 0 and 1")

    test_frame = load_per_sample(eval_dir)
    test_truth, test_score, test_argmax = _frame_arrays(test_frame)
    test_curve = threshold_curve(test_truth, test_score, args.threshold_points)
    test_curve.to_csv(out_dir / "lowvis_threshold_curve_test_diagnostic.csv", index=False)

    test_ifs = _ifs_metrics(test_frame)
    diagnostic_points = select_thresholds(
        test_curve,
        precision_targets=precision_targets,
        ifs_fpr=test_ifs.get("fpr") if test_ifs else None,
    )
    if not diagnostic_points.empty:
        diagnostic_points.insert(0, "status", "test_diagnostic_not_for_threshold_selection")
        diagnostic_points.to_csv(out_dir / "test_diagnostic_matched_points.csv", index=False)

    operating_rows = []
    current_metrics = binary_metrics(test_truth, test_argmax)
    operating_rows.append({"source": "model_current_frozen_decision", "n": len(test_truth), **current_metrics})
    if test_ifs:
        operating_rows.append({"source": "ifs_diagnostic", **test_ifs})
    pd.DataFrame(operating_rows).to_csv(out_dir / "current_operating_point_metrics.csv", index=False)

    frozen_metrics = pd.DataFrame()
    if selection_dir is not None:
        selection_frame = load_per_sample(selection_dir)
        selection_truth, selection_score, _ = _frame_arrays(selection_frame)
        selection_curve = threshold_curve(selection_truth, selection_score, args.threshold_points)
        selection_curve.to_csv(out_dir / "lowvis_threshold_curve_validation.csv", index=False)
        selection_ifs = _ifs_metrics(selection_frame)
        selected = select_thresholds(
            selection_curve,
            precision_targets=precision_targets,
            ifs_fpr=selection_ifs.get("fpr") if selection_ifs else None,
        )
        selected.insert(0, "selection_source", str(selection_dir))
        selected.to_csv(out_dir / "selected_operating_points_validation.csv", index=False)
        rows = []
        for _, selected_row in selected.iterrows():
            threshold = float(selected_row["threshold"])
            metrics = binary_metrics(test_truth, test_score >= threshold)
            rows.append(
                {
                    "selection_rule": selected_row["selection_rule"],
                    "threshold_frozen_on_validation": threshold,
                    "selection_source": str(selection_dir),
                    "evaluation_source": str(eval_dir),
                    "n": len(test_truth),
                    **metrics,
                }
            )
        frozen_metrics = pd.DataFrame(rows)
        frozen_metrics.to_csv(out_dir / "frozen_operating_points_test_metrics.csv", index=False)

    calibration_frames = []
    calibration_summaries = []
    class_specs = (
        ("fog", test_frame["y_true"].to_numpy(dtype=np.int16) == 0, test_frame["pmst_p_fog"].to_numpy(dtype=float)),
        ("mist", test_frame["y_true"].to_numpy(dtype=np.int16) == 1, test_frame["pmst_p_mist"].to_numpy(dtype=float)),
        ("low_vis", test_truth, test_score),
    )
    lowvis_calibration = None
    for name, truth, score in class_specs:
        table, summary = calibration_table(truth, score, args.calibration_bins)
        table.insert(0, "target", name)
        calibration_frames.append(table)
        calibration_summaries.append({"target": name, **summary})
        if name == "low_vis":
            lowvis_calibration = table
    pd.concat(calibration_frames, ignore_index=True).to_csv(out_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(calibration_summaries).to_csv(out_dir / "calibration_summary.csv", index=False)

    event_rows: List[Dict[str, object]] = []
    for path in sorted(eval_dir.glob("fig9_event_*_hourly_metrics.csv")):
        event_rows.extend(event_dynamics_rows(path, args.activity_fraction))
    if event_rows:
        pd.DataFrame(event_rows).to_csv(out_dir / "event_footprint_dynamics.csv", index=False)

    generated_plots = _write_plots(test_curve, lowvis_calibration, out_dir) if lowvis_calibration is not None else []
    manifest = {
        "analysis": "lowvis_operational_tradeoff_audit_v1",
        "evaluation_source": str(eval_dir),
        "selection_source": str(selection_dir) if selection_dir else None,
        "threshold_selection_policy": (
            "validation_only_then_frozen_on_test" if selection_dir else "none_test_curve_is_diagnostic_only"
        ),
        "n_test": int(len(test_frame)),
        "lowvis_prevalence_test": float(np.mean(test_truth)),
        "lowvis_average_precision_test": average_precision(test_truth, test_score),
        "threshold_points": int(args.threshold_points),
        "calibration_bins": int(args.calibration_bins),
        "event_activity_fraction": float(args.activity_fraction),
        "event_onset_dissipation_policy": "errors_are_nan_when_observed_or_predicted_activity_is boundary-censored",
        "generated_plots": generated_plots,
        "frozen_operating_point_count": int(len(frozen_metrics)),
    }
    with (out_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[operational-audit] output={out_dir}", flush=True)


if __name__ == "__main__":
    main()
