#!/usr/bin/env python3
"""Evaluate standalone low-visibility trajectory diffusion/Gaussian candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", default="/public/home/putianshu/vis_mlp/train")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--members", type=int, default=50)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold-file", default="")
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--mainline-probs", default="")
    p.add_argument("--mainline-meta", default="")
    p.add_argument("--mainline-thresholds", default="", help="Optional JSON with fog/mist thresholds")
    p.add_argument("--compare-result-dir", default="")
    p.add_argument("--seed", type=int, default=20260702)
    return p.parse_args()


def add_train_dir(path: str) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def target_classes(visibility: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(visibility).shape, 2, dtype=np.int16)
    out[np.asarray(visibility) < 1000.0] = 1
    out[np.asarray(visibility) < 500.0] = 0
    return out


def predict_classes(probabilities: np.ndarray, fog_threshold: float, mist_threshold: float) -> np.ndarray:
    probs = np.asarray(probabilities)
    pred = np.full(probs.shape[:-1], 2, dtype=np.int16)
    fog = (probs[..., 0] > fog_threshold) & (probs[..., 0] >= probs[..., 1])
    mist = (probs[..., 1] > mist_threshold) & (probs[..., 1] > probs[..., 0])
    pred[fog] = 0
    pred[mist] = 1
    return pred


def classification_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    out: Dict[str, float] = {}
    names = ("Fog", "Mist", "Clear")
    for cls, name in enumerate(names):
        tp = int(np.sum((pred == cls) & (y_true == cls)))
        fp = int(np.sum((pred == cls) & (y_true != cls)))
        fn = int(np.sum((pred != cls) & (y_true == cls)))
        out[f"{name}_P"] = safe_div(tp, tp + fp)
        out[f"{name}_R"] = safe_div(tp, tp + fn)
        out[f"{name}_CSI"] = safe_div(tp, tp + fp + fn)
        out[f"{name}_support"] = int(np.sum(y_true == cls))
    low_pred = pred <= 1
    low_true = y_true <= 1
    clear = y_true == 2
    hits = int(np.sum(low_pred & low_true))
    false = int(np.sum(low_pred & ~low_true))
    misses = int(np.sum(~low_pred & low_true))
    out.update(
        {
            "low_vis_precision": safe_div(hits, hits + false),
            "low_vis_recall": safe_div(hits, hits + misses),
            "low_vis_csi": safe_div(hits, hits + false + misses),
            "false_positive_rate": safe_div(int(np.sum(low_pred & clear)), int(np.sum(clear))),
            "accuracy": float(np.mean(y_true == pred)) if len(y_true) else math.nan,
            "n": int(len(y_true)),
        }
    )
    return out


def selection_score(metrics: Mapping[str, float]) -> float:
    return float(
        0.25 * metrics["Fog_CSI"]
        + 0.25 * metrics["Mist_CSI"]
        + 0.20 * metrics["Fog_R"]
        + 0.20 * metrics["Mist_R"]
        + 0.10 * metrics["low_vis_precision"]
        - 0.05 * metrics["false_positive_rate"]
    )


def fit_thresholds(probabilities: np.ndarray, visibility: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(visibility)
    probs = np.asarray(probabilities)[valid]
    truth = target_classes(np.asarray(visibility)[valid])
    grid = np.arange(0.10, 0.9501, 0.03)
    tiers = ((0.10, 0.10, 0.88), (0.05, 0.05, 0.84))
    best: Optional[Tuple[float, float, float, Dict[str, float], int]] = None
    for tier_id, (min_fog_p, min_mist_p, min_clear_r) in enumerate(tiers, start=1):
        for fog_th in grid:
            for mist_th in grid:
                pred = predict_classes(probs, float(fog_th), float(mist_th))
                metrics = classification_metrics(truth, pred)
                if metrics["Fog_P"] < min_fog_p or metrics["Mist_P"] < min_mist_p or metrics["Clear_R"] < min_clear_r:
                    continue
                score = selection_score(metrics)
                if best is None or score > best[0]:
                    best = (score, float(fog_th), float(mist_th), metrics, tier_id)
        if best is not None:
            break
    if best is None:
        pred = np.argmax(probs, axis=-1)
        metrics = classification_metrics(truth, pred)
        return {
            "selection_status": "no_feasible_gate",
            "decision_rule": "argmax",
            "fog_threshold": 0.5,
            "mist_threshold": 0.5,
            "tier": None,
            "validation_metrics": metrics,
        }
    return {
        "selection_status": "selected",
        "decision_rule": "mutual_thresholds",
        "fog_threshold": best[1],
        "mist_threshold": best[2],
        "tier": best[4],
        "selection_score": best[0],
        "validation_metrics": best[3],
    }


def reliability_table(probabilities: np.ndarray, truth: np.ndarray, mask: np.ndarray, bins: int = 10) -> pd.DataFrame:
    rows = []
    valid = np.asarray(mask, dtype=bool)
    for cls, name in ((0, "Fog"), (1, "Mist"), (3, "Low-vis")):
        prob = probabilities[..., 0] + probabilities[..., 1] if cls == 3 else probabilities[..., cls]
        obs = (truth <= 1) if cls == 3 else (truth == cls)
        p = prob[valid]
        o = obs[valid].astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        for b in range(bins):
            take = (p >= edges[b]) & ((p <= edges[b + 1]) if b == bins - 1 else (p < edges[b + 1]))
            rows.append(
                {
                    "class": name,
                    "bin_left": edges[b],
                    "bin_right": edges[b + 1],
                    "n": int(take.sum()),
                    "mean_probability": float(np.mean(p[take])) if take.any() else np.nan,
                    "observed_frequency": float(np.mean(o[take])) if take.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def energy_score(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = np.mean(np.linalg.norm(samples - target[:, None, :], axis=-1), axis=1)
    pair = samples[:, :, None, :] - samples[:, None, :, :]
    second = 0.5 * np.mean(np.linalg.norm(pair, axis=-1), axis=(1, 2))
    return first - second


def variogram_score(samples: np.ndarray, target: np.ndarray, lags: Sequence[int] = (1, 3, 6, 12)) -> np.ndarray:
    total = np.zeros(samples.shape[0], dtype=np.float64)
    count = 0
    for lag in lags:
        obs = np.abs(target[:, lag:] - target[:, :-lag]) ** 0.5
        pred = np.mean(np.abs(samples[:, :, lag:] - samples[:, :, :-lag]) ** 0.5, axis=1)
        total += np.mean((obs - pred) ** 2, axis=1)
        count += 1
    return total / max(count, 1)


def event_rows(samples: np.ndarray, visibility: np.ndarray, indices: np.ndarray, target_leads: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    leads = np.asarray(target_leads)
    for pos, row_index in enumerate(indices):
        obs = visibility[pos]
        obs_low = obs < 1000.0
        sampled_low = samples[pos] < 1000.0
        event_probability = float(np.mean(np.any(sampled_low, axis=1)))
        sampled_min = np.min(samples[pos], axis=1)
        sampled_duration = sampled_low.sum(axis=1)
        onset_values = []
        for member_mask in sampled_low:
            onset_values.append(float(leads[np.argmax(member_mask)]) if member_mask.any() else np.nan)
        obs_onset = float(leads[np.argmax(obs_low)]) if obs_low.any() else np.nan
        pred_onset = float(np.nanmedian(onset_values)) if np.isfinite(onset_values).any() else np.nan
        rows.append(
            {
                "row_index": int(row_index),
                "observed_event": int(obs_low.any()),
                "event_probability": event_probability,
                "observed_onset_lead": obs_onset,
                "predicted_onset_lead_median": pred_onset,
                "onset_error_h": pred_onset - obs_onset if np.isfinite(pred_onset) and np.isfinite(obs_onset) else np.nan,
                "observed_duration_h": int(obs_low.sum()),
                "predicted_duration_h_median": float(np.median(sampled_duration)),
                "duration_error_h": float(np.median(sampled_duration) - obs_low.sum()),
                "observed_min_visibility_m": float(np.min(obs)),
                "predicted_min_visibility_m_median": float(np.median(sampled_min)),
                "min_visibility_error_m": float(np.median(sampled_min) - np.min(obs)),
            }
        )
    return rows


class LeadAccumulator:
    def __init__(self, length: int):
        self.count = np.zeros(length, dtype=np.float64)
        self.abs_error = np.zeros(length, dtype=np.float64)
        self.sq_error = np.zeros(length, dtype=np.float64)
        self.crps = np.zeros(length, dtype=np.float64)
        self.coverage = {50: np.zeros(length), 80: np.zeros(length), 90: np.zeros(length)}
        self.width = {50: np.zeros(length), 80: np.zeros(length), 90: np.zeros(length)}

    def update(self, samples: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        valid = np.asarray(mask, dtype=bool)
        median = np.median(samples, axis=1)
        first = np.mean(np.abs(samples - target[:, None, :]), axis=1)
        pair = np.mean(np.abs(samples[:, :, None, :] - samples[:, None, :, :]), axis=(1, 2))
        crps = first - 0.5 * pair
        self.count += valid.sum(axis=0)
        self.abs_error += np.where(valid, np.abs(median - target), 0.0).sum(axis=0)
        self.sq_error += np.where(valid, (median - target) ** 2, 0.0).sum(axis=0)
        self.crps += np.where(valid, crps, 0.0).sum(axis=0)
        for level in (50, 80, 90):
            alpha = (100 - level) / 200.0
            lo, hi = np.quantile(samples, [alpha, 1.0 - alpha], axis=1)
            self.coverage[level] += np.where(valid, (target >= lo) & (target <= hi), 0.0).sum(axis=0)
            self.width[level] += np.where(valid, hi - lo, 0.0).sum(axis=0)

    def table(self, leads: Sequence[int]) -> pd.DataFrame:
        denom = np.maximum(self.count, 1.0)
        data: Dict[str, object] = {
            "lead_hour": list(leads),
            "n": self.count.astype(int),
            "MAE_m": self.abs_error / denom,
            "RMSE_m": np.sqrt(self.sq_error / denom),
            "CRPS_m": self.crps / denom,
        }
        for level in (50, 80, 90):
            data[f"coverage_{level}"] = self.coverage[level] / denom
            data[f"width_{level}_m"] = self.width[level] / denom
        return pd.DataFrame(data)


def summarize_leads(table: pd.DataFrame) -> Dict[str, float]:
    weights = np.maximum(table.n.to_numpy(), 1)
    out = {
        "MAE_m": float(np.average(table.MAE_m, weights=weights)),
        "RMSE_m": float(np.sqrt(np.average(table.RMSE_m ** 2, weights=weights))),
        "CRPS_m": float(np.average(table.CRPS_m, weights=weights)),
    }
    for level in (50, 80, 90):
        out[f"coverage_{level}"] = float(np.average(table[f"coverage_{level}"], weights=weights))
        out[f"width_{level}_m"] = float(np.average(table[f"width_{level}_m"], weights=weights))
    return out


def choose_device(value: str) -> torch.device:
    if value == "cpu" or (value == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda:0")


def evaluate_split(args, split: str, model, model_type: str, schedule, scaler, common, device, out_dir: Path) -> Dict[str, object]:
    dataset = common.LowVisTrajectoryDataset(args.data_dir, split, scaler)
    n = len(dataset) if args.limit_samples <= 0 else min(len(dataset), args.limit_samples)
    indices = np.arange(n, dtype=np.int64)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    quantile_levels = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)
    quantiles = np.lib.format.open_memmap(
        out_dir / f"trajectory_quantiles_{split}.npy", mode="w+", dtype="float32", shape=(n, len(quantile_levels), common.TARGET_LENGTH)
    )
    probabilities = np.lib.format.open_memmap(
        out_dir / f"class_probabilities_{split}.npy", mode="w+", dtype="float32", shape=(n, common.TARGET_LENGTH, 3)
    )
    lead_acc = LeadAccumulator(common.TARGET_LENGTH)
    energy_values: List[np.ndarray] = []
    variogram_values: List[np.ndarray] = []
    comparison_energy_values: List[np.ndarray] = []
    comparison_variogram_values: List[np.ndarray] = []
    events: List[Dict[str, object]] = []
    comparison_events: List[Dict[str, object]] = []
    best_cases: List[Tuple[float, int, np.ndarray, np.ndarray]] = []
    cursor = 0
    model.eval()
    for raw_batch in loader:
        batch = common.move_batch(raw_batch, device)
        if model_type == "diffusion":
            generated = common.ddim_sample(model, schedule, batch, members=args.members, steps=args.ddim_steps)
        else:
            generated = common.gaussian_sample(model, batch, members=args.members)
        samples = scaler.inverse_target(generated.cpu().numpy())
        visibility = raw_batch["visibility"].numpy()
        mask = raw_batch["target_mask"].numpy().astype(bool)
        bsz = samples.shape[0]
        slc = slice(cursor, cursor + bsz)
        quantiles[slc] = np.quantile(samples, quantile_levels, axis=1).transpose(1, 0, 2)
        probabilities[slc] = common.visibility_class_probabilities(samples)
        lead_acc.update(samples, visibility, mask)
        comparison_positions = np.asarray(common.COMPARISON_TARGET_POSITIONS, dtype=np.int64)
        complete = np.all(mask, axis=1)
        if complete.any():
            complete_samples = samples[complete]
            complete_vis = visibility[complete]
            energy_values.append(energy_score(complete_samples, complete_vis))
            variogram_values.append(variogram_score(complete_samples, complete_vis))
            complete_indices = np.arange(cursor, cursor + bsz)[complete]
            events.extend(event_rows(complete_samples, complete_vis, complete_indices, common.TARGET_LEADS))
            for local, global_idx in zip(np.where(complete)[0], complete_indices):
                best_cases.append((float(np.min(visibility[local])), int(global_idx), samples[local], visibility[local]))
        comparison_complete = np.all(mask[..., comparison_positions], axis=1)
        if comparison_complete.any():
            comparison_samples = samples[comparison_complete][..., comparison_positions]
            comparison_vis = visibility[comparison_complete][..., comparison_positions]
            comparison_indices = np.arange(cursor, cursor + bsz)[comparison_complete]
            comparison_energy_values.append(energy_score(comparison_samples, comparison_vis))
            comparison_variogram_values.append(variogram_score(comparison_samples, comparison_vis))
            comparison_events.extend(
                event_rows(comparison_samples, comparison_vis, comparison_indices, common.COMPARISON_LEADS)
            )
        cursor += bsz
        if cursor % max(args.batch_size * 20, 1) == 0 or cursor == n:
            print(f"[{split}] sampled {cursor}/{n}", flush=True)
    quantiles.flush()
    probabilities.flush()
    lead_table = lead_acc.table(common.TARGET_LEADS)
    lead_table.to_csv(out_dir / f"metrics_by_lead_{split}.csv", index=False)
    event_df = pd.DataFrame(events)
    event_df.to_csv(out_dir / f"event_trajectory_metrics_{split}.csv", index=False)
    comparison_event_df = pd.DataFrame(comparison_events)
    comparison_event_df.to_csv(out_dir / f"event_trajectory_metrics_{split}_12_48.csv", index=False)
    complete_n = int(len(event_df))
    overall: Dict[str, object] = {
        "split": split,
        "n_trajectories": n,
        "n_complete_trajectories": complete_n,
        "n_complete_trajectories_12_48": int(len(comparison_event_df)),
        "Energy_Score": float(np.mean(np.concatenate(energy_values))) if energy_values else np.nan,
        "Variogram_Score": float(np.mean(np.concatenate(variogram_values))) if variogram_values else np.nan,
        "Energy_Score_12_48": (
            float(np.mean(np.concatenate(comparison_energy_values))) if comparison_energy_values else np.nan
        ),
        "Variogram_Score_12_48": (
            float(np.mean(np.concatenate(comparison_variogram_values))) if comparison_variogram_values else np.nan
        ),
    }
    overall.update(summarize_leads(lead_table))
    comparison_table = lead_table[lead_table.lead_hour.isin(common.COMPARISON_LEADS)].reset_index(drop=True)
    overall.update({f"{key}_12_48": value for key, value in summarize_leads(comparison_table).items()})
    if not event_df.empty:
        overall["event_occurrence_brier"] = float(np.mean((event_df.event_probability - event_df.observed_event) ** 2))
        for key in ("onset_error_h", "duration_error_h", "min_visibility_error_m"):
            overall[f"{key}_MAE"] = float(np.nanmean(np.abs(event_df[key])))
    if not comparison_event_df.empty:
        overall["event_occurrence_brier_12_48"] = float(
            np.mean((comparison_event_df.event_probability - comparison_event_df.observed_event) ** 2)
        )
        for key in ("onset_error_h", "duration_error_h", "min_visibility_error_m"):
            overall[f"{key}_MAE_12_48"] = float(np.nanmean(np.abs(comparison_event_df[key])))
    (out_dir / f"probabilistic_metrics_{split}.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # Save full ensembles only for the three most severe complete observed cases.
    case_dir = out_dir / f"selected_cases_{split}"
    case_dir.mkdir(exist_ok=True)
    for rank, (_, row_index, case_samples, case_vis) in enumerate(sorted(best_cases, key=lambda x: x[0])[:3], start=1):
        np.savez_compressed(case_dir / f"case_{rank:02d}.npz", row_index=row_index, samples=case_samples, observation=case_vis)
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        q10, q50, q90 = np.quantile(case_samples, [0.1, 0.5, 0.9], axis=0)
        leads = np.asarray(common.TARGET_LEADS)
        ax.fill_between(leads, q10, q90, color="#4C78A8", alpha=0.25, label="10-90% ensemble")
        ax.plot(leads, q50, color="#1F4E79", lw=2.0, label="Ensemble median")
        ax.plot(leads, case_vis, color="black", lw=1.7, label="Observed visibility")
        ax.axhline(500, color="#C44E52", ls="--", lw=1)
        ax.axhline(1000, color="#DD8452", ls="--", lw=1)
        ax.set(xlabel="Lead time (h)", ylabel="Visibility (m)", title=f"Trajectory case {rank}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(case_dir / f"case_{rank:02d}.png", dpi=180)
        plt.close(fig)
    return {"overall": overall, "probabilities": probabilities, "dataset": dataset, "n": n}


def append_classification_outputs(
    split: str,
    result: Mapping[str, object],
    thresholds: Mapping[str, object],
    out_dir: Path,
    target_leads: Sequence[int],
    comparison_positions: Sequence[int],
) -> Dict[str, float]:
    dataset = result["dataset"]
    n = int(result["n"])
    visibility = np.asarray(dataset.visibility[:n], dtype=np.float32)
    mask = np.asarray(dataset.target_mask[:n], dtype=bool) & np.isfinite(visibility)
    probabilities = np.asarray(result["probabilities"])
    truth = target_classes(visibility)
    if thresholds.get("decision_rule", "mutual_thresholds") == "argmax":
        pred = np.argmax(probabilities, axis=-1).astype(np.int16)
    else:
        pred = predict_classes(probabilities, float(thresholds["fog_threshold"]), float(thresholds["mist_threshold"]))
    metrics = classification_metrics(truth[mask], pred[mask])
    metrics["Brier_Fog"] = float(np.mean((probabilities[..., 0][mask] - (truth[mask] == 0)) ** 2))
    metrics["Brier_Mist"] = float(np.mean((probabilities[..., 1][mask] - (truth[mask] == 1)) ** 2))
    metrics["Brier_LowVis"] = float(np.mean(((probabilities[..., 0] + probabilities[..., 1])[mask] - (truth[mask] <= 1)) ** 2))
    (out_dir / f"classification_metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    reliability = reliability_table(probabilities, truth, mask)
    reliability.to_csv(out_dir / f"reliability_{split}.csv", index=False)

    comparison_positions = np.asarray(comparison_positions, dtype=np.int64)
    comparison_probs = probabilities[:, comparison_positions]
    comparison_truth = truth[:, comparison_positions]
    comparison_mask = mask[:, comparison_positions]
    comparison_pred = pred[:, comparison_positions]
    comparison_metrics = classification_metrics(
        comparison_truth[comparison_mask], comparison_pred[comparison_mask]
    )
    comparison_metrics["Brier_Fog"] = float(
        np.mean((comparison_probs[..., 0][comparison_mask] - (comparison_truth[comparison_mask] == 0)) ** 2)
    )
    comparison_metrics["Brier_Mist"] = float(
        np.mean((comparison_probs[..., 1][comparison_mask] - (comparison_truth[comparison_mask] == 1)) ** 2)
    )
    comparison_metrics["Brier_LowVis"] = float(
        np.mean(
            (
                (comparison_probs[..., 0] + comparison_probs[..., 1])[comparison_mask]
                - (comparison_truth[comparison_mask] <= 1)
            )
            ** 2
        )
    )
    (out_dir / f"classification_metrics_{split}_12_48.json").write_text(
        json.dumps(comparison_metrics, indent=2), encoding="utf-8"
    )
    reliability_table(comparison_probs, comparison_truth, comparison_mask).to_csv(
        out_dir / f"reliability_{split}_12_48.csv", index=False
    )
    metrics.update({f"{key}_12_48": value for key, value in comparison_metrics.items()})

    meta = pd.read_csv(dataset.data_dir / f"meta_{split}.csv", nrows=n)
    flat = pd.DataFrame(
        {
            "station_id": np.repeat(meta.station_id.to_numpy(), truth.shape[1])[mask.reshape(-1)],
            "truth": truth.reshape(-1)[mask.reshape(-1)],
            "pred": pred.reshape(-1)[mask.reshape(-1)],
        }
    )
    station_rows = []
    for station, group in flat.groupby("station_id", sort=False):
        row = {"station_id": station}
        row.update(classification_metrics(group.truth.to_numpy(), group.pred.to_numpy()))
        station_rows.append(row)
    pd.DataFrame(station_rows).to_csv(out_dir / f"station_metrics_{split}.csv", index=False)

    lead_rows = []
    for pos, lead in enumerate(target_leads):
        valid = mask[:, pos]
        row = {"lead_hour": lead}
        row.update(classification_metrics(truth[:, pos][valid], pred[:, pos][valid]))
        lead_rows.append(row)
    pd.DataFrame(lead_rows).to_csv(out_dir / f"classification_by_lead_{split}.csv", index=False)
    return metrics


def plot_summary(out_dir: Path, split: str) -> None:
    lead = pd.read_csv(out_dir / f"metrics_by_lead_{split}.csv")
    cls = pd.read_csv(out_dir / f"classification_by_lead_{split}.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)
    axes[0, 0].plot(lead.lead_hour, lead.CRPS_m, color="#4C78A8")
    axes[0, 0].set_ylabel("CRPS (m)")
    axes[0, 1].plot(lead.lead_hour, lead.coverage_90, color="#59A14F")
    axes[0, 1].axhline(0.9, color="black", ls="--", lw=1)
    axes[0, 1].set_ylabel("90% coverage")
    axes[1, 0].plot(cls.lead_hour, cls.low_vis_csi, label="Low-vis CSI", color="#E15759")
    axes[1, 0].plot(cls.lead_hour, cls.low_vis_recall, label="Low-vis recall", color="#F28E2B")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(cls.lead_hour, cls.false_positive_rate, color="#B07AA1")
    axes[1, 1].set_ylabel("Low-vis false-positive rate")
    for ax in axes[1]:
        ax.set_xlabel("Lead time (h)")
    fig.suptitle(f"Trajectory probabilistic evaluation: {split}")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_trajectory_summary_{split}.png", dpi=180)
    plt.close(fig)

    rel = pd.read_csv(out_dir / f"reliability_{split}.csv")
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for name, group in rel.groupby("class"):
        group = group[group.n > 0]
        ax.plot(group.mean_probability, group.observed_frequency, marker="o", ms=3, label=name)
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
    ax.set(xlabel="Forecast probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_reliability_{split}.png", dpi=180)
    plt.close(fig)

    station = pd.read_csv(out_dir / f"station_metrics_{split}.csv")
    station = station[station.Fog_support + station.Mist_support >= 5].nlargest(20, "low_vis_csi")
    if not station.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(station))
        ax.bar(x, station.low_vis_csi, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(station.station_id.astype(str), rotation=60, ha="right", fontsize=7)
        ax.set(xlabel="Station", ylabel="Low-vis CSI", title=f"Top supported station skill: {split}")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_station_lowvis_csi_{split}.png", dpi=180)
        plt.close(fig)


def compare_result_dirs(out_dir: Path, other_dir: Path) -> None:
    rows = []
    for label, path in (("candidate", out_dir), ("comparison", other_dir)):
        metrics_path = path / "overall_metrics.json"
        if not metrics_path.exists():
            return
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"model": label}
        row.update(payload.get("test", {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "candidate_vs_comparison_metrics.csv", index=False)
    lead_a = out_dir / "metrics_by_lead_test.csv"
    lead_b = other_dir / "metrics_by_lead_test.csv"
    cls_a = out_dir / "classification_by_lead_test.csv"
    cls_b = other_dir / "classification_by_lead_test.csv"
    if lead_a.exists() and lead_b.exists():
        left = pd.read_csv(lead_a).add_suffix("_candidate").rename(columns={"lead_hour_candidate": "lead_hour"})
        right = pd.read_csv(lead_b).add_suffix("_comparison").rename(columns={"lead_hour_comparison": "lead_hour"})
        merged = left.merge(right, on="lead_hour", how="inner", validate="one_to_one")
        merged.to_csv(out_dir / "candidate_vs_comparison_by_lead.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
        for metric, ax in (("CRPS_m", axes[0]), ("MAE_m", axes[1])):
            ax.plot(merged.lead_hour, merged[f"{metric}_candidate"], label="Candidate", color="#4C78A8")
            ax.plot(merged.lead_hour, merged[f"{metric}_comparison"], label="Comparison", color="#F28E2B")
            ax.set(xlabel="Lead time (h)", ylabel=metric.replace("_", " "))
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_candidate_vs_comparison_by_lead.png", dpi=180)
        plt.close(fig)
    if cls_a.exists() and cls_b.exists():
        left = pd.read_csv(cls_a).add_suffix("_candidate").rename(columns={"lead_hour_candidate": "lead_hour"})
        right = pd.read_csv(cls_b).add_suffix("_comparison").rename(columns={"lead_hour_comparison": "lead_hour"})
        left.merge(right, on="lead_hour", how="inner", validate="one_to_one").to_csv(
            out_dir / "candidate_vs_comparison_classification_by_lead.csv", index=False
        )


def exact_mainline_comparison(
    args: argparse.Namespace,
    test_result: Mapping[str, object],
    candidate_thresholds: Mapping[str, object],
    out_dir: Path,
    comparison_leads: Sequence[int],
    comparison_positions: Sequence[int],
) -> None:
    """Join mainline probabilities only on exact init/station/lead keys."""
    if not (args.mainline_probs and args.mainline_meta):
        if args.mainline_probs or args.mainline_meta:
            raise ValueError("Both --mainline-probs and --mainline-meta are required")
        return
    main_probs = np.load(args.mainline_probs, mmap_mode="r")
    main_meta = pd.read_csv(args.mainline_meta)
    if len(main_meta) != len(main_probs):
        raise ValueError("Mainline probabilities and metadata have different row counts")
    required = {"init_time", "station_id", "lead_hour"}
    missing = required - set(main_meta.columns)
    if missing:
        raise ValueError(f"Mainline metadata lacks exact trajectory keys: {sorted(missing)}")
    dataset = test_result["dataset"]
    n = int(test_result["n"])
    meta = pd.read_csv(dataset.data_dir / "meta_test.csv", nrows=n)
    leads = np.asarray(comparison_leads, dtype=np.int16)
    positions = np.asarray(comparison_positions, dtype=np.int64)
    visibility = np.asarray(dataset.visibility[:n], dtype=np.float32)[:, positions]
    mask = np.asarray(dataset.target_mask[:n], dtype=bool)[:, positions] & np.isfinite(visibility)
    candidate_probs = np.asarray(test_result["probabilities"])[:, positions]
    candidate = pd.DataFrame(
        {
            "init_time": np.repeat(pd.to_datetime(meta.init_time).to_numpy(), len(leads)),
            "station_id": np.repeat(meta.station_id.astype(str).to_numpy(), len(leads)),
            "lead_hour": np.tile(leads, n),
            "visibility": visibility.reshape(-1),
            "valid": mask.reshape(-1),
            "candidate_p0": candidate_probs[..., 0].reshape(-1),
            "candidate_p1": candidate_probs[..., 1].reshape(-1),
            "candidate_p2": candidate_probs[..., 2].reshape(-1),
        }
    )
    main = main_meta.copy()
    main["init_time"] = pd.to_datetime(main["init_time"])
    main["station_id"] = main["station_id"].astype(str)
    main["lead_hour"] = pd.to_numeric(main["lead_hour"], errors="coerce").round().astype("Int64")
    for cls in range(3):
        main[f"mainline_p{cls}"] = np.asarray(main_probs[:, cls], dtype=np.float32)
    keep_cols = ["init_time", "station_id", "lead_hour", "mainline_p0", "mainline_p1", "mainline_p2"]
    if main.duplicated(["init_time", "station_id", "lead_hour"]).any():
        raise ValueError("Mainline metadata contains duplicate exact trajectory keys")
    joined = candidate.merge(main[keep_cols], on=["init_time", "station_id", "lead_hour"], how="inner", validate="one_to_one")
    joined = joined[joined.valid]
    if len(joined) == 0:
        raise ValueError("Exact mainline trajectory join produced zero valid rows")
    truth = target_classes(joined.visibility.to_numpy())
    candidate_matrix = joined[["candidate_p0", "candidate_p1", "candidate_p2"]].to_numpy()
    if candidate_thresholds.get("decision_rule", "mutual_thresholds") == "argmax":
        candidate_pred = np.argmax(candidate_matrix, axis=1)
    else:
        candidate_pred = predict_classes(
            candidate_matrix,
            float(candidate_thresholds["fog_threshold"]),
            float(candidate_thresholds["mist_threshold"]),
        )
    main_matrix = joined[["mainline_p0", "mainline_p1", "mainline_p2"]].to_numpy()
    if args.mainline_thresholds:
        main_thresholds = json.loads(Path(args.mainline_thresholds).read_text(encoding="utf-8"))
        if isinstance(main_thresholds.get("thresholds"), dict):
            main_thresholds = main_thresholds["thresholds"]
        main_pred = predict_classes(
            main_matrix,
            float(main_thresholds.get("fog_threshold", main_thresholds.get("fog", 0.5))),
            float(main_thresholds.get("mist_threshold", main_thresholds.get("mist", 0.5))),
        )
        main_rule = "supplied_validation_thresholds"
    else:
        main_pred = np.argmax(main_matrix, axis=1)
        main_rule = "argmax_no_threshold_file"
    rows = []
    for label, pred in (("trajectory_candidate", candidate_pred), ("static_rnn_mainline", main_pred)):
        row = {"model": label}
        row.update(classification_metrics(truth, pred))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "exact_mainline_comparison.csv", index=False)
    diagnostics = {
        "candidate_rows": int(candidate.valid.sum()),
        "mainline_rows": int(len(main)),
        "exact_matched_rows": int(len(joined)),
        "candidate_rule": "candidate_validation_thresholds",
        "mainline_rule": main_rule,
    }
    (out_dir / "exact_mainline_comparison_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    add_train_dir(args.train_dir)
    import lowvis_trajectory_diffusion as common

    common.seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    common.read_dataset_config(args.data_dir)
    model_config = checkpoint["model_config"]
    model_type = str(model_config["model_type"])
    model = common.model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = choose_device(args.device)
    model = model.to(device).eval()
    schedule = common.DiffusionSchedule(
        int(model_config.get("diffusion_steps", 1000)),
        ddim_clip_x0=float(model_config.get("ddim_clip_x0", 0.0)),
    ).to(device)
    scaler = common.TrajectoryScaler.from_dict(checkpoint["scaler"])
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    results = {}
    for split in splits:
        results[split] = evaluate_split(args, split, model, model_type, schedule, scaler, common, device, out_dir)

    if args.threshold_file:
        thresholds = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    else:
        if "val" not in results:
            raise ValueError("Validation split is required to fit thresholds when --threshold-file is absent")
        val = results["val"]
        vis = np.asarray(val["dataset"].visibility[: val["n"]], dtype=np.float32)
        mask = np.asarray(val["dataset"].target_mask[: val["n"]], dtype=bool)
        positions = np.asarray(common.COMPARISON_TARGET_POSITIONS, dtype=np.int64)
        thresholds = fit_thresholds(
            np.asarray(val["probabilities"])[:, positions],
            vis[:, positions],
            mask[:, positions],
        )
        thresholds["fitted_leads"] = list(common.COMPARISON_LEADS)
        (out_dir / "validation_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    combined = {}
    for split, result in results.items():
        classification = append_classification_outputs(
            split,
            result,
            thresholds,
            out_dir,
            common.TARGET_LEADS,
            common.COMPARISON_TARGET_POSITIONS,
        )
        combined[split] = dict(result["overall"])
        combined[split].update(classification)
        plot_summary(out_dir, split)
    (out_dir / "overall_metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    run_config = {
        "candidate_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "model_config": model_config,
        "checkpoint_weights_source": checkpoint.get("weights_source", "online"),
        "checkpoint_ema_updates": int(checkpoint.get("ema_updates", 0)),
        "members": args.members,
        "ddim_steps": args.ddim_steps,
        "splits": splits,
        "thresholds": thresholds,
        "mainline_comparison_requested": bool(args.mainline_probs or args.mainline_meta),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if args.compare_result_dir:
        compare_result_dirs(out_dir, Path(args.compare_result_dir))
    if "test" in results:
        exact_mainline_comparison(
            args,
            results["test"],
            thresholds,
            out_dir,
            common.COMPARISON_LEADS,
            common.COMPARISON_TARGET_POSITIONS,
        )
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
