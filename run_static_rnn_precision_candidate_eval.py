#!/usr/bin/env python3
"""Evaluate precision-loss candidate checkpoints on validation or frozen test data."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import run_static_rnn_lowvis_eval_journal as journal
import run_static_rnn_lowvis_sampling_eval as sampling
import plot_spatial as spatial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_json", default=os.environ.get("PAPER_EVAL_CONFIG", str(journal.VIS_EVAL_DIR / journal.DEFAULT_CONFIG_NAME)))
    p.add_argument("--manifest", required=True, help="TSV written by submit_static_rnn_precision_loss_candidates_chain.sh")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--base", default=str(journal.DEFAULT_BASE))
    p.add_argument("--train_dir", default=str(journal.DEFAULT_TRAIN_DIR))
    p.add_argument("--data_dir", default=journal.DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="static_rnn_precision_candidate_eval")
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--meta_time_shift_hours", type=float, default=0.0)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--run_event_eval", action="store_true")
    p.add_argument("--event_top_k", type=int, default=3)
    p.add_argument("--event_window_hours", type=int, default=3)
    p.add_argument("--event_min_fog_stations", type=int, default=80)
    p.add_argument("--event_min_regions", type=int, default=3)
    p.add_argument("--event_min_lon_span", type=float, default=10.0)
    p.add_argument("--event_min_lat_span", type=float, default=4.0)
    p.add_argument("--event_gap_hours", type=int, default=24)
    p.add_argument("--event_preferred_times", default="10-30 22:00")
    p.add_argument("--event_env_source", choices=["grid", "none"], default="none")
    p.add_argument("--event_env_max_events", type=int, default=3)
    p.add_argument("--event_env_rh2m_var", default="rh2m")
    p.add_argument("--event_env_rh2m_vmin", type=float, default=40.0)
    p.add_argument("--event_env_rh2m_vmax", type=float, default=100.0)
    p.add_argument("--event_env_tianji_template", default="/tj01/sd3op/userpp/pp_data/{init_yyyymmddhh}/stage26Q/multi_model_sources/{init_yyyymmddhh}/{variable}.nc")
    p.add_argument("--event_env_pm10_dir", default="pm10_data")
    p.add_argument("--event_env_pm10_var", default="pm10")
    p.add_argument("--event_env_pm10_vmin", type=float, default=0.0)
    p.add_argument("--event_env_pm10_vmax", type=float, default=240.0)
    p.add_argument("--ifs_vis_nc", default="VIS_IDW_KDTree_20250101_20251231.nc")
    p.add_argument("--ifs_vis_var", default="VIS")
    p.add_argument("--shp_path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    return p.parse_args()


def load_targets(args: argparse.Namespace, base: Path) -> List[tuple[str, str, journal.EvalTarget]]:
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute() and not manifest_path.is_file():
        train_relative = base / "train" / manifest_path
        if train_relative.is_file():
            manifest_path = train_relative
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Precision-loss manifest not found: {manifest_path}")
    args.manifest = str(manifest_path.resolve())
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    required = {"candidate_id", "candidate_label", "run_id", "s2_checkpoint"}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    targets: List[tuple[str, str, journal.EvalTarget]] = []
    for row in manifest.to_dict("records"):
        path = Path(row["s2_checkpoint"])
        if not path.is_absolute():
            path = journal.as_abs_under(base, path)
        if not path.exists():
            if args.allow_missing:
                print(f"[skip] missing checkpoint: {path}", flush=True)
                continue
            raise FileNotFoundError(path)
        label = row["candidate_label"] or row["candidate_id"]
        targets.append(
            (
                row["candidate_id"],
                row.get("seed", ""),
                journal.EvalTarget(label=label, run_id=row["run_id"], checkpoint=path, variant=journal.VARIANTS[0]),
            )
        )
    return targets


def validation_events(args: argparse.Namespace, meta: pd.DataFrame, y_cls: np.ndarray) -> pd.DataFrame:
    return spatial.detect_widespread_fog_events(
        meta,
        y_cls,
        top_k=args.event_top_k,
        window_hours=args.event_window_hours,
        min_fog_stations=args.event_min_fog_stations,
        min_regions=args.event_min_regions,
        min_lon_span=args.event_min_lon_span,
        min_lat_span=args.event_min_lat_span,
        gap_hours=args.event_gap_hours,
        required_valid_mask=None,
        preferred_event_times=args.event_preferred_times,
    )


def validation_event_metric_rows(
    events: pd.DataFrame,
    meta: pd.DataFrame,
    y_cls: np.ndarray,
    pred: np.ndarray,
    window_hours: int,
) -> List[Dict[str, object]]:
    times = pd.to_datetime(meta["time"], errors="coerce")
    rows: List[Dict[str, object]] = []
    for _, event in events.iterrows():
        peak = pd.Timestamp(event["peak_time"])
        hourly = []
        for offset in range(-int(window_hours), int(window_hours) + 1):
            mask = (times == peak + pd.Timedelta(hours=offset)).to_numpy()
            if int(mask.sum()) == 0:
                continue
            hourly.append(spatial._compute_case_metrics(y_cls[mask], pred[mask]))
        if not hourly:
            continue
        h = pd.DataFrame(hourly)
        rows.append(
            {
                "event_rank": int(event["event_rank"]),
                "peak_time": peak,
                "pmst_low_vis_recall_mean": float(h["low_vis_recall"].mean()),
                "pmst_low_vis_precision_mean": float(h["low_vis_precision"].mean()),
                "pmst_low_vis_far_mean": float(h["low_vis_far"].mean()),
                "pmst_low_vis_csi_mean": float(h["low_vis_csi"].mean()),
                "pmst_low_vis_fpr_mean": float(h["low_vis_fpr"].mean()),
                "pmst_low_vis_area_ratio_mean": float(h["low_vis_area_ratio"].mean()),
                "pmst_clear_to_fog_fp_window": int(h["clear_to_fog_fp"].sum()),
                "pmst_clear_to_mist_fp_window": int(h["clear_to_mist_fp"].sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args = journal.apply_paper_eval_config(
        args,
        "static_rnn_precision_candidate_eval",
        default_dir=journal.VIS_EVAL_DIR,
    )
    args.eval_split = args.split
    args.threshold_source = "argmax"
    args.fog_th = 0.5
    args.mist_th = 0.5
    args.save_outputs = True

    base = Path(args.base).expanduser()
    train_dir = Path(args.train_dir).expanduser()
    data_dir = journal.as_abs_under(base, args.data_dir)
    ckpt_dir = journal.as_abs_under(base, args.ckpt_dir)
    out_dir = journal.unique_dir(journal.as_abs_under(base, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path, y_cls, y_raw, meta = journal.load_main_data(
        data_dir,
        args.limit_samples,
        args.meta_time_shift_hours,
        split=args.split,
    )
    dataset_cfg = journal.read_dataset_build_config(data_dir)
    cfg_dyn = int(dataset_cfg.get("dyn_vars") or 0)
    cfg_fe = int(dataset_cfg.get("fe_dim") or 0)
    if cfg_dyn > 0 and cfg_fe > 0:
        dyn, fe = cfg_dyn, cfg_fe
    else:
        dyn, fe = journal.infer_layout_from_x(x_path, args.window_size)
    dynamic_feature_order = journal.config_dynamic_order(dataset_cfg, dyn)
    mod = journal.load_static_rnn_module(train_dir)
    layout = mod.Layout(
        window_size=args.window_size,
        dyn_vars=dyn,
        fe_dim=fe,
        dynamic_feature_order=dynamic_feature_order,
    )
    device = journal.resolve_device(args.device)
    targets = load_targets(args, base)
    if not targets:
        raise FileNotFoundError("No candidate checkpoints available")

    overall_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    fixed_validation_events = pd.DataFrame()
    requested_event_eval = bool(args.run_event_eval)
    if requested_event_eval and args.split == "val":
        fixed_validation_events = validation_events(args, meta, y_cls)
        fixed_validation_events.to_csv(out_dir / "validation_event_case_summary.csv", index=False)
        if fixed_validation_events.empty:
            raise RuntimeError("No validation events satisfy the pre-registered observation-only detection rules")
        args.run_event_eval = False
    for eval_id, (candidate_id, seed, target) in enumerate(targets):
        overall, per_class, confusion, events = sampling.evaluate_target(
            args,
            target,
            eval_id,
            x_path,
            y_cls,
            y_raw,
            meta,
            base,
            ckpt_dir,
            out_dir,
            mod,
            layout,
            device,
        )
        for row in [overall, *per_class, *confusion, *events]:
            row["candidate_id"] = candidate_id
            row["seed"] = seed
            row["eval_split"] = args.split
        overall_rows.append(overall)
        class_rows.extend(per_class)
        confusion_rows.extend(confusion)
        event_rows.extend(events)

        candidate_dir = out_dir / f"{eval_id}_{target.label}_event_eval"
        if not candidate_dir.exists():
            candidate_dir.mkdir(parents=True, exist_ok=True)
            np.save(candidate_dir / "probs.npy", np.load(out_dir / f"{eval_id}_{target.label}_probs.npy"))
            journal.export_per_sample(candidate_dir / "per_sample_eval.csv", meta, y_cls, y_raw, np.argmax(np.load(candidate_dir / "probs.npy"), axis=1), np.load(candidate_dir / "probs.npy"))

        if requested_event_eval and args.split == "val":
            probs = np.load(candidate_dir / "probs.npy")
            pred = np.argmax(probs, axis=1)
            validation_rows = validation_event_metric_rows(
                fixed_validation_events,
                meta,
                y_cls,
                pred,
                args.event_window_hours,
            )
            for row in validation_rows:
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "eval_split": args.split,
                        "run_id": target.run_id,
                        "label": target.label,
                    }
                )
            event_rows.extend(validation_rows)

    overall_df = pd.DataFrame(overall_rows)
    class_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    event_df = pd.DataFrame(event_rows)
    overall_path = out_dir / f"precision_candidates_{args.split}_overall_metrics.csv"
    class_path = out_dir / f"precision_candidates_{args.split}_per_class_metrics.csv"
    confusion_path = out_dir / f"precision_candidates_{args.split}_confusion_counts.csv"
    event_path = out_dir / f"precision_candidates_{args.split}_event_metrics.csv"
    overall_df.to_csv(overall_path, index=False, float_format="%.8f")
    class_df.to_csv(class_path, index=False, float_format="%.8f")
    confusion_df.to_csv(confusion_path, index=False, float_format="%.8f")
    if not event_df.empty:
        event_df.to_csv(event_path, index=False, float_format="%.8f")

    member_provenance: List[Dict[str, object]] = []
    for row in overall_rows:
        experiment_id = int(row["experiment_id"])
        label = str(row["label"])
        probability_path = out_dir / f"{experiment_id}_{label}_probs.npy"
        per_sample_path = out_dir / f"{experiment_id}_{label}_event_eval" / "per_sample_eval.csv"
        checkpoint_path = Path(str(row["checkpoint"])).expanduser()
        scaler_path = Path(str(row["scaler"])).expanduser()
        for required_path in (probability_path, per_sample_path, checkpoint_path, scaler_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing provenance input after candidate evaluation: {required_path}")
        member_provenance.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "seed": str(row["seed"]),
                "experiment_id": experiment_id,
                "label": label,
                "run_id": str(row["run_id"]),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": journal.sha256_file(checkpoint_path),
                "scaler": str(scaler_path.resolve()),
                "scaler_sha256": journal.sha256_file(scaler_path),
                "probability_file": str(probability_path.resolve()),
                "probability_sha256": journal.sha256_file(probability_path),
                "per_sample_file": str(per_sample_path.resolve()),
                "per_sample_sha256": journal.sha256_file(per_sample_path),
            }
        )

    config = {
        "experiment_status": "candidate_only",
        "replaces_mainline": False,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": journal.sha256_file(Path(args.manifest)),
        "eval_split": args.split,
        "threshold_source": "argmax",
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "layout": asdict(layout),
        "targets": [target.run_id for _, _, target in targets],
        "members": member_provenance,
        "dataset_provenance": {
            "y_file": str((data_dir / f"y_{args.split}.npy").resolve()),
            "y_sha256": journal.sha256_file(data_dir / f"y_{args.split}.npy"),
            "meta_file": str((data_dir / f"meta_{args.split}.csv").resolve()),
            "meta_sha256": journal.sha256_file(data_dir / f"meta_{args.split}.csv"),
        },
        "outputs": {
            "overall": str(overall_path),
            "per_class": str(class_path),
            "confusion": str(confusion_path),
            "events": str(event_path) if not event_df.empty else "",
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[table] {overall_path}")
    print(f"[table] {class_path}")
    print(f"[table] {confusion_path}")
    if not event_df.empty:
        print(f"[table] {event_path}")


if __name__ == "__main__":
    main()
