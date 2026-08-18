#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rerun only Fig. 9 event figures from an existing Static-RNN eval directory.

This script does not rerun model inference. It reads ``per_sample_eval.csv`` and
the existing event summary, optionally replaces selected event centers, then
regenerates the event-only figures:

* per-event spatial and metric panels,
* three-event peak grid,
* three-event footprint evolution,
* event environment grids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent
if str(VIS_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_EVAL_DIR))

import run_static_rnn_lowvis_eval_journal as journal
from plot_spatial import (
    compute_event_hourly_metrics,
    plot_event_metric_comparison,
    plot_event_summary_comparison,
    plot_three_events_footprint_row,
    plot_three_events_peak_row,
    plot_widespread_event_panels,
    summarize_event_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerun Static-RNN event figures from existing per-sample eval output.")
    p.add_argument("--eval_dir", required=True, help="Existing eval dir containing per_sample_eval.csv.")
    p.add_argument("--out_dir", default="", help="Output dir. Defaults to <eval_dir>/event_rerun_custom.")
    p.add_argument("--base", default="/public/home/putianshu/vis_mlp")
    p.add_argument("--event_summary", default="", help="Optional event_case_summary.csv path.")
    p.add_argument(
        "--replace_event",
        action="append",
        default=[],
        help="Replace an event center as rank=YYYY-mm-ddTHH:MM:SS, e.g. 1=2025-10-30T22:00:00. Can be repeated.",
    )
    p.add_argument("--window_hours", type=int, default=3)
    p.add_argument(
        "--event_window_hours",
        type=int,
        default=None,
        help="Compatibility alias used by the main event-grid helper; defaults to --window_hours.",
    )
    p.add_argument("--event_env_max_events", type=int, default=3)
    p.add_argument(
        "--environment_grid_only",
        action="store_true",
        help=(
            "Draw only one event-environment grid from existing per-sample "
            "results; skip every other event figure."
        ),
    )
    p.add_argument(
        "--environment_grids_only",
        action="store_true",
        help="Draw only all selected 7-hour event-environment grids; skip every other event figure.",
    )
    p.add_argument(
        "--environment_event_rank",
        type=int,
        default=1,
        help="Event rank to draw when --environment_grid_only is used.",
    )
    p.add_argument(
        "--event_env_include_csi",
        action="store_true",
        help="Add the optional hourly Low-vis CSI column to event environment grids.",
    )
    p.add_argument(
        "--event_env_with_pangu",
        action="store_true",
        help="Draw an additional event-grid version with Pangu-driven VisCast predictions.",
    )
    p.add_argument(
        "--event_env_with_source_models",
        action="store_true",
        help="Attach IFS-driven and Tianji-driven VisCast predictions from paired overlap evaluation.",
    )
    p.add_argument(
        "--event_env_overlap_eval",
        default="",
        help="Optional per_sample_paired_eval.csv; defaults to <eval_dir>/overlap_forecast_source/.",
    )
    p.add_argument(
        "--event_env_pangu_eval_root",
        default="",
        help="Q-core fair-evaluation root containing seed_<seed>/per_sample_<tag>.csv.",
    )
    p.add_argument("--event_env_pangu_seeds", default="42:2025:20260702")
    p.add_argument(
        "--event_env_pangu_tag",
        default="pangu2025_q_core_t925_no_rh2m",
    )
    p.add_argument("--event_env_source", choices=["grid", "none"], default="grid")
    p.add_argument("--shp_path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    p.add_argument(
        "--event_env_tianji_template",
        default="/tj01/sd3op/userpp/pp_data/{init_yyyymmddhh}/stage26Q/multi_model_sources/{init_yyyymmddhh}/{variable}.nc",
    )
    p.add_argument("--event_env_rh2m_var", default="rh2m")
    p.add_argument("--event_env_rh2m_vmin", type=float, default=40.0)
    p.add_argument("--event_env_rh2m_vmax", type=float, default=100.0)
    p.add_argument("--event_env_pm10_dir", default="pm10_data")
    p.add_argument("--event_env_pm10_var", default="pm10")
    p.add_argument("--event_env_pm10_vmin", type=float, default=0.0)
    p.add_argument("--event_env_pm10_vmax", type=float, default=240.0)
    return p.parse_args()


def parse_replacements(items: Sequence[str]) -> Dict[int, pd.Timestamp]:
    out: Dict[int, pd.Timestamp] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"--replace_event must be rank=time, got: {item}")
        left, right = str(item).split("=", 1)
        rank = int(left.strip())
        ts = pd.Timestamp(right.strip()).floor("h")
        if pd.isna(ts):
            raise ValueError(f"Cannot parse replacement time: {right}")
        out[rank] = ts
    return out


def read_eval_table(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "per_sample_eval.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per_sample_eval.csv: {path}")
    df = pd.read_csv(path)
    if "time" not in df.columns:
        if "time_utc" in df.columns:
            df["time"] = df["time_utc"]
        else:
            raise KeyError(f"{path} must contain time or time_utc")
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.floor("h")
    df = df[df["time"].notna()].copy()
    for col in ("y_true", "vis_raw_m", "pmst_pred", "ifs_diagnostic_pred", "ifs_diagnostic_vis_m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ifs_diagnostic_valid" in df.columns:
        df["ifs_diagnostic_valid"] = df["ifs_diagnostic_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["ifs_diagnostic_valid"] = False
    return df


def load_event_summary(eval_dir: Path, explicit: str) -> pd.DataFrame:
    path = Path(explicit) if explicit else eval_dir / "event_case_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event summary: {path}")
    df = pd.read_csv(path)
    for col in ("peak_time", "actual_peak_time", "start_time", "end_time", "window_start", "window_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def region_count_for_hour(df: pd.DataFrame) -> int:
    # Keep this script independent of the full scenario helper; the value is
    # metadata only for replacement rows and does not drive plotting.
    return 0


def span(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) == 0:
        return 0.0
    return float(vals.max() - vals.min())


def update_event_row(row: pd.Series, eval_df: pd.DataFrame, peak_time: pd.Timestamp, window_hours: int) -> pd.Series:
    out = row.copy()
    sub = eval_df[eval_df["time"] == peak_time]
    fog = sub[sub["y_true"] == 0] if "y_true" in sub else sub.iloc[0:0]
    low_vis = sub[sub["y_true"] <= 1] if "y_true" in sub else sub.iloc[0:0]
    out["peak_time"] = peak_time
    out["actual_peak_time"] = peak_time
    out["peak_fog_count"] = int(len(fog))
    out["peak_ultralow_count"] = int(len(fog))
    out["actual_peak_fog_count"] = int(len(fog))
    out["actual_peak_ultralow_count"] = int(len(fog))
    out["peak_region_count"] = region_count_for_hour(fog)
    out["actual_peak_region_count"] = region_count_for_hour(fog)
    out["peak_lon_span"] = span(fog["lon"]) if "lon" in fog else 0.0
    out["actual_peak_lon_span"] = out["peak_lon_span"]
    out["peak_lat_span"] = span(fog["lat"]) if "lat" in fog else 0.0
    out["actual_peak_lat_span"] = out["peak_lat_span"]
    out["start_time"] = peak_time - pd.Timedelta(hours=window_hours)
    out["end_time"] = peak_time + pd.Timedelta(hours=window_hours)
    out["duration_h"] = 2 * int(window_hours) + 1
    out["window_start"] = out["start_time"]
    out["window_end"] = out["end_time"]
    needed = [peak_time + pd.Timedelta(hours=h) for h in range(-window_hours, window_hours + 1)]
    available = set(pd.DatetimeIndex(eval_df["time"]).asi8.tolist())
    flags = [pd.Timestamp(t).value in available for t in needed]
    out["window_complete"] = bool(all(flags))
    out["window_available_hours"] = int(sum(flags))
    out["window_required_hours"] = int(len(flags))
    out["total_fog_station_hours"] = int(
        sum(int(((eval_df["time"] == t) & (eval_df["y_true"] == 0)).sum()) for t in needed)
    )
    out["total_ultralow_station_hours"] = int(out["total_fog_station_hours"])
    out["event_score"] = float(out["total_fog_station_hours"]) + 2.0 * int(len(fog)) + int(len(low_vis))
    out["selection_tier"] = "manual_replacement"
    out["selection_tier_rank"] = -1
    return out


def apply_replacements(event_df: pd.DataFrame, eval_df: pd.DataFrame, replacements: Dict[int, pd.Timestamp], window_hours: int) -> pd.DataFrame:
    out = event_df.copy()
    if "event_rank" not in out.columns:
        out.insert(0, "event_rank", np.arange(1, len(out) + 1))
    for rank, ts in replacements.items():
        mask = out["event_rank"].astype(int) == int(rank)
        if not mask.any():
            raise ValueError(f"Cannot replace event rank {rank}; event summary only has ranks {out['event_rank'].tolist()}")
        idx = out.index[mask][0]
        out.loc[idx] = update_event_row(out.loc[idx], eval_df, ts, window_hours)
    out = sort_events_chronologically(out)
    return out


def sort_events_chronologically(event_df: pd.DataFrame) -> pd.DataFrame:
    out = event_df.copy()
    out["__peak_time_sort"] = pd.to_datetime(out["peak_time"], errors="coerce")
    out = out.sort_values(["__peak_time_sort", "event_rank"]).drop(columns=["__peak_time_sort"]).reset_index(drop=True)
    if "event_rank" in out.columns:
        out = out.drop(columns=["event_rank"])
    out.insert(0, "event_rank", np.arange(1, len(out) + 1))
    return out


def arrays_from_eval(eval_df: pd.DataFrame):
    meta = eval_df.copy()
    y_cls = eval_df["y_true"].to_numpy(dtype=np.int64)
    y_raw = eval_df["vis_raw_m"].to_numpy(dtype=np.float64)
    pmst_pred = eval_df["pmst_pred"].to_numpy(dtype=np.int64)
    ifs_pred = eval_df["ifs_diagnostic_pred"].fillna(-1).to_numpy(dtype=np.int64)
    ifs_valid = eval_df["ifs_diagnostic_valid"].to_numpy(dtype=bool)
    return meta, y_cls, y_raw, pmst_pred, ifs_pred, ifs_valid


def parse_seed_list(value: str) -> List[int]:
    values = [item.strip() for item in str(value).replace(",", ":").split(":") if item.strip()]
    if not values:
        raise ValueError("At least one Pangu seed is required")
    return [int(item) for item in values]


def normalize_station_key(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )


def attach_pangu_seed_mean_predictions(
    eval_df: pd.DataFrame,
    eval_root: Path,
    seeds: Sequence[int],
    source_tag: str,
) -> tuple[pd.DataFrame, List[str], Dict[str, object]]:
    """Attach argmax predictions from mean Pangu-driven class probabilities."""

    seed_tables: List[pd.DataFrame] = []
    source_paths: List[str] = []
    required = {"time", "station_id", "p_fog", "p_mist", "p_clear"}
    for seed in seeds:
        path = eval_root / f"seed_{int(seed)}" / f"per_sample_{source_tag}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Pangu event-prediction source: {path}")
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{path}: missing required columns {missing}")
        time = pd.to_datetime(frame["time"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("h")
        if time.isna().any():
            raise ValueError(f"{path}: invalid event timestamps")
        table = pd.DataFrame(
            {
                "time": time,
                "station_key": normalize_station_key(frame["station_id"]),
                f"p_fog_{seed}": pd.to_numeric(frame["p_fog"], errors="coerce"),
                f"p_mist_{seed}": pd.to_numeric(frame["p_mist"], errors="coerce"),
                f"p_clear_{seed}": pd.to_numeric(frame["p_clear"], errors="coerce"),
            }
        )
        if table[["time", "station_key"]].duplicated().any():
            raise ValueError(f"{path}: duplicate (time, station_id) rows")
        probability_columns = [f"p_fog_{seed}", f"p_mist_{seed}", f"p_clear_{seed}"]
        if not np.isfinite(table[probability_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: non-finite class probabilities")
        seed_tables.append(table)
        source_paths.append(str(path))

    combined = seed_tables[0]
    for table in seed_tables[1:]:
        combined = combined.merge(
            table,
            on=["time", "station_key"],
            how="inner",
            validate="one_to_one",
        )
    if combined.empty:
        raise ValueError("Pangu seed outputs have no common (time, station_id) rows")

    mean_probabilities = np.column_stack(
        [
            combined[[f"p_{label}_{seed}" for seed in seeds]].mean(axis=1).to_numpy(dtype=float)
            for label in ("fog", "mist", "clear")
        ]
    )
    combined["pangu_pred"] = np.argmax(mean_probabilities, axis=1).astype(np.int64)
    combined["pangu_valid"] = True

    out = eval_df.copy()
    out["station_key"] = normalize_station_key(out["station_id"])
    out = out.merge(
        combined[["time", "station_key", "pangu_pred", "pangu_valid"]],
        on=["time", "station_key"],
        how="left",
        validate="many_to_one",
    )
    out["pangu_valid"] = out["pangu_valid"].fillna(False).astype(bool)
    matched = int(out["pangu_valid"].sum())
    if matched == 0:
        raise ValueError("Pangu predictions do not overlap the event-evaluation sample table")
    metadata = {
        "eval_root": str(eval_root),
        "source_tag": str(source_tag),
        "seeds": [int(seed) for seed in seeds],
        "combination": "argmax_of_equal_weight_mean_class_probabilities",
        "matched_rows": matched,
        "total_event_eval_rows": int(len(out)),
    }
    return out, source_paths, metadata


def attach_overlap_source_predictions(
    eval_df: pd.DataFrame,
    source_path: Path,
) -> tuple[pd.DataFrame, List[str], Dict[str, object]]:
    """Attach paired IFS-driven and Tianji-driven VisCast predictions."""

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing paired source evaluation: {source_path}")
    frame = pd.read_csv(source_path)
    required = {"time", "station_id", "tianji_pred", "ifs_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{source_path}: missing required columns {missing}")
    time = pd.to_datetime(frame["time"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("h")
    if time.isna().any():
        raise ValueError(f"{source_path}: invalid event timestamps")
    table = pd.DataFrame(
        {
            "time": time,
            "station_key": normalize_station_key(frame["station_id"]),
            "tianji_driven_pred": pd.to_numeric(frame["tianji_pred"], errors="coerce"),
            "ifs_driven_pred": pd.to_numeric(frame["ifs_pred"], errors="coerce"),
        }
    )
    if table[["time", "station_key"]].duplicated().any():
        raise ValueError(f"{source_path}: duplicate (time, station_id) rows")
    out = eval_df.copy()
    out["station_key"] = normalize_station_key(out["station_id"])
    out = out.merge(
        table,
        on=["time", "station_key"],
        how="left",
        validate="many_to_one",
    )
    out["tianji_driven_valid"] = np.isfinite(out["tianji_driven_pred"].to_numpy(dtype=float))
    out["ifs_driven_valid"] = np.isfinite(out["ifs_driven_pred"].to_numpy(dtype=float))
    matched = int(np.count_nonzero(out["tianji_driven_valid"] & out["ifs_driven_valid"]))
    if matched == 0:
        raise ValueError("Paired IFS/Tianji predictions do not overlap the event-evaluation sample table")
    metadata = {
        "source_path": str(source_path),
        "matched_rows": matched,
        "total_event_eval_rows": int(len(out)),
        "prediction_columns": {"ifs": "ifs_pred", "tianji": "tianji_pred"},
    }
    return out, [str(source_path)], metadata


def main() -> None:
    args = parse_args()
    if args.event_window_hours is None:
        args.event_window_hours = int(args.window_hours)
    else:
        args.window_hours = int(args.event_window_hours)
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else eval_dir / "event_rerun_custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base).expanduser()

    eval_df = read_eval_table(eval_dir)
    event_df = load_event_summary(eval_dir, args.event_summary)
    original_event_ranks = (
        pd.to_numeric(event_df["event_rank"], errors="coerce")
        if "event_rank" in event_df.columns
        else pd.Series(np.arange(1, len(event_df) + 1), index=event_df.index)
    )
    replacements = parse_replacements(args.replace_event)
    if replacements:
        event_df = apply_replacements(event_df, eval_df, replacements, args.window_hours)
    else:
        if args.environment_grid_only:
            event_df = event_df.copy()
            event_df["event_rank"] = original_event_ranks.astype(int)
        else:
            event_df = sort_events_chronologically(event_df)
    event_df.to_csv(out_dir / "event_case_summary.csv", index=False)

    source_model_metadata: Dict[str, object] = {}
    source_model_sources: List[str] = []
    if args.event_env_with_source_models:
        source_eval_path = (
            Path(args.event_env_overlap_eval).expanduser().resolve()
            if str(args.event_env_overlap_eval).strip()
            else eval_dir / "overlap_forecast_source" / "per_sample_paired_eval.csv"
        )
        eval_df, source_model_sources, source_model_metadata = attach_overlap_source_predictions(
            eval_df,
            source_eval_path,
        )

    pangu_metadata: Dict[str, object] = {}
    pangu_sources: List[str] = []
    if args.event_env_with_pangu:
        if not str(args.event_env_pangu_eval_root).strip():
            raise ValueError("--event_env_with_pangu requires --event_env_pangu_eval_root")
        eval_df, pangu_sources, pangu_metadata = attach_pangu_seed_mean_predictions(
            eval_df,
            Path(args.event_env_pangu_eval_root).expanduser().resolve(),
            parse_seed_list(args.event_env_pangu_seeds),
            args.event_env_pangu_tag,
        )

    if args.environment_grid_only:
        selected = event_df[
            pd.to_numeric(event_df["event_rank"], errors="coerce")
            == int(args.environment_event_rank)
        ]
        if len(selected) != 1:
            raise ValueError(
                "--environment_event_rank must resolve exactly one event; "
                f"requested={args.environment_event_rank}, "
                f"available={event_df['event_rank'].tolist()}"
            )
        shp_gdf = journal.read_shapefile(args.shp_path) if args.shp_path else None
        manifest = journal.Manifest(out_dir)
        sources = [
            str(eval_dir / "per_sample_eval.csv"),
            str(out_dir / "event_case_summary.csv"),
        ] + source_model_sources + pangu_sources
        journal.plot_event_environment_grid(
            args,
            base,
            eval_df,
            selected.iloc[0],
            out_dir,
            manifest,
            sources,
            shp_gdf=shp_gdf,
        )
        manifest.write()
        run_config = {
            "eval_dir": str(eval_dir),
            "out_dir": str(out_dir),
            "base": str(base),
            "environment_grid_only": True,
            "environment_event_rank": int(args.environment_event_rank),
            "window_hours": int(args.window_hours),
            "event_env_include_csi": bool(args.event_env_include_csi),
            "event_env_with_source_models": bool(args.event_env_with_source_models),
            "source_model_prediction_source": source_model_metadata,
            "event_env_with_pangu": bool(args.event_env_with_pangu),
            "pangu_prediction_source": pangu_metadata,
        }
        (out_dir / "event_rerun_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[event] single environment grid complete: "
            f"rank={args.environment_event_rank}, out={out_dir}",
            flush=True,
        )
        return

    if args.environment_grids_only:
        shp_gdf = journal.read_shapefile(args.shp_path) if args.shp_path else None
        manifest = journal.Manifest(out_dir)
        sources = [
            str(eval_dir / "per_sample_eval.csv"),
            str(out_dir / "event_case_summary.csv"),
        ] + source_model_sources + pangu_sources
        journal.plot_event_environment_grids(
            args,
            base,
            eval_df,
            event_df,
            out_dir,
            manifest,
            sources,
            shp_gdf=shp_gdf,
        )
        manifest.write()
        run_config = {
            "eval_dir": str(eval_dir),
            "out_dir": str(out_dir),
            "base": str(base),
            "environment_grids_only": True,
            "event_env_max_events": int(args.event_env_max_events),
            "window_hours": int(args.window_hours),
            "event_env_include_csi": bool(args.event_env_include_csi),
            "event_env_with_source_models": bool(args.event_env_with_source_models),
            "source_model_prediction_source": source_model_metadata,
            "event_env_with_pangu": bool(args.event_env_with_pangu),
            "pangu_prediction_source": pangu_metadata,
        }
        (out_dir / "event_rerun_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[event] 7-hour environment grids complete: out={out_dir}", flush=True)
        return

    meta, y_cls, y_raw, pmst_pred, ifs_pred, ifs_valid = arrays_from_eval(eval_df)
    shp_gdf = journal.read_shapefile(args.shp_path) if args.shp_path else None
    manifest = journal.Manifest(out_dir)
    sources = [
        str(eval_dir / "per_sample_eval.csv"),
        str(out_dir / "event_case_summary.csv"),
    ] + source_model_sources + pangu_sources

    summary_rows: List[dict] = []
    event_df_top = sort_events_chronologically(event_df).head(3).copy()
    for _, event_row in event_df_top.iterrows():
        rank = int(event_row["event_rank"])
        hourly = compute_event_hourly_metrics(
            meta,
            y_cls,
            pmst_pred,
            ifs_pred,
            ifs_valid,
            center_time=event_row["peak_time"],
            window_hours=args.window_hours,
        )
        hourly_path = out_dir / f"fig9_event_{rank}_hourly_metrics.csv"
        hourly.to_csv(hourly_path, index=False, float_format="%.4f")
        summary_rows.append(summarize_event_metrics(hourly, event_row))

        plot_widespread_event_panels(
            meta,
            y_raw,
            pmst_pred,
            ifs_pred,
            ifs_valid,
            event_row,
            str(out_dir / f"fig9_event_{rank}_spatial.png"),
            shp_gdf=shp_gdf,
            window_hours=args.window_hours,
        )
        plot_event_metric_comparison(hourly, event_row, str(out_dir / f"fig9_event_{rank}_metrics.png"))

    event_summary_df = pd.DataFrame(summary_rows)
    if not event_summary_df.empty:
        event_summary_df.to_csv(out_dir / "fig9_event_summary_metrics.csv", index=False, float_format="%.4f")
        plot_event_summary_comparison(event_summary_df, str(out_dir / "fig9_event_summary.png"))

    three_footprint_path = out_dir / "fig_three_events_footprint_row.png"
    plot_three_events_footprint_row(
        meta,
        y_raw,
        pmst_pred,
        event_df,
        str(three_footprint_path),
        shp_gdf=shp_gdf,
        window_hours=args.window_hours,
    )
    manifest.add(three_footprint_path.name, sources, notes="Manual event-only rerun footprint row.", n=int(len(eval_df)))

    three_peak_path = out_dir / "fig_three_events_peak_row.png"
    plot_three_events_peak_row(
        meta,
        y_raw,
        pmst_pred,
        event_df,
        str(three_peak_path),
        shp_gdf=shp_gdf,
    )
    manifest.add(three_peak_path.name, sources, notes="Manual event-only rerun peak row.", n=int(len(eval_df)))

    journal.plot_event_peak_grid(eval_df, event_df, out_dir, manifest, sources, shp_gdf=shp_gdf)
    hourly_paths = [out_dir / f"fig9_event_{int(row.event_rank)}_hourly_metrics.csv" for row in event_df_top.itertuples()]
    journal.plot_event_footprint(hourly_paths, out_dir, manifest, [str(p) for p in hourly_paths])

    journal.plot_event_environment_grids(
        args,
        base,
        eval_df,
        event_df,
        out_dir,
        manifest,
        sources,
        shp_gdf=shp_gdf,
    )
    manifest.write()

    run_config = {
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "base": str(base),
        "replace_event": {str(k): str(v) for k, v in replacements.items()},
        "window_hours": int(args.window_hours),
        "event_env_source": str(args.event_env_source),
        "event_env_max_events": int(args.event_env_max_events),
        "event_env_include_csi": bool(args.event_env_include_csi),
        "event_env_with_source_models": bool(args.event_env_with_source_models),
        "source_model_prediction_source": source_model_metadata,
        "event_env_with_pangu": bool(args.event_env_with_pangu),
        "pangu_prediction_source": pangu_metadata,
    }
    (out_dir / "event_rerun_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[event] wrote {out_dir / 'event_case_summary.csv'}", flush=True)
    print(f"[event] figures under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
