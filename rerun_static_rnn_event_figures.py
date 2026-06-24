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
    replacements = parse_replacements(args.replace_event)
    if replacements:
        event_df = apply_replacements(event_df, eval_df, replacements, args.window_hours)
    else:
        event_df = sort_events_chronologically(event_df)
    event_df.to_csv(out_dir / "event_case_summary.csv", index=False)

    meta, y_cls, y_raw, pmst_pred, ifs_pred, ifs_valid = arrays_from_eval(eval_df)
    shp_gdf = journal.read_shapefile(args.shp_path) if args.shp_path else None
    manifest = journal.Manifest(out_dir)
    sources = [str(eval_dir / "per_sample_eval.csv"), str(out_dir / "event_case_summary.csv")]

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
    }
    (out_dir / "event_rerun_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[event] wrote {out_dir / 'event_case_summary.csv'}", flush=True)
    print(f"[event] figures under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
