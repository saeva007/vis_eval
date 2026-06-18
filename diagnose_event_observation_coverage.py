#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose observation coverage and peak-centering quality for Fig. 9 events.

The script reads an evaluation output directory containing ``per_sample_eval.csv``
and, when available, ``event_case_summary.csv`` plus the per-event hourly metric
tables. It reports whether apparent event jumps are likely caused by missing
test/observation rows and recommends alternative event centers with complete,
stable windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Fig. 9 event observation coverage.")
    p.add_argument("--eval_dir", required=True, help="Directory containing per_sample_eval.csv and event_case_summary.csv.")
    p.add_argument("--out_dir", default="", help="Output directory. Defaults to <eval_dir>/event_observation_diagnostics.")
    p.add_argument("--window_hours", type=int, default=3)
    p.add_argument("--coverage_reference_quantile", type=float, default=0.95)
    p.add_argument("--min_coverage_frac", type=float, default=0.80)
    p.add_argument("--min_fog_stations", type=int, default=80)
    p.add_argument("--gap_hours", type=int, default=24)
    p.add_argument("--recommend_top_k", type=int, default=12)
    p.add_argument(
        "--allow_noncenter_peak",
        action="store_true",
        help="Allow recommended centers whose observed fog maximum is not exactly at hour offset 0.",
    )
    p.add_argument("--make_plot", action="store_true", help="Also write a compact coverage diagnostic PNG if matplotlib is available.")
    return p.parse_args()


def read_per_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing per-sample table: {path}")
    df = pd.read_csv(path)
    time_col = "time_utc" if "time_utc" in df.columns else "time"
    if time_col not in df.columns:
        raise KeyError(f"{path} must contain either time_utc or time")
    df["event_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df["event_time"].notna()].copy()
    for col in ("y_true", "pmst_pred", "ifs_diagnostic_pred"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ifs_diagnostic_valid" in df.columns:
        df["ifs_diagnostic_valid"] = df["ifs_diagnostic_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["ifs_diagnostic_valid"] = False
    return df


def _span(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) == 0:
        return 0.0
    return float(vals.max() - vals.min())


def hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["obs_fog"] = work["y_true"] == 0
    work["obs_low_vis"] = work["y_true"] <= 1
    work["pmst_fog"] = work.get("pmst_pred", pd.Series(index=work.index, dtype=float)) == 0
    work["pmst_low_vis"] = work.get("pmst_pred", pd.Series(index=work.index, dtype=float)) <= 1
    work["ifs_fog"] = (work.get("ifs_diagnostic_pred", pd.Series(index=work.index, dtype=float)) == 0) & work["ifs_diagnostic_valid"]
    work["ifs_low_vis"] = (work.get("ifs_diagnostic_pred", pd.Series(index=work.index, dtype=float)) <= 1) & work["ifs_diagnostic_valid"]

    grouped = (
        work.groupby("event_time")
        .agg(
            n_total=("station_id", "size"),
            n_unique_stations=("station_id", "nunique"),
            n_matched_ifs=("ifs_diagnostic_valid", "sum"),
            obs_fog_count=("obs_fog", "sum"),
            obs_low_vis_count=("obs_low_vis", "sum"),
            pmst_fog_count=("pmst_fog", "sum"),
            pmst_low_vis_count=("pmst_low_vis", "sum"),
            ifs_fog_count=("ifs_fog", "sum"),
            ifs_low_vis_count=("ifs_low_vis", "sum"),
        )
        .reset_index()
        .rename(columns={"event_time": "time"})
        .sort_values("time")
    )

    fog = work[work["obs_fog"]].copy()
    if not fog.empty and {"lat", "lon"}.issubset(fog.columns):
        spans = (
            fog.groupby("event_time")
            .agg(
                fog_lon_span=("lon", _span),
                fog_lat_span=("lat", _span),
            )
            .reset_index()
            .rename(columns={"event_time": "time"})
        )
        grouped = grouped.merge(spans, on="time", how="left")
    else:
        grouped["fog_lon_span"] = 0.0
        grouped["fog_lat_span"] = 0.0
    grouped[["fog_lon_span", "fog_lat_span"]] = grouped[["fog_lon_span", "fog_lat_span"]].fillna(0.0)
    grouped["obs_fog_rate_per_1000"] = np.where(grouped["n_total"] > 0, grouped["obs_fog_count"] / grouped["n_total"] * 1000.0, np.nan)
    grouped["obs_low_vis_rate_per_1000"] = np.where(grouped["n_total"] > 0, grouped["obs_low_vis_count"] / grouped["n_total"] * 1000.0, np.nan)
    return grouped


def load_events(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "event_case_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("peak_time", "actual_peak_time", "start_time", "end_time", "window_start", "window_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def lookup_hour(hourly: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, object]]:
    return {pd.Timestamp(row["time"]): row.to_dict() for _, row in hourly.iterrows()}


def build_window_rows(
    event_row: pd.Series,
    hourly_by_time: Dict[pd.Timestamp, Dict[str, object]],
    reference_n_total: float,
    window_hours: int,
    min_coverage_frac: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rank = int(event_row.get("event_rank", 0)) if pd.notna(event_row.get("event_rank", np.nan)) else 0
    peak = pd.Timestamp(event_row["peak_time"])
    rows: List[Dict[str, object]] = []
    for offset in range(-window_hours, window_hours + 1):
        t = peak + pd.Timedelta(hours=offset)
        source = hourly_by_time.get(t, {})
        n_total = int(source.get("n_total", 0) or 0)
        n_ref = float(reference_n_total) if reference_n_total > 0 else np.nan
        row = {
            "event_rank": rank,
            "selected_peak_time": peak,
            "actual_peak_time": event_row.get("actual_peak_time", pd.NaT),
            "time": t,
            "hour_offset": offset,
            "has_test_samples": bool(n_total > 0),
            "coverage_reference_n_total": n_ref,
            "coverage_frac_global": n_total / n_ref if n_ref and np.isfinite(n_ref) else np.nan,
        }
        for col in (
            "n_total",
            "n_unique_stations",
            "n_matched_ifs",
            "obs_fog_count",
            "obs_low_vis_count",
            "pmst_fog_count",
            "pmst_low_vis_count",
            "ifs_fog_count",
            "ifs_low_vis_count",
            "obs_fog_rate_per_1000",
            "obs_low_vis_rate_per_1000",
        ):
            row[col] = source.get(col, np.nan)
        rows.append(row)
    win = pd.DataFrame(rows)
    local_ref = float(pd.to_numeric(win["n_total"], errors="coerce").max() or 0.0)
    win["coverage_local_max_n_total"] = local_ref
    win["coverage_frac_local_max"] = np.where(local_ref > 0, win["n_total"] / local_ref, np.nan)
    win["coverage_flag"] = (win["coverage_frac_global"] < min_coverage_frac) | (win["coverage_frac_local_max"] < min_coverage_frac)

    count_idx = int(pd.to_numeric(win["obs_fog_count"], errors="coerce").fillna(-1).idxmax())
    rate_idx = int(pd.to_numeric(win["obs_fog_rate_per_1000"], errors="coerce").fillna(-1).idxmax())
    count_peak = win.loc[count_idx]
    rate_peak = win.loc[rate_idx]
    actual_peak = event_row.get("actual_peak_time", pd.NaT)
    actual_peak = pd.Timestamp(actual_peak) if pd.notna(actual_peak) else pd.NaT
    actual_offsets = []
    if pd.notna(actual_peak):
        actual_offsets = [offset for offset in range(-window_hours, window_hours + 1) if peak + pd.Timedelta(hours=offset) == actual_peak]
    actual_offset = actual_offsets[0] if actual_offsets else np.nan
    actual_window_complete, actual_window_min_cov = window_complete_and_min_coverage(
        actual_peak,
        hourly_by_time,
        reference_n_total,
        window_hours,
    ) if pd.notna(actual_peak) else (False, np.nan)

    bad_hours = win[win["coverage_flag"]]
    notes: List[str] = []
    if not bad_hours.empty:
        worst = bad_hours.sort_values("coverage_frac_global").iloc[0]
        notes.append(
            f"coverage dip at {pd.Timestamp(worst['time']):%Y-%m-%d %H:00} "
            f"(offset {int(worst['hour_offset'])}, n={int(worst['n_total'])}, "
            f"global coverage={float(worst['coverage_frac_global']):.2f})"
        )
    if int(count_peak["hour_offset"]) != 0:
        notes.append(
            f"selected center is not local count peak; max obs fog is offset {int(count_peak['hour_offset'])} "
            f"at {pd.Timestamp(count_peak['time']):%Y-%m-%d %H:00}"
        )
    if int(count_peak["hour_offset"]) == window_hours:
        notes.append("observed fog maximum sits on the right edge of the plotted window")
    if pd.notna(actual_peak) and actual_peak != peak:
        if actual_window_complete:
            notes.append("actual event peak has a complete centered window; consider recentering")
        else:
            notes.append("actual event peak cannot be centered with the current complete-window rule")
    if not notes:
        notes.append("coverage and centering look acceptable")

    summary = {
        "event_rank": rank,
        "selected_peak_time": peak,
        "actual_peak_time": actual_peak,
        "actual_peak_hour_offset_in_selected_window": actual_offset,
        "selected_obs_fog_count": float(win.loc[win["hour_offset"] == 0, "obs_fog_count"].iloc[0]),
        "window_count_peak_time": pd.Timestamp(count_peak["time"]),
        "window_count_peak_hour_offset": int(count_peak["hour_offset"]),
        "window_count_peak_obs_fog_count": float(count_peak["obs_fog_count"]),
        "window_rate_peak_time": pd.Timestamp(rate_peak["time"]),
        "window_rate_peak_hour_offset": int(rate_peak["hour_offset"]),
        "window_rate_peak_obs_fog_per_1000": float(rate_peak["obs_fog_rate_per_1000"]),
        "min_coverage_frac_global": float(pd.to_numeric(win["coverage_frac_global"], errors="coerce").min()),
        "min_coverage_frac_local_max": float(pd.to_numeric(win["coverage_frac_local_max"], errors="coerce").min()),
        "coverage_dip_hours": int(win["coverage_flag"].sum()),
        "selected_window_complete": bool(win["has_test_samples"].all()),
        "actual_peak_centered_window_complete": bool(actual_window_complete),
        "actual_peak_centered_window_min_coverage_frac": float(actual_window_min_cov) if np.isfinite(actual_window_min_cov) else np.nan,
        "notes": "; ".join(notes),
    }
    return win, summary


def window_complete_and_min_coverage(
    center: pd.Timestamp,
    hourly_by_time: Dict[pd.Timestamp, Dict[str, object]],
    reference_n_total: float,
    window_hours: int,
) -> Tuple[bool, float]:
    if pd.isna(center):
        return False, np.nan
    covs = []
    complete = True
    for offset in range(-window_hours, window_hours + 1):
        t = pd.Timestamp(center) + pd.Timedelta(hours=offset)
        row = hourly_by_time.get(t)
        if not row:
            complete = False
            covs.append(0.0)
            continue
        n_total = float(row.get("n_total", 0) or 0)
        covs.append(n_total / reference_n_total if reference_n_total > 0 else np.nan)
        if n_total <= 0:
            complete = False
    return complete, float(np.nanmin(covs)) if covs else np.nan


def build_candidate_table(
    hourly: pd.DataFrame,
    reference_n_total: float,
    window_hours: int,
    min_coverage_frac: float,
    min_fog_stations: int,
    allow_noncenter_peak: bool,
) -> pd.DataFrame:
    by_time = lookup_hour(hourly)
    rows: List[Dict[str, object]] = []
    for _, center_row in hourly.iterrows():
        center = pd.Timestamp(center_row["time"])
        complete, min_cov = window_complete_and_min_coverage(center, by_time, reference_n_total, window_hours)
        if not complete or min_cov < min_coverage_frac:
            continue
        window = []
        for offset in range(-window_hours, window_hours + 1):
            window.append(by_time[center + pd.Timedelta(hours=offset)])
        win_df = pd.DataFrame(window)
        count_peak_idx = int(pd.to_numeric(win_df["obs_fog_count"], errors="coerce").fillna(-1).idxmax())
        count_peak_time = pd.Timestamp(win_df.loc[count_peak_idx, "time"])
        count_peak_offset = int((count_peak_time - center) / pd.Timedelta(hours=1))
        center_fog = int(center_row["obs_fog_count"])
        center_low_vis = int(center_row["obs_low_vis_count"])
        if center_fog < min_fog_stations:
            continue
        if not allow_noncenter_peak and count_peak_offset != 0:
            continue
        score = (
            3.0 * center_fog
            + 1.0 * center_low_vis
            + 2.0 * float(center_row.get("fog_lon_span", 0.0))
            + 2.0 * float(center_row.get("fog_lat_span", 0.0))
            - 50.0 * abs(count_peak_offset)
        )
        rows.append(
            {
                "center_time": center,
                "obs_fog_count": center_fog,
                "obs_low_vis_count": center_low_vis,
                "obs_fog_rate_per_1000": float(center_row["obs_fog_rate_per_1000"]),
                "n_total": int(center_row["n_total"]),
                "coverage_frac_global": float(center_row["n_total"] / reference_n_total),
                "min_window_coverage_frac": float(min_cov),
                "window_count_peak_time": count_peak_time,
                "window_count_peak_hour_offset": count_peak_offset,
                "fog_lon_span": float(center_row.get("fog_lon_span", 0.0)),
                "fog_lat_span": float(center_row.get("fog_lat_span", 0.0)),
                "event_score": float(score),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["event_score", "obs_fog_count"], ascending=[False, False]).reset_index(drop=True)


def pick_recommendations(candidates: pd.DataFrame, gap_hours: int, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    selected = []
    for _, row in candidates.iterrows():
        center = pd.Timestamp(row["center_time"])
        too_close = False
        for existing in selected:
            if abs(center - pd.Timestamp(existing["center_time"])) <= pd.Timedelta(hours=gap_hours):
                too_close = True
                break
        if too_close:
            continue
        selected.append(row.to_dict())
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)


def write_report(
    out_dir: Path,
    eval_dir: Path,
    reference_n_total: float,
    hourly: pd.DataFrame,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Event Observation Coverage Diagnostics",
        "",
        f"- eval_dir: `{eval_dir}`",
        f"- test time range: `{hourly['time'].min()}` to `{hourly['time'].max()}`",
        f"- reference hourly rows: {reference_n_total:.1f} (q={args.coverage_reference_quantile})",
        f"- minimum acceptable coverage fraction: {args.min_coverage_frac:.2f}",
        "",
    ]
    if not summary.empty:
        lines.append("## Selected Events")
        lines.append("")
        cols = [
            "event_rank",
            "selected_peak_time",
            "actual_peak_time",
            "window_count_peak_time",
            "window_count_peak_hour_offset",
            "min_coverage_frac_global",
            "coverage_dip_hours",
            "actual_peak_centered_window_complete",
            "notes",
        ]
        lines.append(markdown_table(summary, cols))
        lines.append("")
    if not recommendations.empty:
        lines.append("## Recommended Replacement Centers")
        lines.append("")
        cols = [
            "center_time",
            "obs_fog_count",
            "obs_low_vis_count",
            "coverage_frac_global",
            "min_window_coverage_frac",
            "window_count_peak_hour_offset",
            "fog_lon_span",
            "fog_lat_span",
            "event_score",
        ]
        lines.append(markdown_table(recommendations.head(args.recommend_top_k), cols))
        lines.append("")
    lines.append("## Rule Of Thumb")
    lines.append("")
    lines.append(
        "Prefer event centers whose +/- window has complete hourly coverage, minimum coverage above the threshold, "
        "and observed fog/low-vis maxima near hour offset 0. If the actual event peak cannot be centered because it "
        "falls at a split boundary, either use an asymmetric/truncated event panel and state that explicitly, or choose "
        "a different recommended center for a clean peak-centered figure."
    )
    (out_dir / "event_observation_coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return ""
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif isinstance(value, pd.Timestamp):
                vals.append(value.strftime("%Y-%m-%d %H:%M"))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def maybe_plot(out_dir: Path, coverage_rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)
        return
    if coverage_rows.empty:
        return
    events = sorted(coverage_rows["event_rank"].dropna().unique())
    fig, axes = plt.subplots(len(events), 1, figsize=(9.0, max(3.0, 2.4 * len(events))), squeeze=False)
    for ax, event_rank in zip(axes[:, 0], events):
        sub = coverage_rows[coverage_rows["event_rank"] == event_rank].copy()
        ax2 = ax.twinx()
        ax.plot(sub["hour_offset"], sub["obs_fog_count"], marker="o", color="#111111", label="Obs fog count")
        ax.plot(sub["hour_offset"], sub["obs_low_vis_count"], marker="o", linestyle="--", color="#444444", label="Obs low-vis count")
        ax2.bar(sub["hour_offset"], sub["coverage_frac_global"], color="#4C78A8", alpha=0.25, label="Coverage")
        ax.axvline(0, color="#777777", linestyle=":")
        ax.set_title(f"Event {int(event_rank)}")
        ax.set_xlabel("Hour relative to selected peak")
        ax.set_ylabel("Station count")
        ax2.set_ylabel("Coverage frac")
        ax2.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "event_observation_coverage_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else eval_dir / "event_observation_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = read_per_sample(eval_dir / "per_sample_eval.csv")
    hourly = hourly_stats(per_sample)
    reference_n_total = float(hourly["n_total"].quantile(args.coverage_reference_quantile))
    if not np.isfinite(reference_n_total) or reference_n_total <= 0:
        reference_n_total = float(hourly["n_total"].max())
    hourly["coverage_reference_n_total"] = reference_n_total
    hourly["coverage_frac_global"] = hourly["n_total"] / reference_n_total
    hourly.to_csv(out_dir / "hourly_observation_coverage_all_test.csv", index=False, float_format="%.6f")

    events = load_events(eval_dir)
    hourly_by_time = lookup_hour(hourly)
    coverage_tables: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    if not events.empty:
        for _, event_row in events.iterrows():
            win, summary = build_window_rows(
                event_row,
                hourly_by_time,
                reference_n_total,
                args.window_hours,
                args.min_coverage_frac,
            )
            coverage_tables.append(win)
            summary_rows.append(summary)
    coverage_df = pd.concat(coverage_tables, axis=0, ignore_index=True) if coverage_tables else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if not coverage_df.empty:
        coverage_df.to_csv(out_dir / "event_observation_coverage_by_hour.csv", index=False, float_format="%.6f")
    if not summary_df.empty:
        summary_df.to_csv(out_dir / "event_observation_coverage_summary.csv", index=False, float_format="%.6f")

    candidates = build_candidate_table(
        hourly,
        reference_n_total,
        args.window_hours,
        args.min_coverage_frac,
        args.min_fog_stations,
        args.allow_noncenter_peak,
    )
    if not candidates.empty:
        candidates.to_csv(out_dir / "candidate_event_centers.csv", index=False, float_format="%.6f")
    recommendations = pick_recommendations(candidates, args.gap_hours, args.recommend_top_k)
    if not recommendations.empty:
        recommendations.to_csv(out_dir / "recommended_event_centers.csv", index=False, float_format="%.6f")

    write_report(out_dir, eval_dir, reference_n_total, hourly, summary_df, recommendations, args)
    config = {
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "window_hours": int(args.window_hours),
        "coverage_reference_quantile": float(args.coverage_reference_quantile),
        "reference_n_total": reference_n_total,
        "min_coverage_frac": float(args.min_coverage_frac),
        "min_fog_stations": int(args.min_fog_stations),
        "gap_hours": int(args.gap_hours),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.make_plot:
        maybe_plot(out_dir, coverage_df, summary_df)

    print(f"[table] {out_dir / 'hourly_observation_coverage_all_test.csv'}", flush=True)
    if not coverage_df.empty:
        print(f"[table] {out_dir / 'event_observation_coverage_by_hour.csv'}", flush=True)
    if not summary_df.empty:
        print(f"[table] {out_dir / 'event_observation_coverage_summary.csv'}", flush=True)
    if not recommendations.empty:
        print(f"[table] {out_dir / 'recommended_event_centers.csv'}", flush=True)
    print(f"[report] {out_dir / 'event_observation_coverage_report.md'}", flush=True)


if __name__ == "__main__":
    main()
