#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visibility-only diurnal diagnostic for time-coordinate sanity checks.

The script reads the raw merged NetCDF and compares low-visibility diurnal
statistics under two interpretations of the file's naive timestamp:

* raw_is_utc:   local clock hour = raw hour + 8
* raw_is_local: local clock hour = raw hour

Use this as supporting evidence together with temperature and shortwave
radiation. Visibility alone is less deterministic than solar variables, but
station-aggregated low-visibility frequency should generally be highest during
night to early morning if the local clock is correctly interpreted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_NC = "/public/home/putianshu/vis_mlp/tianji_auto_station/merged_final_all_vars.nc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate visibility diurnal statistics under UTC/local time assumptions.")
    p.add_argument("--nc", default=DEFAULT_NC, help="Input merged Tianji/station NetCDF.")
    p.add_argument("--out_dir", default="visibility_time_alignment_check", help="Output directory.")
    p.add_argument("--start", default="", help="Raw timestamp slice start, e.g. 2025-10-25.")
    p.add_argument("--end", default="", help="Raw timestamp slice end, e.g. 2025-11-03.")
    p.add_argument("--top_stations", type=int, default=300, help="Use top-N stations by low-vis count; <=0 uses all.")
    p.add_argument("--tz_offset", type=int, default=8, help="Local clock offset from UTC.")
    p.add_argument("--engine", default="h5netcdf", help="xarray engine.")
    return p.parse_args()


def first_existing(ds: xr.Dataset, names: Iterable[str], required: bool = True) -> Optional[str]:
    for name in names:
        if name in ds.data_vars or name in ds.coords:
            return name
    if required:
        raise KeyError(f"None of these variables/coords found: {list(names)}")
    return None


def station_dim_name(ds: xr.Dataset) -> str:
    for name in ("station_id", "station", "num_station", "id"):
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError("Cannot find station dimension/coordinate.")


def select_time_window(ds: xr.Dataset, start_value: str, end_value: str) -> xr.Dataset:
    raw_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    start = pd.Timestamp(start_value) if start_value else raw_times.min()
    end = pd.Timestamp(end_value) if end_value else raw_times.max()
    return ds.sel(time=slice(start, end))


def load_visibility_matrix(ds: xr.Dataset, station_dim: str, vis_var: str, top_stations: int):
    da = ds[vis_var].transpose("time", station_dim).load()
    vis = np.asarray(da.values, dtype=np.float32)
    if np.nanmax(vis) < 100.0:
        vis = vis * 1000.0
    valid = np.isfinite(vis) & (vis >= 0.0) & (vis <= 30000.0)
    low = valid & (vis < 1000.0)
    station_ids = np.asarray(ds[station_dim].values)

    if top_stations and top_stations > 0 and top_stations < vis.shape[1]:
        counts = low.sum(axis=0)
        keep = np.argsort(counts)[-int(top_stations):]
        keep = keep[np.argsort(station_ids[keep])]
        vis = vis[:, keep]
        valid = valid[:, keep]
        low = low[:, keep]
        station_ids = station_ids[keep]

    return vis, valid, station_ids


def aggregate_by_hour(vis: np.ndarray, valid: np.ndarray, hours: np.ndarray) -> pd.DataFrame:
    rows = []
    fog = valid & (vis < 500.0)
    mist = valid & (vis >= 500.0) & (vis < 1000.0)
    low = fog | mist
    for hour in range(24):
        tmask = hours == hour
        if not np.any(tmask):
            rows.append(
                {
                    "hour": hour,
                    "n": 0,
                    "fog_count": 0,
                    "mist_count": 0,
                    "low_vis_count": 0,
                    "low_vis_rate": np.nan,
                    "visibility_m_median": np.nan,
                }
            )
            continue
        vmask = valid[tmask]
        n = int(vmask.sum())
        low_count = int(low[tmask].sum())
        values = vis[tmask][vmask]
        rows.append(
            {
                "hour": hour,
                "n": n,
                "fog_count": int(fog[tmask].sum()),
                "mist_count": int(mist[tmask].sum()),
                "low_vis_count": low_count,
                "low_vis_rate": float(low_count / n) if n else np.nan,
                "visibility_m_median": float(np.nanmedian(values)) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def peak_hour(table: pd.DataFrame, column: str) -> Optional[int]:
    if column not in table or table[column].isna().all():
        return None
    return int(table.loc[table[column].idxmax(), "hour"])


def plot_compare(tab_utc: pd.DataFrame, tab_local: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.4), sharex=True)
    for ax in axes:
        ax.axvspan(0, 6, color="#D9E4F5", alpha=0.35, lw=0)
        ax.axvspan(18, 24, color="#D9E4F5", alpha=0.35, lw=0)
        ax.grid(alpha=0.25)

    axes[0].plot(tab_utc["hour"], tab_utc["low_vis_rate"] * 100.0, "o-", lw=2.0, label="raw time is UTC: local=raw+8")
    axes[0].plot(tab_local["hour"], tab_local["low_vis_rate"] * 100.0, "s-", lw=1.8, label="raw time is BJT/local: local=raw")
    axes[0].set_ylabel("Low-vis event rate (%)")
    axes[0].legend(frameon=False)

    width = 0.38
    axes[1].bar(tab_utc["hour"] - width / 2, tab_utc["fog_count"], width=width, color="#2E5A87", alpha=0.85, label="Ultra-low, raw UTC")
    axes[1].bar(tab_utc["hour"] - width / 2, tab_utc["mist_count"], width=width, bottom=tab_utc["fog_count"], color="#E69F00", alpha=0.85, label="Moderate-low, raw UTC")
    axes[1].bar(tab_local["hour"] + width / 2, tab_local["low_vis_count"], width=width, color="#7F7F7F", alpha=0.35, label="Low-vis event, raw local")
    axes[1].set_ylabel("Counts")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)

    axes[2].plot(tab_utc["hour"], tab_utc["visibility_m_median"], "o-", lw=2.0, label="raw UTC")
    axes[2].plot(tab_local["hour"], tab_local["visibility_m_median"], "s-", lw=1.8, label="raw local")
    axes[2].axhline(1000.0, color="#E69F00", ls="--", lw=1.0)
    axes[2].axhline(500.0, color="#2E5A87", ls=":", lw=1.0)
    axes[2].set_ylabel("Median vis (m)")
    axes[2].set_xlabel("Local clock hour")
    axes[2].set_xticks(range(0, 24, 2))

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.nc, engine=args.engine)
    station_dim = station_dim_name(ds)
    vis_var = first_existing(ds, ("visibility", "vis", "VIS", "Visibility"))
    ds_win = select_time_window(ds, args.start, args.end)
    times = pd.DatetimeIndex(pd.to_datetime(ds_win["time"].values))
    raw_hours = times.hour.to_numpy(dtype=int)
    utc_as_local_hours = (raw_hours + int(args.tz_offset)) % 24

    vis, valid, station_ids = load_visibility_matrix(ds_win, station_dim, vis_var, args.top_stations)
    tab_utc = aggregate_by_hour(vis, valid, utc_as_local_hours)
    tab_local = aggregate_by_hour(vis, valid, raw_hours)

    tab_utc.to_csv(out_dir / "visibility_diurnal_raw_is_utc.csv", index=False)
    tab_local.to_csv(out_dir / "visibility_diurnal_raw_is_local.csv", index=False)
    pd.DataFrame({"station_id": station_ids}).to_csv(out_dir / "stations_used.csv", index=False)

    summary = {
        "input_nc": str(args.nc),
        "vis_variable": vis_var,
        "raw_time_start": str(times.min()),
        "raw_time_end": str(times.max()),
        "n_times": int(len(times)),
        "n_stations_used": int(len(station_ids)),
        "top_stations": int(args.top_stations),
        "raw_is_utc": {
            "low_vis_rate_peak_hour": peak_hour(tab_utc, "low_vis_rate"),
            "low_vis_count_peak_hour": peak_hour(tab_utc, "low_vis_count"),
            "min_median_visibility_hour": int(tab_utc.loc[tab_utc["visibility_m_median"].idxmin(), "hour"]),
        },
        "raw_is_bjt_or_local": {
            "low_vis_rate_peak_hour": peak_hour(tab_local, "low_vis_rate"),
            "low_vis_count_peak_hour": peak_hour(tab_local, "low_vis_count"),
            "min_median_visibility_hour": int(tab_local.loc[tab_local["visibility_m_median"].idxmin(), "hour"]),
        },
        "interpretation_hint": (
            "Visibility is a weather outcome, so use it as supporting evidence. "
            "A local-clock early-morning low-vis event peak is more physically plausible "
            "for ultra-low-prone samples than a daytime/afternoon peak."
        ),
    }
    with open(out_dir / "visibility_diurnal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    title = f"Visibility diurnal alignment check ({len(station_ids)} stations)"
    plot_compare(tab_utc, tab_local, out_dir / "visibility_diurnal_compare.png", title)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[OK] wrote: {out_dir}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
