#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Station-level diurnal plots for checking Tianji/visibility time alignment.

This diagnostic intentionally does not apply any time correction to the input
NetCDF first. It compares two interpretations of the raw naive timestamp:

1. raw_is_utc:   local clock = raw_time + UTC offset
2. raw_is_local: local clock = raw_time

The physically plausible interpretation should place shortwave radiation near
local noon and 2 m temperature near local afternoon. Visibility/low-visibility
counts can then be read on the same local-clock axis.
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DEFAULT_NC = "/public/home/putianshu/vis_mlp/tianji_auto_station/merged_final_all_vars.nc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot station visibility and meteorological diurnal cycles under "
            "raw-time-as-UTC and raw-time-as-local interpretations."
        )
    )
    p.add_argument("--nc", default=DEFAULT_NC, help="Input merged Tianji/station NetCDF.")
    p.add_argument("--out_dir", default="time_alignment_check", help="Output directory.")
    p.add_argument("--station_id", default="", help="Station id to plot. If omitted, choose by low-vis count.")
    p.add_argument("--lat", type=float, default=np.nan, help="Choose nearest station to this latitude.")
    p.add_argument("--lon", type=float, default=np.nan, help="Choose nearest station to this longitude.")
    p.add_argument("--start", default="", help="Raw timestamp slice start, e.g. 2025-01-01.")
    p.add_argument("--end", default="", help="Raw timestamp slice end, e.g. 2025-01-15.")
    p.add_argument("--default_days", type=int, default=14, help="Days used from file start if --start/--end omitted.")
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


def coerce_station_value(value: str, coord: xr.DataArray):
    vals = coord.values
    if np.issubdtype(vals.dtype, np.integer):
        return int(value)
    if np.issubdtype(vals.dtype, np.floating):
        return float(value)
    return str(value)


def select_time_window(ds: xr.Dataset, args: argparse.Namespace) -> xr.Dataset:
    raw_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    if args.start or args.end:
        start = pd.Timestamp(args.start) if args.start else raw_times.min()
        end = pd.Timestamp(args.end) if args.end else raw_times.max()
    else:
        start = raw_times.min()
        end = start + pd.Timedelta(days=int(args.default_days))
    return ds.sel(time=slice(start, end))


def choose_station(ds: xr.Dataset, args: argparse.Namespace, station_dim: str, vis_var: str):
    coord = ds[station_dim]
    if args.station_id:
        return coerce_station_value(args.station_id, coord), "requested station_id"

    lat_name = first_existing(ds, ("lat", "latitude", "station_lat"), required=False)
    lon_name = first_existing(ds, ("lon", "longitude", "station_lon"), required=False)
    if np.isfinite(args.lat) and np.isfinite(args.lon) and lat_name and lon_name:
        lats = np.asarray(ds[lat_name].values, dtype=float)
        lons = np.asarray(ds[lon_name].values, dtype=float)
        dist2 = (lats - args.lat) ** 2 + (lons - args.lon) ** 2
        i = int(np.nanargmin(dist2))
        return coord.values[i].item(), "nearest lat/lon"

    vis = ds[vis_var]
    vis_m = vis
    finite_max = float(vis_m.max(skipna=True).values)
    if finite_max < 100.0:
        vis_m = vis_m * 1000.0
    low_counts = ((vis_m >= 0.0) & (vis_m < 1000.0)).sum("time")
    i = int(np.nanargmax(low_counts.values))
    return coord.values[i].item(), "max low-vis count in selected window"


def to_1d_values(da: xr.DataArray) -> np.ndarray:
    values = np.asarray(da.values)
    return values.reshape(-1)


def build_station_frame(ds_st: xr.Dataset, args: argparse.Namespace) -> pd.DataFrame:
    vis_var = first_existing(ds_st, ("visibility", "vis", "VIS", "Visibility"))
    t2m_var = first_existing(ds_st, ("T2M", "TMP2m", "t2m", "t2mz", "2t"))
    sw_var = first_existing(ds_st, ("SW_RAD", "DSWRFsfc", "ssrd", "SWDOWN", "swrad"), required=False)
    rh_var = first_existing(ds_st, ("RH2M", "rh2m", "rh", "RH"), required=False)

    raw_time = pd.DatetimeIndex(pd.to_datetime(ds_st["time"].values))
    vis = to_1d_values(ds_st[vis_var].load()).astype(float)
    if np.nanmax(vis) < 100.0:
        vis = vis * 1000.0
    vis = np.where((vis >= 0.0) & (vis <= 30000.0), vis, np.nan)

    t2m = to_1d_values(ds_st[t2m_var].load()).astype(float)
    t2m_c = t2m - 273.15 if np.nanmedian(t2m) > 150.0 else t2m

    df = pd.DataFrame(
        {
            "raw_time": raw_time,
            "local_if_raw_utc": raw_time + pd.Timedelta(hours=int(args.tz_offset)),
            "local_if_raw_local": raw_time,
            "visibility_m": vis,
            "t2m_c": t2m_c,
        }
    )
    if sw_var is not None:
        df["sw_rad"] = to_1d_values(ds_st[sw_var].load()).astype(float)
    if rh_var is not None:
        df["rh2m"] = to_1d_values(ds_st[rh_var].load()).astype(float)
    df["fog"] = (df["visibility_m"] >= 0.0) & (df["visibility_m"] < 500.0)
    df["mist"] = (df["visibility_m"] >= 500.0) & (df["visibility_m"] < 1000.0)
    df["low_vis"] = df["fog"] | df["mist"]
    return df


def diurnal_table(df: pd.DataFrame, local_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hour"] = tmp[local_col].dt.hour
    rows = []
    for hour in range(24):
        sub = tmp[tmp["hour"] == hour]
        n = int(len(sub))
        low = int(sub["low_vis"].sum()) if n else 0
        row = {
            "hour": hour,
            "n": n,
            "visibility_m_median": float(sub["visibility_m"].median()) if n else np.nan,
            "t2m_c_median": float(sub["t2m_c"].median()) if n else np.nan,
            "fog_count": int(sub["fog"].sum()) if n else 0,
            "mist_count": int(sub["mist"].sum()) if n else 0,
            "low_vis_count": low,
            "low_vis_rate": float(low / n) if n else np.nan,
        }
        if "sw_rad" in sub:
            row["sw_rad_median"] = float(sub["sw_rad"].median()) if n else np.nan
        if "rh2m" in sub:
            row["rh2m_median"] = float(sub["rh2m"].median()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def circular_distance(hour: float, target: float) -> float:
    if not np.isfinite(hour):
        return float("inf")
    d = abs(float(hour) - float(target))
    return min(d, 24.0 - d)


def peak_hour(table: pd.DataFrame, column: str) -> Optional[int]:
    if column not in table or table[column].isna().all():
        return None
    return int(table.loc[table[column].idxmax(), "hour"])


def summarize_interpretation(table: pd.DataFrame) -> dict:
    t_peak = peak_hour(table, "t2m_c_median")
    sw_peak = peak_hour(table, "sw_rad_median")
    low_peak = peak_hour(table, "low_vis_rate")
    score = circular_distance(t_peak if t_peak is not None else np.nan, 15.0)
    if sw_peak is not None:
        score += circular_distance(sw_peak, 12.0)
    return {
        "t2m_peak_hour": t_peak,
        "sw_rad_peak_hour": sw_peak,
        "low_vis_rate_peak_hour": low_peak,
        "sanity_score_lower_is_better": score,
    }


def plot_diurnal_compare(tab_utc: pd.DataFrame, tab_local: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
    specs = [
        ("t2m_c_median", "2 m temperature median (deg C)"),
        ("sw_rad_median" if "sw_rad_median" in tab_utc else "visibility_m_median", "Shortwave median (W m-2)" if "sw_rad_median" in tab_utc else "Visibility median (m)"),
        ("low_vis_rate", "Observed low-vis rate"),
    ]
    for ax, (col, ylabel) in zip(axes, specs):
        ax.axvspan(0, 6, color="#D9E4F5", alpha=0.35, lw=0)
        ax.axvspan(18, 24, color="#D9E4F5", alpha=0.35, lw=0)
        ax.plot(tab_utc["hour"], tab_utc[col], marker="o", lw=2.0, label="raw time is UTC: local=raw+8")
        ax.plot(tab_local["hour"], tab_local[col], marker="s", lw=1.8, label="raw time is BJT/local: local=raw")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Local clock hour")
    axes[-1].set_xticks(range(0, 24, 2))
    axes[0].legend(frameon=False, ncol=1, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col")
    variants = [
        ("local_if_raw_utc", "raw time is UTC: x = raw + 8 h"),
        ("local_if_raw_local", "raw time is BJT/local: x = raw"),
    ]
    for j, (col, label) in enumerate(variants):
        axes[0, j].plot(df[col], df["t2m_c"], color="#D55E00", lw=1.4)
        axes[0, j].set_title(label)
        axes[0, j].set_ylabel("2 m temperature (deg C)")
        axes[0, j].grid(alpha=0.25)

        axes[1, j].plot(df[col], df["visibility_m"], color="#2E5A87", lw=1.2)
        axes[1, j].axhline(1000, color="#E69F00", lw=1.0, ls="--")
        axes[1, j].axhline(500, color="#2E5A87", lw=1.0, ls=":")
        axes[1, j].set_ylabel("Visibility (m)")
        axes[1, j].set_yscale("symlog", linthresh=1000)
        axes[1, j].grid(alpha=0.25)
        axes[1, j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.nc, engine=args.engine)
    if "time" not in ds.coords:
        raise KeyError("Input file has no time coordinate.")
    station_dim = station_dim_name(ds)
    vis_var = first_existing(ds, ("visibility", "vis", "VIS", "Visibility"))
    ds_win = select_time_window(ds, args)
    station, station_reason = choose_station(ds_win, args, station_dim, vis_var)
    ds_st = ds_win.sel({station_dim: station}).load()

    df = build_station_frame(ds_st, args)
    station_safe = str(station).replace("/", "_").replace("\\", "_")
    csv_path = out_dir / f"station_{station_safe}_time_alignment_series.csv"
    df.to_csv(csv_path, index=False)

    tab_utc = diurnal_table(df, "local_if_raw_utc")
    tab_local = diurnal_table(df, "local_if_raw_local")
    tab_utc.to_csv(out_dir / f"station_{station_safe}_diurnal_raw_is_utc.csv", index=False)
    tab_local.to_csv(out_dir / f"station_{station_safe}_diurnal_raw_is_local.csv", index=False)

    summary = {
        "input_nc": str(args.nc),
        "station_id": str(station),
        "station_selection": station_reason,
        "raw_time_start": str(df["raw_time"].min()),
        "raw_time_end": str(df["raw_time"].max()),
        "n_times": int(len(df)),
        "raw_is_utc": summarize_interpretation(tab_utc),
        "raw_is_bjt_or_local": summarize_interpretation(tab_local),
        "interpretation_hint": (
            "The lower sanity score is closer to SW_RAD noon and T2M afternoon peaks. "
            "Use this as a visual diagnostic, not as a substitute for source metadata."
        ),
    }
    with open(out_dir / f"station_{station_safe}_time_alignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    title = f"Station {station_safe} time-alignment diagnostic"
    plot_diurnal_compare(tab_utc, tab_local, out_dir / f"station_{station_safe}_diurnal_compare.png", title)
    plot_timeseries(df, out_dir / f"station_{station_safe}_timeseries_compare.png", title)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[OK] wrote: {out_dir}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
