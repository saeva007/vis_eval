"""
 Paper Figure 8/9: Spatial maps and widespread Low-vis event evaluation.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, Normalize

try:
    from .plot_style import (
        setup_paper_style,
        save_figure,
        add_panel_label,
        CMAP_SKILL,
        CLASS_COLORS,
        PALETTE,
    )
    from .metrics_core import binary_metrics_from_preds
    from .plot_scenarios import derive_scenario_columns
except ImportError:
    from plot_style import (
        setup_paper_style,
        save_figure,
        add_panel_label,
        CMAP_SKILL,
        CLASS_COLORS,
        PALETTE,
    )
    from metrics_core import binary_metrics_from_preds
    from plot_scenarios import derive_scenario_columns


CLASS_SHORT_NAMES = ["Ultra-low", "Moderate-low", "Clear"]
CLASS_BOUNDS = [-0.5, 0.5, 1.5, 2.5]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm(CLASS_BOUNDS, CLASS_CMAP.N)
VIS_MIN_EVENT = 50.0
VIS_MAX_EVENT = 2000.0


def _meta_utc_times(meta):
    """Return the UTC-naive valid-time series used by event plots and IFS matching."""
    time_col = "time_utc" if "time_utc" in meta.columns else "time"
    if time_col not in meta.columns:
        raise ValueError("meta must contain 'time' or 'time_utc' for UTC event evaluation.")
    times = pd.Series(pd.to_datetime(meta[time_col], errors="coerce"), index=meta.index)
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return times


def build_event_visibility_cmap():
    """
    Continuous visibility colormap aligned with class colors.

    Design:
    - very low visibility -> Ultra-low blue
    - 1000 m -> Moderate-low amber
    - high visibility -> Clear gray
    This keeps the first row semantically consistent with the class-color rows.
    """
    fog_side = "#7291B1"  # lighter blue for the mid ultra-low range
    mist_pos = (1000.0 - VIS_MIN_EVENT) / (VIS_MAX_EVENT - VIS_MIN_EVENT)
    anchors = [
        (0.00, PALETTE["Fog"]),
        (0.23, fog_side),          # around the ultra-low threshold neighborhood
        (mist_pos, PALETTE["Mist"]),
        (1.00, PALETTE["Clear"]),
    ]
    return LinearSegmentedColormap.from_list("event_visibility_semantic", anchors)


def load_china_shapefile(shp_path):
    """Load China boundary shapefile."""
    try:
        import geopandas as gpd
        return gpd.read_file(shp_path)
    except Exception as e:
        print(f"  [Spatial] WARN: Could not load shapefile {shp_path}: {e}")
        return None


def plot_station_map(
    sta_df,
    value_col,
    title,
    output_path,
    shp_gdf=None,
    min_events=None,
    mask_col=None,
    cmap=CMAP_SKILL,
    vmin=None,
    vmax=None,
):
    """
    Plot station-level metric on China map.
    sta_df: DataFrame with lon, lat, value_col, and optionally mask_col
    min_events: if mask_col provided, mask stations where mask_col < min_events
    """
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    if shp_gdf is not None:
        shp_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
    else:
        ax.set_xlim(70, 140)
        ax.set_ylim(15, 55)
        ax.set_aspect("equal")

    df = sta_df.copy()
    if mask_col is not None and min_events is not None:
        df = df[df[mask_col] >= min_events]
        n_masked = len(sta_df) - len(df)
        if n_masked > 0:
            print(f"  [Map] Masked {n_masked} stations with {mask_col}<{min_events}")

    if len(df) == 0:
        ax.set_title(title + " (no stations after mask)")
        plt.tight_layout()
        if output_path:
            save_figure(fig, output_path)
        return fig

    vals = df[value_col].values
    lons = df["lon"].values
    lats = df["lat"].values

    valid = np.isfinite(vals)
    if not np.any(valid):
        ax.set_title(title)
        plt.tight_layout()
        if output_path:
            save_figure(fig, output_path)
        return fig

    sc = ax.scatter(
        lons[valid],
        lats[valid],
        c=vals[valid],
        s=8,
        cmap=cmap,
        vmin=vmin if vmin is not None else np.nanmin(vals[valid]),
        vmax=vmax if vmax is not None else np.nanmax(vals[valid]),
    )
    plt.colorbar(sc, ax=ax, label=value_col)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def aggregate_station_metrics(meta, y_true, y_pred, probs=None):
    """
    Aggregate per-station metrics from flat arrays.
    meta: DataFrame with station_id, lat, lon (and optionally hour, month for scenario)
    y_true, y_pred: (N,) arrays
    Returns DataFrame with station_id, lat, lon, fog_recall, fog_precision, fpr_fog, overall_acc, etc.
    """
    meta = meta.reset_index(drop=True)
    n = len(meta)
    if len(y_true) != n or len(y_pred) != n:
        raise ValueError("meta, y_true, y_pred must have same length")

    y_fog = (y_true == 0).astype(int)
    y_mist = (y_true == 1).astype(int)
    y_clear = (y_true == 2).astype(int)
    pred_fog = (y_pred == 0).astype(int)
    pred_mist = (y_pred == 1).astype(int)
    pred_clear = (y_pred == 2).astype(int)

    results = []
    for sid in meta["station_id"].unique():
        mask = (meta["station_id"] == sid).values
        if mask.sum() == 0:
            continue

        n_fog = y_fog[mask].sum()
        n_mist = y_mist[mask].sum()
        n_clear = y_clear[mask].sum()
        hits_fog = ((y_fog[mask] == 1) & (pred_fog[mask] == 1)).sum()
        hits_mist = ((y_mist[mask] == 1) & (pred_mist[mask] == 1)).sum()
        hits_clear = ((y_clear[mask] == 1) & (pred_clear[mask] == 1)).sum()
        fa_fog = ((y_clear[mask] == 1) & (pred_fog[mask] == 1)).sum()

        fog_recall = hits_fog / n_fog if n_fog > 0 else np.nan
        fog_prec = hits_fog / pred_fog[mask].sum() if pred_fog[mask].sum() > 0 else np.nan
        mist_recall = hits_mist / n_mist if n_mist > 0 else np.nan
        mist_prec = hits_mist / pred_mist[mask].sum() if pred_mist[mask].sum() > 0 else np.nan
        fpr_fog = fa_fog / n_clear if n_clear > 0 else np.nan
        acc = (y_true[mask] == y_pred[mask]).mean()

        row = meta[mask].iloc[0][["station_id", "lat", "lon"]].to_dict()
        row["fog_recall"] = fog_recall
        row["fog_precision"] = fog_prec
        row["mist_recall"] = mist_recall
        row["mist_precision"] = mist_prec
        row["fpr_fog"] = fpr_fog
        row["overall_acc"] = acc
        row["n_fog"] = int(n_fog)
        row["n_mist"] = int(n_mist)
        row["n_clear"] = int(n_clear)
        row["n_total"] = int(mask.sum())
        results.append(row)

    return pd.DataFrame(results)


def plot_fog_recall_map(sta_df, output_path, shp_path=None, min_fog_events=5):
    """Ultra-low recall by station."""
    shp = load_china_shapefile(shp_path) if shp_path else None
    return plot_station_map(
        sta_df,
        "fog_recall",
        f"Station Ultra-low Recall (>= {min_fog_events} Ultra-low obs)",
        output_path,
        shp_gdf=shp,
        min_events=min_fog_events,
        mask_col="n_fog",
        vmin=0,
        vmax=1,
    )


def plot_fpr_map(sta_df, output_path, shp_path=None, min_clear_events=20):
    """False positive rate by station."""
    shp = load_china_shapefile(shp_path) if shp_path else None
    return plot_station_map(
        sta_df,
        "fpr_fog",
        f"Station False Positive Rate (≥{min_clear_events} clear obs)",
        output_path,
        shp_gdf=shp,
        min_events=min_clear_events,
        mask_col="n_clear",
        vmin=0,
        vmax=0.2,
    )


def plot_mist_recall_map(sta_df, output_path, shp_path=None, min_mist_events=5):
    """Moderate-low recall by station (same style as the Ultra-low recall map)."""
    shp = load_china_shapefile(shp_path) if shp_path else None
    return plot_station_map(
        sta_df,
        "mist_recall",
        f"Station Moderate-low Recall (>= {min_mist_events} moderate-low obs)",
        output_path,
        shp_gdf=shp,
        min_events=min_mist_events,
        mask_col="n_mist",
        vmin=0,
        vmax=1,
    )


def plot_accuracy_map(sta_df, output_path, shp_path=None, min_total=50):
    """Overall accuracy by station."""
    shp = load_china_shapefile(shp_path) if shp_path else None
    return plot_station_map(
        sta_df,
        "overall_acc",
        f"Station Overall Accuracy (≥{min_total} obs)",
        output_path,
        shp_gdf=shp,
        min_events=min_total,
        mask_col="n_total",
        vmin=0.8,
        vmax=1.0,
    )


def classify_visibility(vis_values, fog_threshold=500.0, mist_threshold=1000.0):
    """Map continuous visibility in meters to Ultra-low/Moderate-low/Clear classes."""
    vis = np.asarray(vis_values, dtype=np.float64)
    cls = np.full(vis.shape, 2, dtype=np.int64)
    cls[vis < mist_threshold] = 1
    cls[vis < fog_threshold] = 0
    return cls


def load_ifs_baseline(meta, ifs_nc_path, vis_var="VIS"):
    """
    Match IFS station visibility to the evaluation samples.

    Returns
    -------
    ifs_preds : np.ndarray
        Class prediction per sample, -1 where IFS is unavailable.
    ifs_vis_raw : np.ndarray
        Raw IFS visibility in meters per sample, NaN where unavailable.
    ifs_valid : np.ndarray[bool]
        Whether the sample has a matched IFS value.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required for IFS event evaluation.") from exc

    if not os.path.exists(ifs_nc_path):
        raise FileNotFoundError(f"IFS NetCDF not found: {ifs_nc_path}")

    ds_ifs = xr.open_dataset(ifs_nc_path)
    try:
        if vis_var not in ds_ifs:
            raise KeyError(f"Variable '{vis_var}' not found in {ifs_nc_path}")
        if "time" not in ds_ifs.coords or "station" not in ds_ifs.coords:
            raise KeyError("IFS dataset must provide 'time' and 'station' coordinates.")

        ifs_vis = np.asarray(ds_ifs[vis_var].values)
        ifs_times = pd.to_datetime(ds_ifs["time"].values)
        ifs_stations = pd.Index(ds_ifs["station"].values.astype(str))

        time_lookup = pd.Series(np.arange(len(ifs_times), dtype=np.int64), index=pd.Index(ifs_times))
        station_lookup = pd.Series(np.arange(len(ifs_stations), dtype=np.int64), index=ifs_stations)

        meta_times = _meta_utc_times(meta)
        meta_stations = meta["station_id"].astype(np.int64).astype(str)
        time_idx = meta_times.map(time_lookup)
        station_idx = meta_stations.map(station_lookup)
        valid = time_idx.notna() & station_idx.notna()

        ifs_vis_raw = np.full(len(meta), np.nan, dtype=np.float64)
        ifs_preds = np.full(len(meta), -1, dtype=np.int64)
        if valid.any():
            t_idx = time_idx[valid].astype(np.int64).to_numpy()
            s_idx = station_idx[valid].astype(np.int64).to_numpy()
            matched_vis = ifs_vis[t_idx, s_idx]
            matched_mask = valid.to_numpy()
            ifs_vis_raw[matched_mask] = matched_vis
            ifs_preds[matched_mask] = classify_visibility(matched_vis)

        print(
            f"  [IFS] Matched {int(valid.sum())}/{len(meta)} samples "
            f"from {os.path.basename(ifs_nc_path)}"
        )
        return ifs_preds, ifs_vis_raw, valid.to_numpy()
    finally:
        ds_ifs.close()


def detect_widespread_fog_events(
    meta,
    y_true,
    top_k=3,
    window_hours=None,
    min_fog_stations=80,
    min_regions=3,
    min_lon_span=10.0,
    min_lat_span=4.0,
    gap_hours=24,
    required_valid_mask=None,
):
    """
    Detect nationwide/widespread Low-vis events from the test set.

    Events are identified from observed Ultra-low stations, then clustered in time as widespread Low-vis events.
    The ranking favors events with many Ultra-low stations and broad regional coverage.
    """
    df = meta[["station_id", "lat", "lon"]].copy()
    df["time"] = _meta_utc_times(meta)
    df["y_true"] = np.asarray(y_true, dtype=np.int64)
    if required_valid_mask is not None:
        required_valid_mask = np.asarray(required_valid_mask, dtype=bool)
        if len(required_valid_mask) != len(df):
            raise ValueError(
                "required_valid_mask length mismatch: "
                f"{len(required_valid_mask)} vs meta rows {len(df)}"
            )
        df["required_valid"] = required_valid_mask
        # Event candidates must be drawn from rows where observation/PMST/IFS
        # are all available. PMST/observation are defined for every test row;
        # this mask adds the IFS availability constraint.
        df = df[df["required_valid"]].copy()
    fog_df = df[df["y_true"] == 0].copy()
    if fog_df.empty:
        return pd.DataFrame(columns=[
            "event_rank", "peak_time", "start_time", "end_time", "duration_h",
            "peak_fog_count", "total_fog_station_hours", "peak_region_count",
            "peak_lon_span", "peak_lat_span", "event_score",
        ])

    fog_df = derive_scenario_columns(fog_df)
    event_columns = [
        "event_rank", "peak_time", "start_time", "end_time", "duration_h",
        "peak_fog_count", "total_fog_station_hours", "peak_region_count",
        "peak_lon_span", "peak_lat_span", "actual_peak_time",
        "actual_peak_fog_count", "actual_peak_region_count", "actual_peak_lon_span",
        "actual_peak_lat_span", "event_score", "selection_tier", "selection_tier_rank",
        "selection_min_fog_stations", "selection_min_regions", "selection_min_lon_span",
        "selection_min_lat_span", "window_start", "window_end", "window_complete",
        "window_available_hours", "window_required_hours",
    ]

    def _region_count(series):
        vals = pd.Series(series)
        vals = vals[vals != "Other"]
        return int(vals.nunique())

    hourly = (
        fog_df.groupby("time")
        .agg(
            n_fog=("station_id", "count"),
            n_regions=("region", _region_count),
            lon_span=("lon", lambda x: float(x.max() - x.min()) if len(x) else 0.0),
            lat_span=("lat", lambda x: float(x.max() - x.min()) if len(x) else 0.0),
        )
        .reset_index()
        .sort_values("time")
    )

    def _empty_events():
        return pd.DataFrame(columns=event_columns)

    available_idx = pd.DatetimeIndex(pd.to_datetime(df["time"]).dropna()).astype("datetime64[ns]")
    available_times = set(available_idx.asi8.tolist())

    def _window_available(peak_time):
        wh = int(window_hours)
        peak = pd.Timestamp(peak_time)
        needed = [peak + pd.Timedelta(hours=offset) for offset in range(-wh, wh + 1)]
        flags = [t.value in available_times for t in needed]
        return pd.Series(
            {
                "window_start": needed[0],
                "window_end": needed[-1],
                "window_complete": bool(all(flags)),
                "window_available_hours": int(sum(flags)),
                "window_required_hours": int(len(flags)),
            }
        )

    def _window_info(events_df):
        if window_hours is None or events_df.empty:
            return events_df
        window_df = events_df["peak_time"].apply(_window_available)
        return pd.concat([events_df.reset_index(drop=True), window_df.reset_index(drop=True)], axis=1)

    def _build_candidates(tier_name, tier_rank, fog_min, regions_min, lon_min, lat_min):
        active = hourly[
            (hourly["n_fog"] >= fog_min) &
            (hourly["n_regions"] >= regions_min) &
            (hourly["lon_span"] >= lon_min) &
            (hourly["lat_span"] >= lat_min)
        ].copy()
        if active.empty:
            return _empty_events()

        events = []
        current_rows = [active.iloc[0]]
        for _, row in active.iloc[1:].iterrows():
            prev_time = pd.Timestamp(current_rows[-1]["time"])
            this_time = pd.Timestamp(row["time"])
            if (this_time - prev_time) <= pd.Timedelta(hours=gap_hours):
                current_rows.append(row)
            else:
                events.append(pd.DataFrame(current_rows))
                current_rows = [row]
        events.append(pd.DataFrame(current_rows))

        event_rows = []
        for ev in events:
            ev = ev.sort_values(["n_fog", "n_regions", "lon_span", "lat_span"], ascending=False)
            actual_peak = ev.iloc[0]
            peak = actual_peak
            if window_hours is not None:
                for _, cand_peak in ev.iterrows():
                    if bool(_window_available(cand_peak["time"])["window_complete"]):
                        peak = cand_peak
                        break
            start_time = pd.Timestamp(ev["time"].min())
            end_time = pd.Timestamp(ev["time"].max())
            duration_h = int((end_time - start_time) / pd.Timedelta(hours=1)) + 1
            total_fog_station_hours = int(ev["n_fog"].sum())
            score = (
                total_fog_station_hours +
                2.0 * float(peak["n_fog"]) +
                40.0 * float(peak["n_regions"]) +
                2.0 * float(peak["lon_span"]) +
                2.0 * float(peak["lat_span"])
            )
            event_rows.append({
                "peak_time": pd.Timestamp(peak["time"]),
                "start_time": start_time,
                "end_time": end_time,
                "duration_h": duration_h,
                "peak_fog_count": int(peak["n_fog"]),
                "total_fog_station_hours": total_fog_station_hours,
                "peak_region_count": int(peak["n_regions"]),
                "peak_lon_span": float(peak["lon_span"]),
                "peak_lat_span": float(peak["lat_span"]),
                "actual_peak_time": pd.Timestamp(actual_peak["time"]),
                "actual_peak_fog_count": int(actual_peak["n_fog"]),
                "actual_peak_region_count": int(actual_peak["n_regions"]),
                "actual_peak_lon_span": float(actual_peak["lon_span"]),
                "actual_peak_lat_span": float(actual_peak["lat_span"]),
                "event_score": float(score),
                "selection_tier": tier_name,
                "selection_tier_rank": int(tier_rank),
                "selection_min_fog_stations": int(fog_min),
                "selection_min_regions": int(regions_min),
                "selection_min_lon_span": float(lon_min),
                "selection_min_lat_span": float(lat_min),
            })

        candidates = pd.DataFrame(event_rows)
        if candidates.empty:
            return _empty_events()
        candidates = _window_info(candidates)
        return candidates.sort_values(
            ["selection_tier_rank", "event_score", "peak_fog_count", "peak_region_count"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

    tiers = [
        (
            "strict",
            0,
            int(min_fog_stations),
            int(min_regions),
            float(min_lon_span),
            float(min_lat_span),
        ),
        (
            "relaxed_spatial",
            1,
            max(1, int(np.ceil(min_fog_stations * 0.85))),
            max(1, int(min_regions) - 1),
            max(0.0, float(min_lon_span) * 0.75),
            max(0.0, float(min_lat_span) * 0.75),
        ),
        (
            "relaxed_count",
            2,
            max(1, int(np.ceil(min_fog_stations * 0.70))),
            max(1, int(min_regions) - 1),
            max(0.0, float(min_lon_span) * 0.60),
            max(0.0, float(min_lat_span) * 0.60),
        ),
        (
            "backup_peak",
            3,
            max(1, int(np.ceil(min_fog_stations * 0.50))),
            1,
            0.0,
            0.0,
        ),
    ]
    all_candidates = pd.concat(
        [_build_candidates(*tier) for tier in tiers],
        axis=0,
        ignore_index=True,
    )
    if all_candidates.empty:
        return _empty_events()

    def _distinct_from_selected(row, selected_rows):
        row_start = pd.Timestamp(row["start_time"])
        row_end = pd.Timestamp(row["end_time"])
        margin = pd.Timedelta(hours=gap_hours)
        for existing in selected_rows:
            ex_start = pd.Timestamp(existing["start_time"])
            ex_end = pd.Timestamp(existing["end_time"])
            if row_start <= ex_end + margin and row_end >= ex_start - margin:
                return False
        return True

    selected_rows = []
    # When a required-valid mask is supplied, do not fall back to incomplete
    # windows: the selected event panels must be fully covered by the
    # observation/model/IFS intersection.
    complete_modes = (True,) if required_valid_mask is not None else (True, False)
    for require_complete in complete_modes:
        for _, row in all_candidates.iterrows():
            if len(selected_rows) >= top_k:
                break
            if require_complete and window_hours is not None and not bool(row.get("window_complete", True)):
                continue
            if _distinct_from_selected(row, selected_rows):
                selected_rows.append(row.to_dict())
        if len(selected_rows) >= top_k:
            break

    if not selected_rows:
        return _empty_events()

    out = pd.DataFrame(selected_rows).sort_values(
        ["selection_tier_rank", "event_score", "peak_fog_count", "peak_region_count"],
        ascending=[True, False, False, False],
    ).head(top_k).reset_index(drop=True)
    if "event_rank" in out.columns:
        out = out.drop(columns=["event_rank"])
    out.insert(0, "event_rank", np.arange(1, len(out) + 1))
    if len(out) < top_k:
        print(
            f"  [Event] Selected {len(out)}/{top_k} distinct events even after relaxed tiers; "
            "check event thresholds, test-set coverage, or IFS overlap.",
            flush=True,
        )
    elif (out["selection_tier"] != "strict").any():
        tiers_used = ", ".join(out["selection_tier"].astype(str).unique())
        print(
            f"  [Event] Filled {top_k} event slots using tiered selection: {tiers_used}.",
            flush=True,
        )
    return out


def _draw_event_basemap(ax, shp_gdf=None):
    """Compact basemap for event panels."""
    if shp_gdf is not None:
        shp_gdf.boundary.plot(ax=ax, color="#404040", linewidth=0.45, zorder=1)
    ax.set_xlim(72, 136)
    ax.set_ylim(17, 54)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#8A8A8A")
        spine.set_linewidth(0.5)


def _format_event_label(event_row):
    peak_time = pd.Timestamp(event_row["peak_time"])
    prefix = f"{peak_time:%Y-%m-%d %H:00} UTC"
    if "actual_peak_time" in event_row and pd.notna(event_row.get("actual_peak_time")):
        actual_peak = pd.Timestamp(event_row["actual_peak_time"])
        if actual_peak != peak_time:
            prefix = f"{prefix} window center; true peak {actual_peak:%Y-%m-%d %H:00} UTC"
    return (
        f"{prefix} | peak Ultra-low={int(event_row['peak_fog_count'])} | "
        f"regions={int(event_row['peak_region_count'])} | "
        f"span={event_row['peak_lon_span']:.1f}°×{event_row['peak_lat_span']:.1f}°"
    )


def _compute_case_metrics(y_true, y_pred):
    """Operational metrics for one time slice."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return {
            "fog_csi": np.nan,
            "fog_recall": np.nan,
            "fog_far": np.nan,
            "low_vis_recall": np.nan,
        }

    fog_metrics = binary_metrics_from_preds((y_true == 0).astype(int), (y_pred == 0).astype(int))
    low_vis_pred = y_pred <= 1
    low_vis_true = y_true <= 1
    low_vis_recall = (
        (low_vis_true & low_vis_pred).sum() / low_vis_true.sum()
        if low_vis_true.sum() > 0 else np.nan
    )
    return {
        "fog_csi": fog_metrics["csi"],
        "fog_recall": fog_metrics["pod"],
        "fog_far": fog_metrics["far"],
        "low_vis_recall": low_vis_recall,
    }


def compute_event_hourly_metrics(
    meta,
    y_true,
    pmst_pred,
    ifs_pred,
    ifs_valid,
    center_time,
    window_hours=3,
):
    """Compute hourly PMST-vs-IFS metrics around one event peak."""
    times = _meta_utc_times(meta)
    rows = []
    for hour_offset in range(-window_hours, window_hours + 1):
        t_now = pd.Timestamp(center_time) + pd.Timedelta(hours=hour_offset)
        time_mask = (times == t_now).to_numpy()
        valid_mask = time_mask & np.asarray(ifs_valid, dtype=bool)
        has_samples = int(time_mask.sum()) > 0

        row = {
            "time": t_now,
            "hour_offset": hour_offset,
            "has_test_samples": bool(has_samples),
            "n_total": int(time_mask.sum()),
            "n_matched_ifs": int(valid_mask.sum()),
        }
        if not has_samples:
            row.update({
                "obs_fog_count": np.nan,
                "obs_low_vis_count": np.nan,
                "pmst_fog_count": np.nan,
                "pmst_low_vis_count": np.nan,
                "ifs_fog_count": np.nan,
                "ifs_low_vis_count": np.nan,
                "pmst_fog_csi": np.nan,
                "pmst_fog_recall": np.nan,
                "pmst_fog_far": np.nan,
                "pmst_low_vis_recall": np.nan,
                "ifs_fog_csi": np.nan,
                "ifs_fog_recall": np.nan,
                "ifs_fog_far": np.nan,
                "ifs_low_vis_recall": np.nan,
            })
            rows.append(row)
            continue

        row.update({
            "obs_fog_count": int(((np.asarray(y_true) == 0) & time_mask).sum()),
            "obs_low_vis_count": int(((np.asarray(y_true) <= 1) & time_mask).sum()),
            "pmst_fog_count": int(((np.asarray(pmst_pred) == 0) & time_mask).sum()),
            "pmst_low_vis_count": int(((np.asarray(pmst_pred) <= 1) & time_mask).sum()),
            "ifs_fog_count": int(((np.asarray(ifs_pred) == 0) & valid_mask).sum()),
            "ifs_low_vis_count": int(((np.asarray(ifs_pred) <= 1) & valid_mask).sum()),
        })

        if valid_mask.sum() == 0:
            row.update({
                "pmst_fog_csi": np.nan,
                "pmst_fog_recall": np.nan,
                "pmst_fog_far": np.nan,
                "pmst_low_vis_recall": np.nan,
                "ifs_fog_csi": np.nan,
                "ifs_fog_recall": np.nan,
                "ifs_fog_far": np.nan,
                "ifs_low_vis_recall": np.nan,
            })
        else:
            y_slice = np.asarray(y_true)[valid_mask]
            pmst_slice = np.asarray(pmst_pred)[valid_mask]
            ifs_slice = np.asarray(ifs_pred)[valid_mask]
            pmst_metrics = _compute_case_metrics(y_slice, pmst_slice)
            ifs_metrics = _compute_case_metrics(y_slice, ifs_slice)
            row.update({f"pmst_{k}": v for k, v in pmst_metrics.items()})
            row.update({f"ifs_{k}": v for k, v in ifs_metrics.items()})

        rows.append(row)
    return pd.DataFrame(rows)


def plot_widespread_event_panels(
    meta,
    y_true_raw,
    pmst_pred,
    ifs_pred,
    ifs_valid,
    event_row,
    output_path,
    shp_gdf=None,
    window_hours=3,
):
    """
    Multi-hour event panel:
    row 1 = observed visibility (continuous)
    row 2 = PMST prediction (class)
    row 3 = IFS prediction (class)
    """
    setup_paper_style()
    center_time = pd.Timestamp(event_row["peak_time"])
    hour_offsets = list(range(-window_hours, window_hours + 1))
    ncols = len(hour_offsets)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 8.6))

    vis_norm = Normalize(VIS_MIN_EVENT, VIS_MAX_EVENT)
    vis_cmap = build_event_visibility_cmap()
    vis_mappable = None

    row_labels = ["Observed visibility", "PMST forecast", "IFS diagnostic VIS"]
    times = _meta_utc_times(meta)

    for col_idx, offset in enumerate(hour_offsets):
        t_now = center_time + pd.Timedelta(hours=offset)
        time_mask = (times == t_now).to_numpy()

        title = f"{t_now:%m-%d}\n{t_now:%H}:00"
        if offset == 0:
            title = f"{t_now:%m-%d}\n{t_now:%H}:00 peak"

        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            _draw_event_basemap(ax, shp_gdf)
            if row_idx == 0:
                ax.set_title(title, fontsize=9, pad=5, color=PALETTE["Fog"] if offset == 0 else "#202020")
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_labels[row_idx],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

            if time_mask.sum() == 0:
                ax.text(0.5, 0.5, "No samples", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="#666666")
                continue

            event_df = meta.loc[time_mask, ["lon", "lat"]].copy()
            lons = event_df["lon"].to_numpy()
            lats = event_df["lat"].to_numpy()

            if row_idx == 0:
                vis_vals = np.asarray(y_true_raw)[time_mask]
                vis_plot = np.clip(vis_vals, VIS_MIN_EVENT, VIS_MAX_EVENT)
                vis_mappable = ax.scatter(
                    lons,
                    lats,
                    c=vis_plot,
                    cmap=vis_cmap,
                    norm=vis_norm,
                    s=9,
                    linewidths=0.05,
                    edgecolors="#FFFFFF",
                    zorder=3,
                    alpha=0.92,
                )
                ax.text(
                    0.02,
                    0.03,
                    f"median={np.nanmedian(vis_vals):.0f} m",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
                )
            else:
                preds = np.asarray(pmst_pred if row_idx == 1 else ifs_pred)[time_mask]
                valid = np.ones(time_mask.sum(), dtype=bool)
                if row_idx == 2:
                    valid = np.asarray(ifs_valid)[time_mask]
                    if (~valid).any():
                        ax.scatter(
                            lons[~valid],
                            lats[~valid],
                            s=7,
                            c="#D8D8D8",
                            linewidths=0,
                            zorder=2,
                            alpha=0.7,
                        )

                ax.scatter(
                    lons[valid],
                    lats[valid],
                    c=preds[valid].astype(float),
                    cmap=CLASS_CMAP,
                    norm=CLASS_NORM,
                    s=9,
                    linewidths=0.05,
                    edgecolors="#FFFFFF",
                    zorder=3,
                    alpha=0.95,
                )
                if valid.any():
                    counts = np.bincount(preds[valid].astype(int), minlength=3)
                    ax.text(
                        0.02,
                        0.03,
                        f"UL={counts[0]} ML={counts[1]} C={counts[2]}",
                        transform=ax.transAxes,
                        fontsize=6.5,
                        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
                    )

    if vis_mappable is not None:
        cbar = fig.colorbar(
            vis_mappable,
            ax=axes[0, :].tolist(),
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
        )
        cbar.set_ticks([VIS_MIN_EVENT, 500.0, 1000.0, VIS_MAX_EVENT])
        cbar.set_ticklabels(["50", "500", "1000", "2000"])
        cbar.set_label(
            "Observed visibility (m) | 500 m=Ultra-low boundary, 1000 m=Low-vis event boundary"
        )

    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(facecolor=CLASS_COLORS[i], edgecolor="none", label=CLASS_SHORT_NAMES[i]) for i in range(3)],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Widespread Low-vis Event Case\n" + _format_event_label(event_row),
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, output_path)
    return fig


def plot_three_events_footprint_row(
    meta,
    y_true_raw,
    pmst_pred,
    event_df,
    output_path,
    shp_gdf=None,
    window_hours=6,
):
    """
    One figure, three event columns side by side. Each column: 2 rows × n_hours maps
    (observed visibility, then PMST class) from -window_hours to +window_hours around peak.
    """
    setup_paper_style()
    if event_df is None or len(event_df) == 0:
        print("  [Event] plot_three_events_footprint_row: empty event_df, skip.")
        return None

    event_df = event_df.head(3).copy()
    n_ev = len(event_df)
    offsets = list(range(-int(window_hours), int(window_hours) + 1))
    n_h = len(offsets)
    times = _meta_utc_times(meta)
    y_true_raw = np.asarray(y_true_raw, dtype=np.float64)
    pmst_pred = np.asarray(pmst_pred, dtype=np.int64)

    vis_norm = Normalize(VIS_MIN_EVENT, VIS_MAX_EVENT)
    vis_cmap = build_event_visibility_cmap()
    vis_mappable = None

    fig = plt.figure(figsize=(max(14.0, n_h * n_ev * 0.95), 6.4))
    gs_outer = fig.add_gridspec(1, n_ev, wspace=0.26, left=0.04, right=0.98, top=0.86, bottom=0.14)

    for ei, (_, er) in enumerate(event_df.iterrows()):
        center_time = pd.Timestamp(er["peak_time"])
        rank = int(er.get("event_rank", ei + 1))
        gsi = gs_outer[0, ei].subgridspec(2, n_h, hspace=0.14, wspace=0.03)
        col_title = f"E{rank}  {center_time.strftime('%Y-%m-%d %H:00')} UTC"

        for j, off in enumerate(offsets):
            t_now = center_time + pd.Timedelta(hours=off)
            time_mask = (times == t_now).to_numpy()

            for row_idx in range(2):
                ax = fig.add_subplot(gsi[row_idx, j])
                _draw_event_basemap(ax, shp_gdf)
                if row_idx == 0:
                    ttl = t_now.strftime("%m-%d\n%H:00")
                    if off == 0:
                        ttl = t_now.strftime("%m-%d\npeak %H:00")
                    ax.set_title(ttl, fontsize=7.5, pad=2)

                if time_mask.sum() == 0:
                    ax.text(
                        0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="#999999",
                    )
                    continue

                lons = meta.loc[time_mask, "lon"].to_numpy()
                lats = meta.loc[time_mask, "lat"].to_numpy()

                if row_idx == 0:
                    vis_vals = y_true_raw[time_mask]
                    vis_plot = np.clip(vis_vals, VIS_MIN_EVENT, VIS_MAX_EVENT)
                    vis_mappable = ax.scatter(
                        lons,
                        lats,
                        c=vis_plot,
                        cmap=vis_cmap,
                        norm=vis_norm,
                        s=8,
                        linewidths=0.05,
                        edgecolors="#FFFFFF",
                        zorder=3,
                        alpha=0.92,
                    )
                else:
                    preds = pmst_pred[time_mask]
                    ax.scatter(
                        lons,
                        lats,
                        c=preds.astype(float),
                        cmap=CLASS_CMAP,
                        norm=CLASS_NORM,
                        s=8,
                        linewidths=0.05,
                        edgecolors="#FFFFFF",
                        zorder=3,
                        alpha=0.95,
                    )

        fig.text(
            (0.04 + (ei + 0.5) / n_ev * 0.94),
            0.93,
            col_title,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="600",
            transform=fig.transFigure,
        )

    fig.text(0.02, 0.48, "Observed\nvisibility (m)", rotation=90, va="center", ha="center", fontsize=10, fontweight="600")
    fig.text(0.02, 0.22, "PMST\n(3-class)", rotation=90, va="center", ha="center", fontsize=10, fontweight="600")

    if vis_mappable is not None:
        cax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
        cb = fig.colorbar(vis_mappable, cax=cax, orientation="horizontal")
        cb.set_ticks([VIS_MIN_EVENT, 500.0, 1000.0, VIS_MAX_EVENT])
        cb.set_ticklabels(["50", "500", "1000", "2000"])
        cb.set_label("Observed visibility (m)")

    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(facecolor=CLASS_COLORS[i], edgecolor="none", label=CLASS_SHORT_NAMES[i]) for i in range(3)],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    fig.suptitle(
        "Three Low-vis events — spatial footprint: observed visibility vs PMST forecast",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    save_figure(fig, output_path)
    plt.close(fig)
    return fig


def plot_three_events_peak_row(
    meta,
    y_true_raw,
    pmst_pred,
    event_df,
    output_path,
    shp_gdf=None,
):
    """One row × three columns: observed visibility at peak hour for each event."""
    setup_paper_style()
    if event_df is None or len(event_df) == 0:
        print("  [Event] plot_three_events_peak_row: empty event_df, skip.")
        return None

    event_df = event_df.head(3).copy()
    times = _meta_utc_times(meta)
    y_true_raw = np.asarray(y_true_raw, dtype=np.float64)
    pmst_pred = np.asarray(pmst_pred, dtype=np.int64)

    vis_norm = Normalize(VIS_MIN_EVENT, VIS_MAX_EVENT)
    vis_cmap = build_event_visibility_cmap()
    vis_mappable = None

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, er) in zip(axes, event_df.iterrows()):
        center_time = pd.Timestamp(er["peak_time"])
        rank = int(er.get("event_rank", 1))
        time_mask = (times == center_time).to_numpy()
        _draw_event_basemap(ax, shp_gdf)

        if time_mask.sum() == 0:
            ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        else:
            lons = meta.loc[time_mask, "lon"].to_numpy()
            lats = meta.loc[time_mask, "lat"].to_numpy()
            vis_vals = y_true_raw[time_mask]
            vis_plot = np.clip(vis_vals, VIS_MIN_EVENT, VIS_MAX_EVENT)
            vis_mappable = ax.scatter(
                lons,
                lats,
                c=vis_plot,
                cmap=vis_cmap,
                norm=vis_norm,
                s=14,
                linewidths=0.06,
                edgecolors="#FFFFFF",
                zorder=3,
                alpha=0.92,
            )
            pmst = pmst_pred[time_mask]
            ax.set_xlabel(
                f"PMST peak: Ultra-low={(pmst == 0).sum()} Moderate-low={(pmst == 1).sum()} Clear={(pmst == 2).sum()}",
                fontsize=8,
                color="#4b5563",
            )

        ax.set_title(
            f"E{rank} peak — observed visibility\n{center_time.strftime('%Y-%m-%d %H:00')} UTC",
            fontsize=10,
            fontweight="600",
            pad=6,
        )

    if vis_mappable is not None:
        cax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
        cb = fig.colorbar(vis_mappable, cax=cax, orientation="horizontal")
        cb.set_ticks([VIS_MIN_EVENT, 500.0, 1000.0, VIS_MAX_EVENT])
        cb.set_ticklabels(["50", "500", "1000", "2000"])
        cb.set_label("Visibility (m)")

    fig.suptitle(
        "Three widespread Low-vis events — observed visibility at peak hour",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, output_path)
    plt.close(fig)
    return fig


def plot_event_metric_comparison(hourly_df, event_row, output_path):
    """Time-evolving event metrics comparing PMST and IFS."""
    setup_paper_style()

    pmst_color = PALETTE["Fog"]
    ifs_color = "#505050"
    obs_color = "#111111"
    x = hourly_df["hour_offset"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True)
    axes = axes.flatten()

    counts_ax = axes[0]
    counts_ax.plot(x, hourly_df["obs_fog_count"], color=obs_color, marker="o", lw=2.2, label="Obs Ultra-low count")
    counts_ax.plot(x, hourly_df["obs_low_vis_count"], color=obs_color, marker="o", lw=1.4, ls="--", label="Obs Low-vis event count")
    counts_ax.plot(x, hourly_df["pmst_fog_count"], color=pmst_color, marker="o", lw=2.0, label="PMST Ultra-low count")
    counts_ax.plot(x, hourly_df["pmst_low_vis_count"], color=pmst_color, marker="o", lw=1.4, ls="--", label="PMST Low-vis event count")
    counts_ax.plot(x, hourly_df["ifs_fog_count"], color=ifs_color, marker="s", lw=2.0, label="IFS Ultra-low count")
    counts_ax.plot(x, hourly_df["ifs_low_vis_count"], color=ifs_color, marker="s", lw=1.4, ls="--", label="IFS Low-vis event count")
    counts_ax.set_ylabel("Station Count")
    counts_ax.set_title("Event Footprint Evolution")
    counts_ax.grid(alpha=0.3)

    metric_specs = [
        ("fog_csi", "Ultra-low CSI"),
        ("fog_recall", "Ultra-low recall"),
        ("fog_far", "Ultra-low FAR (lower better)"),
        ("low_vis_recall", "Low-vis event recall"),
    ]
    for ax, (metric, title) in zip(axes[1:], metric_specs):
        ax.plot(x, hourly_df[f"pmst_{metric}"], color=pmst_color, marker="o", lw=2.0, label="PMST")
        ax.plot(x, hourly_df[f"ifs_{metric}"], color=ifs_color, marker="s", lw=2.0, label="IFS")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    for ax in axes[1 + len(metric_specs):]:
        ax.axis("off")

    for idx, ax in enumerate(axes):
        if not ax.axison:
            continue
        add_panel_label(ax, chr(ord("a") + idx), x=-0.12, y=1.02)
        ax.axvline(0, color="#888888", lw=1.0, ls="--", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:+d}h" if v != 0 else "0h" for v in x])
        if idx >= 3:
            ax.set_xlabel("Hour Relative to Event Peak")

    handles, labels = counts_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "PMST vs IFS During Widespread Low-vis Event\n" + _format_event_label(event_row),
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, output_path)
    return fig


def summarize_event_metrics(hourly_df, event_row):
    """Collapse hourly event metrics into one summary row."""
    peak_mask = hourly_df["hour_offset"] == 0

    def _safe_mean(series_name):
        vals = hourly_df[series_name].to_numpy(dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

    def _safe_peak(series_name):
        vals = hourly_df.loc[peak_mask, series_name].to_numpy(dtype=float)
        return float(np.nanmean(vals)) if len(vals) and np.isfinite(vals).any() else np.nan

    peak_time = pd.Timestamp(event_row["peak_time"])
    return {
        "event_rank": int(event_row["event_rank"]),
        "peak_time": peak_time,
        "peak_fog_count": int(event_row["peak_fog_count"]),
        "peak_region_count": int(event_row["peak_region_count"]),
        "duration_h": int(event_row["duration_h"]),
        "pmst_fog_csi_mean": _safe_mean("pmst_fog_csi"),
        "ifs_fog_csi_mean": _safe_mean("ifs_fog_csi"),
        "pmst_fog_recall_mean": _safe_mean("pmst_fog_recall"),
        "ifs_fog_recall_mean": _safe_mean("ifs_fog_recall"),
        "pmst_low_vis_recall_mean": _safe_mean("pmst_low_vis_recall"),
        "ifs_low_vis_recall_mean": _safe_mean("ifs_low_vis_recall"),
        "pmst_fog_csi_peak": _safe_peak("pmst_fog_csi"),
        "ifs_fog_csi_peak": _safe_peak("ifs_fog_csi"),
        "pmst_fog_recall_peak": _safe_peak("pmst_fog_recall"),
        "ifs_fog_recall_peak": _safe_peak("ifs_fog_recall"),
        "pmst_low_vis_recall_peak": _safe_peak("pmst_low_vis_recall"),
        "ifs_low_vis_recall_peak": _safe_peak("ifs_low_vis_recall"),
    }


def plot_event_summary_comparison(summary_df, output_path):
    """Single figure summarizing PMST vs IFS across all selected events."""
    if summary_df.empty:
        return None

    setup_paper_style()
    pmst_color = PALETTE["Fog"]
    ifs_color = "#5B5B5B"

    event_labels = [
        f"E{int(r.event_rank)}\n{pd.Timestamp(r.peak_time):%m-%d %H}"
        for r in summary_df.itertuples()
    ]
    x = np.arange(len(summary_df))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    axes = axes.flatten()
    panels = [
        ("fog_csi_mean", "Event-Mean Ultra-low CSI"),
        ("fog_recall_mean", "Event-Mean Ultra-low recall"),
        ("low_vis_recall_mean", "Event-Mean Low-vis event recall"),
        ("low_vis_recall_peak", "Peak-hour Low-vis event recall"),
    ]

    for idx, (ax, (suffix, title)) in enumerate(zip(axes, panels)):
        pmst_vals = summary_df[f"pmst_{suffix}"].to_numpy(dtype=float)
        ifs_vals = summary_df[f"ifs_{suffix}"].to_numpy(dtype=float)
        ax.bar(x - width / 2, pmst_vals, width, color=pmst_color, label="PMST")
        ax.bar(x + width / 2, ifs_vals, width, color=ifs_color, label="IFS")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        add_panel_label(ax, chr(ord("a") + idx), x=-0.12, y=1.02)

        for xi, pmst_v, ifs_v in zip(x, pmst_vals, ifs_vals):
            if np.isfinite(pmst_v) and np.isfinite(ifs_v):
                delta = pmst_v - ifs_v
                ax.text(
                    xi,
                    max(pmst_v, ifs_v) + 0.03,
                    f"Δ={delta:+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=pmst_color if delta >= 0 else ifs_color,
                )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(event_labels)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Widespread Low-vis Events: PMST vs IFS Summary", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output_path)
    return fig


def run_widespread_event_evaluation(
    meta,
    y_true,
    y_true_raw,
    pmst_pred,
    output_dir,
    shp_path=None,
    ifs_nc_path=None,
    top_k=3,
    window_hours=3,
    min_fog_stations=80,
    min_regions=3,
    min_lon_span=10.0,
    min_lat_span=4.0,
    gap_hours=24,
):
    """
    End-to-end event evaluation:
    1. match IFS baseline
    2. identify nationwide/widespread Low-vis events
    3. save spatial case panels and PMST-vs-IFS metric panels
    """
    if ifs_nc_path is None:
        print("  [Event] No IFS file provided, skipping widespread Low-vis event evaluation.")
        return pd.DataFrame()
    if not os.path.exists(ifs_nc_path):
        print(f"  [Event] IFS file not found: {ifs_nc_path}. Skipping event evaluation.")
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    shp_gdf = load_china_shapefile(shp_path) if shp_path and os.path.exists(shp_path) else None
    ifs_pred, _, ifs_valid = load_ifs_baseline(meta, ifs_nc_path)
    print(
        "  [Event] Selecting cases on observation/model/IFS intersection: "
        f"{int(np.asarray(ifs_valid, dtype=bool).sum())}/{len(meta)} rows matched.",
        flush=True,
    )

    event_df = detect_widespread_fog_events(
        meta,
        y_true,
        top_k=top_k,
        window_hours=window_hours,
        min_fog_stations=min_fog_stations,
        min_regions=min_regions,
        min_lon_span=min_lon_span,
        min_lat_span=min_lat_span,
        gap_hours=gap_hours,
        required_valid_mask=ifs_valid,
    )
    summary_path = os.path.join(output_dir, "event_case_summary.csv")
    event_df.to_csv(summary_path, index=False)
    print(f"  [Event] Summary saved → {summary_path}")

    if event_df.empty:
        print("  [Event] No widespread Low-vis events met the current thresholds.")
        return event_df

    event_summary_rows = []
    for _, event_row in event_df.iterrows():
        rank = int(event_row["event_rank"])
        spatial_path = os.path.join(output_dir, f"fig9_event_{rank}_spatial.png")
        metrics_path = os.path.join(output_dir, f"fig9_event_{rank}_metrics.png")
        hourly_metrics = compute_event_hourly_metrics(
            meta,
            y_true,
            pmst_pred,
            ifs_pred,
            ifs_valid,
            center_time=event_row["peak_time"],
            window_hours=window_hours,
        )
        hourly_csv = os.path.join(output_dir, f"fig9_event_{rank}_hourly_metrics.csv")
        hourly_metrics.to_csv(hourly_csv, index=False, float_format="%.4f")
        print(f"  [Event] Event {rank} hourly metrics → {hourly_csv}")
        event_summary_rows.append(summarize_event_metrics(hourly_metrics, event_row))

        plot_widespread_event_panels(
            meta,
            y_true_raw,
            pmst_pred,
            ifs_pred,
            ifs_valid,
            event_row,
            spatial_path,
            shp_gdf=shp_gdf,
            window_hours=window_hours,
        )
        plot_event_metric_comparison(hourly_metrics, event_row, metrics_path)

    event_summary_df = pd.DataFrame(event_summary_rows)
    event_summary_csv = os.path.join(output_dir, "fig9_event_summary_metrics.csv")
    event_summary_df.to_csv(event_summary_csv, index=False, float_format="%.4f")
    print(f"  [Event] Summary metrics → {event_summary_csv}")
    plot_event_summary_comparison(
        event_summary_df,
        os.path.join(output_dir, "fig9_event_summary.png"),
    )

    return event_df


__all__ = [
    "load_china_shapefile",
    "plot_station_map",
    "aggregate_station_metrics",
    "plot_fog_recall_map",
    "plot_mist_recall_map",
    "plot_fpr_map",
    "plot_accuracy_map",
    "detect_widespread_fog_events",
    "plot_three_events_footprint_row",
    "plot_three_events_peak_row",
    "run_widespread_event_evaluation",
]
