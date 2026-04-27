"""
Paper Figure 7: Scenario robustness (season, day/night, coastal/inland, region).
All stratification fields (hour, month, is_coastal, region) are derived from
metadata columns: time, station_id, lon, lat.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

try:
    from .plot_style import setup_paper_style, apply_palette, save_figure, CLASS_COLORS
    from .metrics_core import (
        pred_from_thresholds,
        pred_from_thresholds_mutual,
        pred_from_joint_thresholds,
        binary_metrics_from_preds,
    )
except ImportError:
    from plot_style import setup_paper_style, apply_palette, save_figure, CLASS_COLORS
    from metrics_core import (
        pred_from_thresholds,
        pred_from_thresholds_mutual,
        pred_from_joint_thresholds,
        binary_metrics_from_preds,
    )


def _pred_from_threshold_rule(probs, fog_th, mist_th, threshold_rule="default"):
    """
    Map threshold_rule to the same pred_* helpers as the training / paper_eval notebooks.
    - default: pred_from_thresholds (mist band then fog overwrite)
    - mutual:  pred_from_thresholds_mutual (aligned with ComprehensiveMetrics._build_full_metrics)
    - joint:   pred_from_joint_thresholds
    """
    rule = (threshold_rule or "default").lower()
    if rule == "mutual":
        return pred_from_thresholds_mutual(probs, fog_th, mist_th)
    if rule == "joint":
        return pred_from_joint_thresholds(probs, fog_th, mist_th)
    return pred_from_thresholds(probs, fog_th, mist_th)


# -----------------------------------------------------------------------------
# Region definitions for China (lon/lat bounds, approximate)
# Based on geographic divisions: 东北、华北、华中、华东、华南、西南、西北
# -----------------------------------------------------------------------------
CHINA_REGION_DEFS = [
    # (name, lat_min, lat_max, lon_min, lon_max)
    ("Northeast", 38.5, 54, 118, 136),       # 东北: 黑龙江、吉林、辽宁
    ("North_China", 32, 42.5, 110, 120),     # 华北: 京津冀晋内蒙中南部
    ("East_China", 27, 35, 115, 123),        # 华东: 江浙沪皖闽赣鲁
    ("Central_China", 26, 34, 108, 118),     # 华中: 豫鄂湘
    ("South_China", 18, 26, 105, 120),       # 华南: 粤桂琼
    ("Southwest", 21, 35, 97, 108),          # 西南: 渝川云贵藏
    ("Northwest", 31, 50, 73, 111),          # 西北: 陕甘青宁新
]

# Coastal definition: east of 118°E, lat 18–42° (approximate eastern seaboard)
COASTAL_LON_MIN = 118
COASTAL_LAT_MIN = 18
COASTAL_LAT_MAX = 42


def enrich_meta_forecast_init(meta):
    """
    From optional columns init_time (YYYYMMDDHH or int) add init_hour (0–23).
    S2 per-init datasets (s2_data_per_init_split_pm10) provide init_time per sample.
    """
    df = meta.copy()
    if "init_hour" in df.columns:
        return df
    if "init_time" not in df.columns:
        df["init_hour"] = np.nan
        return df
    s = df["init_time"].astype(str).str.replace(r"\.0$", "", regex=True)
    init_dt = pd.to_datetime(s, format="%Y%m%d%H", errors="coerce")
    if init_dt.isna().all():
        init_dt = pd.to_datetime(df["init_time"], errors="coerce")
    df["init_dt"] = init_dt
    df["init_hour"] = init_dt.dt.hour
    return df


def derive_scenario_columns(meta):
    """
    Derive hour, month, is_coastal, region from time, lon, lat.
    meta: DataFrame with required columns [time, lon, lat] (or time, longitude, latitude).
    station_id optional.
    """
    df = meta.copy()
    # Alias common column names
    if "latitude" in df.columns and "lat" not in df.columns:
        df["lat"] = df["latitude"]
    if "longitude" in df.columns and "lon" not in df.columns:
        df["lon"] = df["longitude"]
    for c in ["time", "lon", "lat"]:
        if c not in df.columns:
            raise ValueError(f"derive_scenario_columns requires '{c}' in meta. Got: {list(meta.columns)}")
    # Parse time if string
    if df["time"].dtype == object or "datetime" not in str(df["time"].dtype):
        df["time"] = pd.to_datetime(df["time"])

    # Hour (0–23) and month (1–12) from time
    df["hour"] = df["time"].dt.hour.values
    df["month"] = df["time"].dt.month.values

    lats = df["lat"].values
    lons = df["lon"].values

    # is_coastal: 1 if in eastern coastal zone (lon>=118, 18<=lat<=42)
    df["is_coastal"] = (
        (lons >= COASTAL_LON_MIN) & (lats >= COASTAL_LAT_MIN) & (lats <= COASTAL_LAT_MAX)
    ).astype(int)

    # region: first match in CHINA_REGION_DEFS, else "Other"
    region = np.full(len(df), "Other", dtype=object)
    for name, lat_min, lat_max, lon_min, lon_max in CHINA_REGION_DEFS:
        mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
        region[mask] = name
    df["region"] = region

    df = enrich_meta_forecast_init(df)
    return df


def _compute_scenario_metrics(y_true, y_pred):
    """Compute detailed metrics for a subset (y_true, y_pred)."""
    y_fog = (y_true == 0).astype(np.int64)
    y_mist = (y_true == 1).astype(np.int64)
    y_clear = (y_true == 2).astype(np.int64)
    pred_fog = (y_pred == 0).astype(np.int64)
    pred_mist = (y_pred == 1).astype(np.int64)

    m_fog = binary_metrics_from_preds(y_fog, pred_fog)
    m_mist = binary_metrics_from_preds(y_mist, pred_mist)

    # Low-vis precision (when pred Fog or Mist, fraction that is truly Fog or Mist)
    low_vis_pred = (y_pred <= 1)
    low_vis_true = (y_true <= 1)
    lv_prec = (low_vis_true & low_vis_pred).sum() / low_vis_pred.sum() if low_vis_pred.sum() > 0 else np.nan

    # FPR: fraction of clear-sky samples predicted as low-vis
    fpr = (y_pred <= 1) & (y_true == 2)
    fpr_val = fpr.sum() / y_clear.sum() if y_clear.sum() > 0 else np.nan

    # Overall accuracy
    acc = (y_pred == y_true).mean()

    # Per-class precision (when pred X, fraction that is truly X)
    n_fog_pred = pred_fog.sum()
    n_mist_pred = pred_mist.sum()
    fog_prec = ((y_true == 0) & (y_pred == 0)).sum() / n_fog_pred if n_fog_pred > 0 else np.nan
    mist_prec = ((y_true == 1) & (y_pred == 1)).sum() / n_mist_pred if n_mist_pred > 0 else np.nan

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    # Binary F1 for rare-class one-vs-rest (same hard labels as CSI: fog / not-fog, mist / not-mist).
    fog_f1 = float(f1_score(y_fog, pred_fog, zero_division=0))
    mist_f1 = float(f1_score(y_mist, pred_mist, zero_division=0))
    # HSS / TSS from the same 2×2 tables as CSI (TSS = POD − FPR == pss_fpr in metrics_core).
    fog_hss = float(m_fog["hss"]) if np.isfinite(m_fog["hss"]) else 0.0
    mist_hss = float(m_mist["hss"]) if np.isfinite(m_mist["hss"]) else 0.0
    fog_tss = float(m_fog["pss_fpr"]) if np.isfinite(m_fog["pss_fpr"]) else 0.0
    mist_tss = float(m_mist["pss_fpr"]) if np.isfinite(m_mist["pss_fpr"]) else 0.0

    return {
        "fog_csi": m_fog["csi"],
        "fog_pod": m_fog["pod"],
        "fog_precision": fog_prec,
        "fog_far": m_fog["far"],
        "fog_pss": m_fog["pss"],
        "fog_f1": fog_f1,
        "fog_hss": fog_hss,
        "fog_tss": fog_tss,
        "mist_csi": m_mist["csi"],
        "mist_pod": m_mist["pod"],
        "mist_precision": mist_prec if np.isfinite(mist_prec) else np.nan,
        "mist_far": m_mist["far"],
        "mist_pss": m_mist["pss"],
        "mist_f1": mist_f1,
        "mist_hss": mist_hss,
        "mist_tss": mist_tss,
        "lv_precision": lv_prec,
        "fpr": fpr_val,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n": len(y_true),
        "n_fog": int(y_fog.sum()),
        "n_mist": int(y_mist.sum()),
        "n_clear": int(y_clear.sum()),
    }


def eval_scenario_detailed(
    meta,
    y_true,
    probs,
    fog_th=0.46,
    mist_th=0.38,
    threshold_rule="default",
    pred=None,
):
    """
    Detailed scenario evaluation. All stratification derived from meta[time,lon,lat].
    Returns (results_dict, meta_enriched).

    If ``pred`` is provided (same length as y_true), it must be the operational class
    predictions used for the main confusion matrix / rare-event report — scenario
    metrics will use this array and will **not** re-apply a different threshold rule.

    If ``pred`` is None, predictions are built from ``probs`` using ``threshold_rule``
    (default | mutual | joint), matching ``metrics_core.pred_from_*``.
    """
    if pred is not None:
        pred = np.asarray(pred, dtype=np.int64).ravel()
        if len(pred) != len(y_true):
            raise ValueError(
                f"eval_scenario_detailed: len(pred)={len(pred)} != len(y_true)={len(y_true)}"
            )
    else:
        pred = _pred_from_threshold_rule(probs, fog_th, mist_th, threshold_rule)
    meta_derived = derive_scenario_columns(meta)
    n = len(meta_derived)

    # Day/Night: 06–18 local as daytime (common meteorological convention)
    hour = meta_derived["hour"].values
    day_cond = (hour >= 6) & (hour < 18)

    # Season (Northern Hemisphere)
    month = meta_derived["month"].values
    djf = (month == 12) | (month <= 2)
    mam = (month >= 3) & (month <= 5)
    jja = (month >= 6) & (month <= 8)
    son = (month >= 9) & (month <= 11)

    # Time-of-day stratification: morning 6–12, afternoon 12–18, midnight 0–6, evening 18–24
    morning = (hour >= 6) & (hour < 12)
    afternoon = (hour >= 12) & (hour < 18)
    midnight = (hour >= 0) & (hour < 6)   # 凌晨
    evening = (hour >= 18)                 # 傍晚至午夜

    scenarios = [
        ("All", np.ones(n, dtype=bool)),
        ("Day (06–18h)", day_cond),
        ("Night (18–06h)", ~day_cond),
        ("Morning (06–12h)", morning),
        ("Afternoon (12–18h)", afternoon),
        ("Midnight (00–06h)", midnight),
        ("Evening (18–24h)", evening),
        ("DJF (Dec–Feb)", djf),
        ("MAM (Mar–May)", mam),
        ("JJA (Jun–Aug)", jja),
        ("SON (Sep–Nov)", son),
        ("Coastal", meta_derived["is_coastal"].values == 1),
        ("Inland", meta_derived["is_coastal"].values == 0),
    ]

    results = {}
    min_samples = 50  # lower threshold for more granular scenarios

    for name, mask in scenarios:
        if mask.sum() < min_samples:
            results[name] = _empty_result(int(mask.sum()))
            continue
        metrics = _compute_scenario_metrics(y_true[mask], pred[mask])
        results[name] = metrics

    # Per region
    for rn in sorted(meta_derived["region"].unique()):
        mask = (meta_derived["region"] == rn).values
        if mask.sum() < min_samples:
            continue
        metrics = _compute_scenario_metrics(y_true[mask], pred[mask])
        results[f"Region_{rn}"] = metrics

    # Forecast cycle (init hour): 00Z vs 12Z vs other — distinct from valid-time hour
    if "init_hour" in meta_derived.columns:
        ih_series = meta_derived["init_hour"]
        if ih_series.notna().any():
            for h in sorted(ih_series.dropna().unique()):
                hi = int(h)
                mask = (meta_derived["init_hour"] == hi).values
                name = "Forecast_init_{:02d}UTC".format(hi)
                if mask.sum() < min_samples:
                    results[name] = _empty_result(int(mask.sum()))
                    continue
                metrics = _compute_scenario_metrics(y_true[mask], pred[mask])
                results[name] = metrics

    return results, meta_derived


def _empty_result(n):
    return {
        "fog_csi": np.nan, "fog_pod": np.nan, "fog_precision": np.nan, "fog_far": np.nan, "fog_pss": np.nan,
        "fog_f1": np.nan, "fog_hss": np.nan, "fog_tss": np.nan,
        "mist_csi": np.nan, "mist_pod": np.nan, "mist_precision": np.nan, "mist_far": np.nan, "mist_pss": np.nan,
        "mist_f1": np.nan, "mist_hss": np.nan, "mist_tss": np.nan,
        "lv_precision": np.nan, "fpr": np.nan, "accuracy": np.nan, "macro_f1": np.nan,
        "n": n, "n_fog": np.nan, "n_mist": np.nan, "n_clear": np.nan,
    }


def eval_scenario(meta, y_true, probs, fog_th=0.46, mist_th=0.38, threshold_rule="default", pred=None):
    """
    Backward-compatible wrapper. Returns dict scenario_name -> {fog_csi, fog_pod, mist_csi, lv_precision, n}.
    """
    results, _ = eval_scenario_detailed(
        meta, y_true, probs, fog_th, mist_th,
        threshold_rule=threshold_rule, pred=pred,
    )
    # Simplify to legacy format
    out = {}
    for k, v in results.items():
        out[k] = {
            "fog_csi": v["fog_csi"],
            "fog_pod": v["fog_pod"],
            "mist_csi": v["mist_csi"],
            "lv_precision": v["lv_precision"],
            "n": v["n"],
        }
    return out


def save_scenario_table(results, output_path):
    """Save detailed scenario metrics to CSV."""
    rows = []
    for name, m in results.items():
        if m["n"] < 10:
            continue
        rows.append({
            "scenario": name,
            "n": m["n"],
            "n_fog": m.get("n_fog", ""),
            "n_mist": m.get("n_mist", ""),
            "n_clear": m.get("n_clear", ""),
            "fog_csi": m["fog_csi"],
            "fog_pod": m["fog_pod"],
            "fog_precision": m.get("fog_precision", np.nan),
            "fog_far": m.get("fog_far", np.nan),
            "fog_pss": m.get("fog_pss", np.nan),
            "fog_f1": m.get("fog_f1", np.nan),
            "fog_hss": m.get("fog_hss", np.nan),
            "fog_tss": m.get("fog_tss", np.nan),
            "mist_csi": m["mist_csi"],
            "mist_pod": m["mist_pod"],
            "mist_precision": m.get("mist_precision", np.nan),
            "mist_far": m.get("mist_far", np.nan),
            "mist_pss": m.get("mist_pss", np.nan),
            "mist_f1": m.get("mist_f1", np.nan),
            "mist_hss": m.get("mist_hss", np.nan),
            "mist_tss": m.get("mist_tss", np.nan),
            "lv_precision": m["lv_precision"],
            "fpr": m.get("fpr", np.nan),
            "accuracy": m.get("accuracy", np.nan),
            "macro_f1": m.get("macro_f1", np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.4f")
    return df


def plot_scenario_robustness(scenario_results, output_path):
    """Grouped bar chart of Fog CSI, Fog POD, Mist CSI, low-vis precision per scenario."""
    setup_paper_style()
    apply_palette()

    names = [k for k in scenario_results if scenario_results[k]["n"] >= 50]
    metrics = ["fog_csi", "fog_pod", "mist_csi", "lv_precision"]
    labels = ["Fog CSI", "Fog POD", "Mist CSI", "Low-vis Prec"]

    x = np.arange(len(names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [scenario_results[n].get(m, np.nan) for n in names]
        vals = [float(v) if np.isfinite(v) else 0 for v in vals]
        ax.bar(x + (i - 2) * w, vals, w, label=lab)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Scenario Robustness (derived from time, lon, lat)")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_scenario_detailed_multipanel(scenario_results, output_path):
    """Four-panel figure: Fog CSI, Mist CSI, Low-vis Precision, FPR by scenario."""
    setup_paper_style()
    apply_palette()

    names = [k for k in scenario_results if scenario_results[k]["n"] >= 50]
    if len(names) == 0:
        return None

    panels = [
        ("fog_csi", "Fog CSI", "viridis"),
        ("mist_csi", "Mist CSI", "plasma"),
        ("lv_precision", "Low-vis Precision", "cividis"),
        ("fpr", "False Positive Rate (lower=better)", "Reds_r"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, title, cmap) in enumerate(panels):
        ax = axes[idx]
        vals = [scenario_results[n].get(metric, np.nan) for n in names]
        vals = [float(v) if np.isfinite(v) else 0 for v in vals]
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.8, len(vals)))
        bars = ax.barh(names, vals, color=colors)
        ax.set_xlim(0, 1.05 if metric != "fpr" else max(vals) * 1.1 if vals else 1)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Detailed Scenario Evaluation (time/lon/lat derived)", fontsize=12, y=1.02)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_scenario_bars(meta, output_path, fog_th=0.46, mist_th=0.38, threshold_rule="default", **kwargs):
    """
    Main entry for run_paper_eval. Expects meta with: time, station_id, lon, lat, y_true, pred.
    Or: time, station_id, lon, lat, y_true, p_fog, p_mist (then build pred via threshold_rule).

    When ``meta`` includes ``pred``, that column is the single source of truth for scenario
    metrics (same labels as confusion matrix / rare-event report). When ``pred`` is absent,
    use ``threshold_rule`` (default | mutual | joint) consistently with ``metrics_core``.
    """
    y_true = meta["y_true"].values
    if "pred" in meta.columns:
        pred = meta["pred"].values
        probs = np.column_stack([
            meta["p_fog"].values if "p_fog" in meta.columns else np.zeros(len(meta)),
            meta["p_mist"].values if "p_mist" in meta.columns else np.zeros(len(meta)),
            1 - (meta["p_fog"].values if "p_fog" in meta.columns else np.zeros(len(meta))) - (meta["p_mist"].values if "p_mist" in meta.columns else np.zeros(len(meta))),
        ])
    else:
        probs = np.column_stack([
            meta["p_fog"].values,
            meta["p_mist"].values,
            1 - meta["p_fog"].values - meta["p_mist"].values,
        ])
        pred = _pred_from_threshold_rule(probs, fog_th, mist_th, threshold_rule)

    results, _ = eval_scenario_detailed(
        meta, y_true, probs, fog_th, mist_th,
        threshold_rule=threshold_rule,
        pred=pred,
    )

    # Save detailed table alongside figure
    base = os.path.splitext(output_path)[0]
    table_path = base + "_scenario_table.csv"
    save_scenario_table(results, table_path)

    # Main bar chart
    plot_scenario_robustness(results, output_path)

    # Detailed multipanel (optional)
    detailed_path = base + "_detailed.png"
    plot_scenario_detailed_multipanel(results, detailed_path)

    return results


def save_metrics_by_valid_hour(y_true, pred, meta_derived, output_path, min_bin=20):
    """
    Stratify by valid-time hour (0–23) from meta time. One peak per 24 h is normal:
    this is *observation/valid* local diurnal structure, not forecast-cycle count.
    """
    hour = np.asarray(meta_derived["hour"].values, dtype=np.int64)
    rows = []
    for hv in range(24):
        m = hour == hv
        n_b = int(m.sum())
        if n_b < min_bin:
            rows.append({"valid_hour_utc": hv, "n": n_b, "note": "below_min_bin"})
            continue
        met = _compute_scenario_metrics(y_true[m], pred[m])
        row = {"valid_hour_utc": hv, "n": n_b, "note": ""}
        row.update(met)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.4f")
    return pd.DataFrame(rows)


def plot_forecast_init_comparison(results, output_path, min_n=50):
    """
    Focused bar chart: Forecast_init_00UTC vs Forecast_init_12UTC (and other init hours).
    """
    keys = [
        k
        for k in sorted(results.keys())
        if k.startswith("Forecast_init_") and results[k].get("n", 0) >= min_n
    ]
    if len(keys) == 0:
        return None
    setup_paper_style()
    apply_palette()
    metrics = ["fog_csi", "fog_pod", "mist_csi", "lv_precision"]
    labels = ["Fog CSI", "Fog POD", "Mist CSI", "Low-vis Prec"]
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.8), 5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [float(results[k].get(m, np.nan) or 0) if np.isfinite(results[k].get(m, np.nan)) else 0 for k in keys]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Metrics by forecast initialization hour (UTC)")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def save_forecast_init_metrics_table(results, output_path):
    """CSV rows for Forecast_init_* scenarios only."""
    rows = []
    for k, v in sorted(results.items()):
        if not k.startswith("Forecast_init_"):
            continue
        if v.get("n", 0) < 1:
            continue
        r = {"scenario": k}
        r.update({kk: v.get(kk) for kk in v if kk != "scenario"})
        rows.append(r)
    if not rows:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.4f")
    return df


# -----------------------------------------------------------------------------
# Stratified confusion summaries and bottleneck table (Priority 1 diagnostics)
# -----------------------------------------------------------------------------

CONFUSION_TYPES = ["fog->mist", "mist->fog", "mist->clear", "clear->mist", "clear->fog"]


def _confusion_type(y_true, pred):
    """Single-row: return confusion type string or 'correct'."""
    if y_true == pred:
        return "correct"
    if y_true == 0 and pred == 1:
        return "fog->mist"
    if y_true == 1 and pred == 0:
        return "mist->fog"
    if y_true == 1 and pred == 2:
        return "mist->clear"
    if y_true == 2 and pred == 1:
        return "clear->mist"
    if y_true == 2 and pred == 0:
        return "clear->fog"
    return "other"


def add_confusion_type_column(df):
    """Add column confusion_type to DataFrame with y_true, pred."""
    ct = [
        _confusion_type(int(df["y_true"].iloc[i]), int(df["pred"].iloc[i]))
        for i in range(len(df))
    ]
    df = df.copy()
    df["confusion_type"] = ct
    return df


def _season_from_month(month):
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    return "Other"


def _daynight_from_hour(hour):
    return "day" if 6 <= int(hour) < 18 else "night"


def build_confusion_summaries_and_bottleneck_table(eval_df, output_dir):
    """
    From per-sample eval DataFrame (with y_true, pred, month, hour, is_coastal,
    region, visibility_band), produce:
    1. Stratified confusion summaries (counts and rates by season, day/night,
       coastal/inland, region, visibility_band).
    2. One bottleneck table ranking top error types by count and rate, with
       JJA and boundary-zone errors separated from full-set average.
    """
    df = add_confusion_type_column(eval_df)
    df["season"] = df["month"].map(_season_from_month)
    df["day_night"] = df["hour"].map(_daynight_from_hour)
    df["coastal_inland"] = df["is_coastal"].map(lambda x: "coastal" if x == 1 else "inland")

    n_total = len(df)

    # ----- Stratified confusion summaries -----
    strata = [
        ("season", "season"),
        ("day_night", "day_night"),
        ("coastal_inland", "coastal_inland"),
        ("region", "region"),
        ("visibility_band", "visibility_band"),
    ]
    rows_summary = []
    for slice_name, col in strata:
        if col not in df.columns:
            continue
        for slice_val in df[col].dropna().unique():
            sub = df[df[col] == slice_val]
            n_s = len(sub)
            for ct in CONFUSION_TYPES:
                cnt = (sub["confusion_type"] == ct).sum()
                rate_all = cnt / n_s if n_s > 0 else 0.0
                n_err_s = (sub["confusion_type"] != "correct").sum()
                rate_among_errors = cnt / n_err_s if n_err_s > 0 else 0.0
                rows_summary.append({
                    "slice_dim": slice_name,
                    "slice_value": slice_val,
                    "confusion_type": ct,
                    "count": int(cnt),
                    "n_slice": int(n_s),
                    "rate_over_slice": round(rate_all, 6),
                    "rate_among_errors_slice": round(rate_among_errors, 6),
                })
    summary_df = pd.DataFrame(rows_summary)
    summary_path = os.path.join(output_dir, "confusion_summaries_stratified.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  [Diagnostics] Stratified confusion summaries -> {summary_path}")

    # ----- Bottleneck table: top error types by count and rate -----
    full_counts = {ct: (df["confusion_type"] == ct).sum() for ct in CONFUSION_TYPES}
    full_rates = {ct: full_counts[ct] / n_total if n_total > 0 else 0 for ct in CONFUSION_TYPES}
    jja_mask = df["season"] == "JJA"
    n_jja = int(jja_mask.sum())
    jja_counts = {ct: (df.loc[jja_mask, "confusion_type"] == ct).sum() for ct in CONFUSION_TYPES}
    jja_rates = {ct: jja_counts[ct] / n_jja if n_jja > 0 else 0 for ct in CONFUSION_TYPES}
    boundary_bands = ["400-600", "600-800", "800-1200"]
    bnd_mask = df["visibility_band"].isin(boundary_bands)
    n_bnd = int(bnd_mask.sum())
    bnd_counts = {ct: (df.loc[bnd_mask, "confusion_type"] == ct).sum() for ct in CONFUSION_TYPES}
    bnd_rates = {ct: bnd_counts[ct] / n_bnd if n_bnd > 0 else 0 for ct in CONFUSION_TYPES}

    bottleneck_rows = []
    for ct in CONFUSION_TYPES:
        bottleneck_rows.append({
            "confusion_type": ct,
            "scope": "full",
            "count": full_counts[ct],
            "rate": round(full_rates[ct], 6),
            "n_scope": n_total,
        })
        bottleneck_rows.append({
            "confusion_type": ct,
            "scope": "JJA",
            "count": jja_counts[ct],
            "rate": round(jja_rates[ct], 6),
            "n_scope": n_jja,
        })
        bottleneck_rows.append({
            "confusion_type": ct,
            "scope": "boundary_zone",
            "count": bnd_counts[ct],
            "rate": round(bnd_rates[ct], 6),
            "n_scope": n_bnd,
        })
    bottleneck_df = pd.DataFrame(bottleneck_rows)
    full_rank = bottleneck_df[bottleneck_df["scope"] == "full"].sort_values("count", ascending=False)
    order = full_rank["confusion_type"].tolist()
    bottleneck_df["rank_full_count"] = bottleneck_df["confusion_type"].map(
        {c: i + 1 for i, c in enumerate(order)}
    )
    bottleneck_path = os.path.join(output_dir, "bottleneck_table.csv")
    bottleneck_df.to_csv(bottleneck_path, index=False)
    print(f"  [Diagnostics] Bottleneck table (JJA + boundary-zone) -> {bottleneck_path}")


__all__ = [
    "derive_scenario_columns",
    "eval_scenario_detailed",
    "eval_scenario",
    "save_scenario_table",
    "plot_scenario_robustness",
    "plot_scenario_detailed_multipanel",
    "plot_scenario_bars",
    "add_confusion_type_column",
    "build_confusion_summaries_and_bottleneck_table",
]
