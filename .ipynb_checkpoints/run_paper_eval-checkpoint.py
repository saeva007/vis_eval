#!/usr/bin/env python3
"""
One-click paper evaluation pipeline for low-visibility model.
Loads model, runs inference, computes rare-event metrics, generates figures/tables.
Usage: python run_paper_eval.py [--exp-id EXP_ID] [--output-dir DIR]

Note: If you see "hipThreadExchangeStreamCaptureMode" / libgalaxyhip.so errors,
this is a PyTorch vs cluster HIP/AMD driver mismatch. Try:
  - CPU-only: install PyTorch CPU build, or run with CUDA_VISIBLE_DEVICES="" and
    use a PyTorch build that does not load HIP when no GPU is visible.
  - Or run on a node with ROCm/HIP version matching your torch build.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Add paths for imports
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "train"))
sys.path.insert(0, PAPER_EVAL_DIR)

# Torch and model imports deferred to main() to allow --help without GPU/HIP
from metrics_core import (
    apply_daynight_offsets,
    pred_from_joint_thresholds,
    pred_from_thresholds,
    pred_from_season_thresholds,
    thresholds_from_season_thresholds,
    compute_rare_event_report,
)
from .plot_style import setup_paper_style, save_figure
from .plot_classification import (
    plot_confusion_matrix_normalized,
    plot_per_class_prf1,
    plot_pr_curves,
    plot_threshold_sweep,
    plot_reliability_diagram,
)
from .plot_spatial import (
    plot_fog_recall_map,
    plot_fpr_map,
    run_widespread_event_evaluation,
)
from .plot_scenarios import (
    plot_scenario_bars,
    derive_scenario_columns,
    build_confusion_summaries_and_bottleneck_table,
)


def load_test_data(data_dir, scaler_path, window_size=12):
    """Load X_test, y_test, meta_test."""
    X_path = os.path.join(data_dir, "X_test.npy")
    y_path = os.path.join(data_dir, "y_test.npy")
    meta_path = os.path.join(data_dir, "meta_test.csv")

    for p in [X_path, y_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    y_raw = np.load(y_path)
    if np.max(y_raw) < 100:
        y_raw = y_raw * 1000.0

    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    y_cls[y_raw >= 500] = 1
    y_cls[y_raw >= 1000] = 2

    meta = pd.read_csv(meta_path, parse_dates=["time"])
    meta["hour"] = meta["time"].dt.hour
    meta["month"] = meta["time"].dt.month

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return X_path, y_cls, y_raw, meta, scaler


# Must match CONFIG['FE_EXTRA_DIMS'] in PMST_net_test_10_s2.py (model expects this many extra dims)
EXTRA_FEAT_DIMS = 36


def run_inference(X_path, scaler, model, device, batch_size=1024, window_size=12, temperature=None):
    """Run model inference, return probs (temperature-scaled if temperature is set)."""
    torch = globals().get("torch")
    F = globals().get("F")
    if torch is None or F is None:
        import torch as _torch
        import torch.nn.functional as _F
        torch, F = _torch, _F
    X = np.load(X_path, mmap_mode="r")
    N = len(X)
    split_dyn = 25 * window_size
    log_mask = np.zeros(split_dyn, dtype=bool)
    for t in range(window_size):
        for i in [2, 4, 9]:
            log_mask[t * 25 + i] = True

    use_cuda = device.type == "cuda"
    non_blocking = use_cuda
    T = 1.0 if temperature is None or temperature == 1.0 else float(temperature)
    all_probs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        rows = X[start:end].astype(np.float32)

        feats = rows[:, : split_dyn + 5]
        feats[:, :split_dyn] = np.where(
            log_mask,
            np.log1p(np.maximum(feats[:, :split_dyn], 0)),
            feats[:, :split_dyn],
        )
        if scaler is not None:
            feats = (feats - scaler.center_) / (scaler.scale_ + 1e-6)

        veg = rows[:, split_dyn + 5 : split_dyn + 6]
        extra = rows[:, split_dyn + 6 :]
        if extra.shape[1] < EXTRA_FEAT_DIMS:
            extra = np.pad(extra, ((0, 0), (0, EXTRA_FEAT_DIMS - extra.shape[1])), mode="constant", constant_values=0)
        elif extra.shape[1] > EXTRA_FEAT_DIMS:
            extra = extra[:, :EXTRA_FEAT_DIMS]
        final = np.concatenate([np.clip(feats, -10, 10), veg, np.clip(extra, -10, 10)], axis=1)
        final = np.nan_to_num(final, nan=0.0)

        x = torch.from_numpy(final).float().to(device, non_blocking=non_blocking)
        with torch.inference_mode():
            fine, _, _ = model(x)
            probs = F.softmax(fine / T, dim=1)
        all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    return probs


def extract_postprocess_features(X_path, window_size=12):
    X = np.load(X_path, mmap_mode="r")
    split_dyn = window_size * 25
    last = X[:, split_dyn - 25:split_dyn].astype(np.float32)
    u10 = last[:, 5]
    v10 = last[:, 7]
    u925 = last[:, 13]
    v925 = last[:, 15]
    wspd10 = np.maximum(last[:, 6], 0.0)
    sw_rad = np.maximum(last[:, 4], 0.0)
    precip = np.maximum(last[:, 2], 0.0)
    cape = np.maximum(last[:, 9], 0.0)
    inversion = last[:, 23]
    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    mixing = (
        1.0 / (1.0 + np.exp(-(sw_rad - 150.0) / 75.0))
    ) * (
        1.0 / (1.0 + np.exp(-(wspd10 - 4.0) / 1.5))
    ) * (
        1.0 / (1.0 + np.exp(-(-inversion + 0.5) / 1.2))
    )
    convective = (
        1.0 / (1.0 + np.exp(-(np.log1p(cape) - np.log(200.0)) * 1.6))
    ) * (
        1.0 / (1.0 + np.exp(-(np.log1p(precip) - np.log(0.1)) * 2.5))
    )
    ventilation = np.tanh((wspd10 * (1.0 + shear_mag)) / 12.0)
    return {
        "is_day": last[:, 24] <= 90.0,
        "mixing": mixing,
        "convective": convective,
        "ventilation": ventilation,
    }


def apply_jja_prior_filters(probs, months, regime, prior_cfg):
    if not prior_cfg:
        return probs
    probs = probs.copy()
    jja = np.isin(months, [6, 7, 8])
    if not np.any(jja):
        return probs
    low_vis_factor = np.ones(jja.sum(), dtype=np.float64)
    low_vis_factor *= np.where(regime["mixing"][jja] > 0.55, prior_cfg.get("mixing_low_vis", 1.0), 1.0)
    low_vis_factor *= np.where(regime["ventilation"][jja] > 0.55, prior_cfg.get("ventilation_low_vis", 1.0), 1.0)
    probs[jja, 0] *= np.where(regime["convective"][jja] > 0.45, prior_cfg.get("convective_fog", 1.0), 1.0)
    probs[jja, 0] *= low_vis_factor
    probs[jja, 1] *= low_vis_factor
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-6, None)
    return probs


def aggregate_station_metrics(meta, y_true, preds):
    """Per-station fog recall, fog precision, FPR, accuracy."""
    stations = meta[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    rows = []
    for _, row in stations.iterrows():
        sid = row["station_id"]
        m = (meta["station_id"] == sid).values
        y_s = y_true[m]
        p_s = preds[m]

        n_fog_true = (y_s == 0).sum()
        n_clear_true = (y_s == 2).sum()
        n_total = len(y_s)

        fog_rec = (y_s == 0) & (p_s == 0)
        fog_recall = fog_rec.sum() / n_fog_true if n_fog_true > 0 else 0.0

        fog_pred = (p_s == 0)
        fog_prec = (y_s == 0) & fog_pred
        fog_precision = fog_prec.sum() / fog_pred.sum() if fog_pred.sum() > 0 else 0.0

        fpr = (p_s <= 1) & (y_s == 2)
        fpr_val = fpr.sum() / n_clear_true if n_clear_true > 0 else 0.0

        acc = (p_s == y_s).mean()

        rows.append({
            "station_id": sid, "lat": row["lat"], "lon": row["lon"],
            "fog_recall": fog_recall, "fog_precision": fog_precision,
            "fpr_fog": fpr_val, "overall_acc": acc,
            "n_fog": n_fog_true, "n_clear": n_clear_true, "n_total": n_total,
        })

    return pd.DataFrame(rows)


def visibility_boundary_band(y_raw):
    """Label each sample by visibility band: <400, 400-600, 600-800, 800-1200, >1200 (meters)."""
    y_raw = np.asarray(y_raw, dtype=np.float64)
    band = np.full(len(y_raw), ">1200", dtype=object)
    band[y_raw < 400] = "<400"
    band[(y_raw >= 400) & (y_raw < 600)] = "400-600"
    band[(y_raw >= 600) & (y_raw < 800)] = "600-800"
    band[(y_raw >= 800) & (y_raw < 1200)] = "800-1200"
    return band


def export_per_sample_table(meta, y_true, y_raw, pred, probs, output_path):
    """
    Export per-sample evaluation table: y_true, y_raw, pred, p_fog, p_mist,
    month, hour, station_id, lat, lon, visibility_band.
    """
    df = meta[["station_id", "lat", "lon", "time", "month", "hour"]].copy()
    df["y_true"] = y_true
    df["y_raw"] = y_raw
    df["pred"] = pred
    df["p_fog"] = probs[:, 0]
    df["p_mist"] = probs[:, 1]
    df["visibility_band"] = visibility_boundary_band(y_raw)
    df.to_csv(output_path, index=False, float_format="%.6f")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="exp_1772692533")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=os.path.join(BASE, "ml_dataset_fe_12h"))
    parser.add_argument("--ckpt-dir", default=os.path.join(BASE, "checkpoints"))
    parser.add_argument("--shp-path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    parser.add_argument("--fog-th", type=float, default=0.46, help="Global fog threshold (used if no season artifact)")
    parser.add_argument("--mist-th", type=float, default=0.38, help="Global mist threshold (used if no season artifact)")
    parser.add_argument("--no-calibration", action="store_true", help="Ignore season_thresholds.pt and temperature; use only --fog-th/--mist-th")
    parser.add_argument("--ifs-nc", default=os.path.join(BASE, "VIS_IDW_KDTree_20250101_20251231.nc"))
    parser.add_argument("--event-top-k", type=int, default=3)
    parser.add_argument("--event-window-hours", type=int, default=3)
    parser.add_argument("--event-min-fog-stations", type=int, default=80)
    parser.add_argument("--event-min-regions", type=int, default=3)
    parser.add_argument("--event-min-lon-span", type=float, default=10.0)
    parser.add_argument("--event-min-lat-span", type=float, default=4.0)
    parser.add_argument("--event-gap-hours", type=int, default=24)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (avoids loading HIP/CUDA; use if torch import fails on cluster)")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size (default: 8192 on GPU, 1024 on CPU)")
    parser.add_argument("--extra-feat-dims", type=int, default=EXTRA_FEAT_DIMS, help="S2 extra feature dimension used by the checkpoint")
    args = parser.parse_args()

    # Import torch only when needed; catch HIP/GPU lib errors
    try:
        import torch
        import torch.nn.functional as F
        globals()["torch"] = torch
        globals()["F"] = F
        if args.cpu or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except (ImportError, OSError) as e:
        err = str(e)
        if "hip" in err.lower() or "HIP" in err or "libgalaxyhip" in err or "symbol" in err:
            print("PyTorch failed to load (HIP/AMD driver mismatch on this node).")
            print("Try: 1) python -m paper_eval.run_paper_eval --cpu  (if you have CPU-only torch)")
            print("     2) Use a conda env with CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            print("     3) Run on a node where ROCm/HIP version matches your PyTorch build.")
        raise

    output_dir = args.output_dir or os.path.join(BASE, f"paper_eval_results_{args.exp_id}")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.exp_id}_S2_PhaseB_best_score.pt")
    scaler_path = os.path.join(args.ckpt_dir, "robust_scaler_w12.pkl")
    season_th_path = os.path.join(args.ckpt_dir, f"{args.exp_id}_season_thresholds.pt")

    # Load calibration artifact by default (season thresholds + temperature)
    season_thresholds = None
    temperature = None
    global_thresholds = None
    daynight_offsets = None
    prior_filters = None
    best_postprocess_mode = None
    if not args.no_calibration and os.path.exists(season_th_path):
        try:
            try:
                cal = torch.load(season_th_path, map_location="cpu", weights_only=True)
            except TypeError:
                cal = torch.load(season_th_path, map_location="cpu")
            season_thresholds = cal.get("season_thresholds") or None
            temperature = cal.get("temperature")
            global_thresholds = cal.get("global_thresholds") or None
            daynight_offsets = cal.get("daynight_offsets") or None
            prior_filters = cal.get("prior_filters") or None
            best_postprocess_mode = cal.get("best_postprocess_mode")
            if temperature is not None:
                temperature = float(temperature)
            print(
                f"  [Calibration] Loaded {season_th_path}: temperature={temperature}, "
                f"seasons={list(season_thresholds.keys()) if season_thresholds else []}, "
                f"mode={best_postprocess_mode}",
            )
        except Exception as e:
            print(f"  [WARN] Could not load calibration artifact: {e}")
    if season_thresholds is None:
        print("  [Calibration] Using global thresholds (no artifact or --no-calibration).")

    # Load model (import here so torch is already loaded)
    from PMST_net_test_10_s2 import ImprovedDualStreamPMSTNet

    print("=" * 60)
    print("Paper Evaluation Pipeline")
    print("=" * 60)
    print(f"  Exp ID    : {args.exp_id}")
    print(f"  Output   : {output_dir}")
    print(f"  Device   : {device}")
    print()

    # 1. Load model
    model = ImprovedDualStreamPMSTNet(
        window_size=12, hidden_dim=512, dropout=0.3, extra_feat_dim=args.extra_feat_dims
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    # 多卡推理：DataParallel 自动把 batch 分到各 DCU
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[1] Model loaded (DataParallel on {torch.cuda.device_count()} devices).")
    else:
        print("[1] Model loaded.")

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 8192 if device.type == "cuda" else 1024
    print(f"     Inference batch_size={batch_size}")

    # 2. Load data
    X_path, y_cls, y_raw, meta, scaler = load_test_data(
        args.data_dir, scaler_path, window_size=12
    )
    print(f"[2] Data loaded: {len(y_cls)} samples, {meta['station_id'].nunique()} stations.")

    # 3. Inference (with temperature scaling when calibration is loaded)
    probs = run_inference(
        X_path, scaler, model, device,
        batch_size=batch_size, window_size=12, temperature=temperature,
    )
    months = meta["month"].values
    regime = extract_postprocess_features(X_path, window_size=12)
    if season_thresholds is not None:
        fog_th, mist_th = thresholds_from_season_thresholds(
            months, season_thresholds, args.fog_th, args.mist_th
        )
        if daynight_offsets and best_postprocess_mode and "daynight" in str(best_postprocess_mode):
            fog_th, mist_th = apply_daynight_offsets(
                fog_th, mist_th, months, regime["is_day"], daynight_offsets
            )
        if prior_filters and best_postprocess_mode and "priors" in str(best_postprocess_mode):
            probs = apply_jja_prior_filters(probs, months, regime, prior_filters)
        if best_postprocess_mode and "joint" in str(best_postprocess_mode):
            preds = pred_from_joint_thresholds(probs, fog_th, mist_th)
        else:
            preds = pred_from_thresholds(probs, fog_th, mist_th)
    elif global_thresholds is not None:
        if best_postprocess_mode and "joint" in str(best_postprocess_mode):
            preds = pred_from_joint_thresholds(
                probs,
                global_thresholds.get("fog_th", args.fog_th),
                global_thresholds.get("mist_th", args.mist_th),
            )
        else:
            preds = pred_from_thresholds(
                probs,
                global_thresholds.get("fog_th", args.fog_th),
                global_thresholds.get("mist_th", args.mist_th),
            )
    else:
        preds = pred_from_thresholds(probs, args.fog_th, args.mist_th)
    print("[3] Inference done.")

    # 4. Rare-event report (use preds when season thresholds were applied)
    report = compute_rare_event_report(
        probs, y_cls, args.fog_th, args.mist_th, pred=preds,
    )
    print("[4] Rare-event metrics:")
    for k, v in report.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}")

    # 5. Save report
    with open(os.path.join(output_dir, "rare_event_report.txt"), "w") as f:
        f.write(classification_report(y_cls, preds, target_names=["Fog", "Mist", "Clear"]))
        f.write("\n\nRare-event metrics:\n")
        for k, v in report.items():
            if isinstance(v, (int, float)):
                f.write(f"  {k}: {v:.4f}\n")

    # 6. Station aggregation
    sta_df = aggregate_station_metrics(meta, y_cls, preds)
    sta_df.to_csv(os.path.join(output_dir, "station_metrics.csv"), index=False)
    print(f"[5] Station metrics saved ({len(sta_df)} stations).")

    # 6b. Per-sample table and confusion/bottleneck diagnostics (Priority 1)
    per_sample_path = os.path.join(output_dir, "per_sample_eval.csv")
    eval_df = export_per_sample_table(
        meta, y_cls, y_raw, preds, probs, per_sample_path,
    )
    print(f"[5b] Per-sample table saved: {per_sample_path} ({len(eval_df)} rows).")
    eval_df = derive_scenario_columns(eval_df)
    build_confusion_summaries_and_bottleneck_table(eval_df, output_dir)

    # 7. Generate figures
    setup_paper_style()
    class_names = ["Fog", "Mist", "Clear"]

    plot_confusion_matrix_normalized(
        y_cls, preds, class_names,
        os.path.join(output_dir, "fig3_confusion_matrix.png"),
    )

    plot_per_class_prf1(report, os.path.join(output_dir, "fig3_prf1_bars.png"))
    plot_pr_curves(probs, y_cls, class_names, os.path.join(output_dir, "fig4_pr_curves.png"))
    plot_threshold_sweep(probs, y_cls, os.path.join(output_dir, "fig4_threshold_sweep.png"),
                         args.fog_th, args.mist_th)
    plot_reliability_diagram(probs, y_cls, os.path.join(output_dir, "fig5_reliability.png"))

    if os.path.exists(args.shp_path):
        plot_fog_recall_map(
            sta_df,
            os.path.join(output_dir, "fig8_station_fog_recall.png"),
            shp_path=args.shp_path,
            min_fog_events=5,
        )
        plot_fpr_map(
            sta_df,
            os.path.join(output_dir, "fig8_station_fpr.png"),
            shp_path=args.shp_path,
            min_clear_events=20,
        )
    else:
        print("  [WARN] Shapefile not found, skipping maps.")

    # Scenario plot: hour, month, is_coastal, region derived from time, lon, lat in plot_scenario_bars
    meta_full = meta.copy()
    meta_full["y_true"] = y_cls
    meta_full["pred"] = preds
    meta_full["p_fog"] = probs[:, 0]
    meta_full["p_mist"] = probs[:, 1]
    plot_scenario_bars(meta_full, os.path.join(output_dir, "fig7_scenario_robustness.png"),
                       fog_th=args.fog_th, mist_th=args.mist_th)

    try:
        event_df = run_widespread_event_evaluation(
            meta=meta,
            y_true=y_cls,
            y_true_raw=y_raw,
            pmst_pred=preds,
            output_dir=output_dir,
            shp_path=args.shp_path,
            ifs_nc_path=args.ifs_nc,
            top_k=args.event_top_k,
            window_hours=args.event_window_hours,
            min_fog_stations=args.event_min_fog_stations,
            min_regions=args.event_min_regions,
            min_lon_span=args.event_min_lon_span,
            min_lat_span=args.event_min_lat_span,
            gap_hours=args.event_gap_hours,
        )
        if len(event_df) > 0:
            print(f"[6] Widespread fog-event evaluation saved for {len(event_df)} events.")
    except Exception as e:
        print(f"  [WARN] Event evaluation skipped: {e}")

    print()
    print("=" * 60)
    print("Evaluation complete.")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
