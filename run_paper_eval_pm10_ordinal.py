#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import classification_report


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "train"))
sys.path.insert(0, PAPER_EVAL_DIR)

from .metrics_core import compute_rare_event_report
from .plot_style import setup_paper_style
from .plot_classification import (
    plot_confusion_matrix_normalized,
    plot_per_class_prf1,
    plot_pr_curves,
    plot_reliability_diagram,
)
from .plot_spatial import (
    plot_fog_recall_map,
    plot_fpr_map,
    run_widespread_event_evaluation,
)
from .plot_scenarios import (
    derive_scenario_columns,
    build_confusion_summaries_and_bottleneck_table,
    enrich_meta_forecast_init,
    save_metrics_by_valid_hour,
    plot_forecast_init_comparison,
    save_forecast_init_metrics_table,
    plot_scenario_bars,
)
from .run_paper_eval import (
    load_test_data,
    aggregate_station_metrics,
    export_per_sample_table,
)


EXTRA_FEAT_DIMS = 36
DYN_VARS_COUNT = 26


def cumulative_to_probs(fine_logits):
    p_500 = 1.0 / (1.0 + np.exp(-fine_logits[:, 0]))
    p_1000 = 1.0 / (1.0 + np.exp(-fine_logits[:, 1]))
    p_1000 = np.maximum(p_1000, p_500)
    p_fog = p_500
    p_mist = np.maximum(p_1000 - p_500, 0.0)
    p_clear = np.maximum(1.0 - p_1000, 0.0)
    probs = np.column_stack([p_fog, p_mist, p_clear])
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-6, None)
    return probs, np.column_stack([p_500, p_1000])


def pred_from_ordinal_thresholds(cum_probs, fog_th, low_vis_th):
    p_500 = np.asarray(cum_probs[:, 0], dtype=np.float64)
    p_1000 = np.asarray(cum_probs[:, 1], dtype=np.float64)
    p_1000 = np.maximum(p_1000, p_500)
    preds = np.full(len(cum_probs), 2, dtype=np.int64)
    preds[p_1000 >= low_vis_th] = 1
    preds[p_500 >= fog_th] = 0
    return preds


def run_inference_ordinal(
    X_path,
    scaler,
    model,
    device,
    batch_size=1024,
    window_size=12,
    extra_feat_dims=EXTRA_FEAT_DIMS,
    dyn_vars_count=DYN_VARS_COUNT,
):
    torch = globals().get("torch")
    X = np.load(X_path, mmap_mode="r")
    n_samples = len(X)
    split_dyn = dyn_vars_count * window_size
    log_mask = np.zeros(split_dyn, dtype=bool)
    pm10_dyn_index = dyn_vars_count - 1
    for t in range(window_size):
        for i in [2, 4, 9, pm10_dyn_index]:
            log_mask[t * dyn_vars_count + i] = True

    all_logits = []
    use_cuda = device.type == "cuda"
    non_blocking = use_cuda
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
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
        if extra.shape[1] < extra_feat_dims:
            extra = np.pad(
                extra,
                ((0, 0), (0, extra_feat_dims - extra.shape[1])),
                mode="constant",
                constant_values=0,
            )
        elif extra.shape[1] > extra_feat_dims:
            extra = extra[:, :extra_feat_dims]

        final = np.concatenate([np.clip(feats, -10, 10), veg, np.clip(extra, -10, 10)], axis=1)
        final = np.nan_to_num(final, nan=0.0)

        x = torch.from_numpy(final).float().to(device, non_blocking=non_blocking)
        with torch.inference_mode():
            fine, _, _ = model(x)
        all_logits.append(fine.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs_3c, probs_cum = cumulative_to_probs(logits)
    return probs_3c, probs_cum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=os.path.join(BASE, "ml_dataset_s2_tianji_12h_pm10_monthtail_2"))
    parser.add_argument("--ckpt-dir", default=os.path.join(BASE, "checkpoints"))
    parser.add_argument("--shp-path", default="/public/home/putianshu/中华人民共和国/中华人民共和国.shp")
    parser.add_argument("--fog-th", type=float, default=0.50)
    parser.add_argument("--low-vis-th", type=float, default=0.50)
    parser.add_argument("--ifs-nc", default=os.path.join(BASE, "VIS_IDW_KDTree_20250101_20251231.nc"))
    parser.add_argument("--event-top-k", type=int, default=3)
    parser.add_argument("--event-window-hours", type=int, default=3)
    parser.add_argument("--event-min-fog-stations", type=int, default=80)
    parser.add_argument("--event-min-regions", type=int, default=3)
    parser.add_argument("--event-min-lon-span", type=float, default=10.0)
    parser.add_argument("--event-min-lat-span", type=float, default=4.0)
    parser.add_argument("--event-gap-hours", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--extra-feat-dims", type=int, default=EXTRA_FEAT_DIMS)
    parser.add_argument("--dyn-vars-count", type=int, default=DYN_VARS_COUNT)
    args = parser.parse_args()

    try:
        import torch
        globals()["torch"] = torch
        if args.cpu or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except (ImportError, OSError):
        raise

    output_dir = args.output_dir or os.path.join(BASE, f"paper_eval_results_{args.exp_id}_ordinal")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.exp_id}_S2_PhaseB_best_score.pt")
    scaler_path = os.path.join(args.ckpt_dir, "robust_scaler_w12_dyn26_s2.pkl")
    ordinal_th_path = os.path.join(args.ckpt_dir, f"{args.exp_id}_ordinal_thresholds.pt")

    from PMST_net_test_13_s2_pm10_ordinal import OrdinalDualStreamPMSTNet

    print("=" * 60)
    print("Paper Evaluation Pipeline (Ordinal)")
    print("=" * 60)
    print(f"  Exp ID  : {args.exp_id}")
    print(f"  Output  : {output_dir}")
    print(f"  Device  : {device}")

    model = OrdinalDualStreamPMSTNet(
        window_size=12,
        hidden_dim=512,
        dropout=0.3,
        extra_feat_dim=args.extra_feat_dims,
        dyn_vars_count=args.dyn_vars_count,
        num_classes=2,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[1] Model loaded (DataParallel on {torch.cuda.device_count()} devices).")
    else:
        print("[1] Model loaded.")

    batch_size = args.batch_size or (8192 if device.type == "cuda" else 1024)
    print(f"     Inference batch_size={batch_size}")

    X_path, y_cls, y_raw, meta, scaler = load_test_data(args.data_dir, scaler_path, window_size=12)
    meta = enrich_meta_forecast_init(meta)
    print(f"[2] Data loaded: {len(y_cls)} samples, {meta['station_id'].nunique()} stations.")

    probs, probs_cum = run_inference_ordinal(
        X_path,
        scaler,
        model,
        device,
        batch_size=batch_size,
        window_size=12,
        extra_feat_dims=args.extra_feat_dims,
        dyn_vars_count=args.dyn_vars_count,
    )

    fog_th = args.fog_th
    low_vis_th = args.low_vis_th
    if os.path.exists(ordinal_th_path):
        try:
            cal = torch.load(ordinal_th_path, map_location="cpu")
            fog_th = float(cal.get("fog_th", fog_th))
            low_vis_th = float(cal.get("low_vis_th", low_vis_th))
            print(f"  [Calibration] Loaded ordinal thresholds: fog_th={fog_th:.4f}, low_vis_th={low_vis_th:.4f}")
        except Exception as e:
            print(f"  [WARN] Could not load ordinal thresholds: {e}")

    preds = pred_from_ordinal_thresholds(probs_cum, fog_th, low_vis_th)
    print("[3] Inference done.")

    report = compute_rare_event_report(probs, y_cls, fog_th, low_vis_th, pred=preds)
    print("[4] Rare-event metrics:")
    for k, v in report.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}")

    with open(os.path.join(output_dir, "rare_event_report.txt"), "w") as f:
        f.write(classification_report(y_cls, preds, target_names=["Fog", "Mist", "Clear"]))
        f.write("\n\nRare-event metrics:\n")
        for k, v in report.items():
            if isinstance(v, (int, float)):
                f.write(f"  {k}: {v:.4f}\n")

    sta_df = aggregate_station_metrics(meta, y_cls, preds)
    sta_df.to_csv(os.path.join(output_dir, "station_metrics.csv"), index=False)
    print(f"[5] Station metrics saved ({len(sta_df)} stations).")

    per_sample_path = os.path.join(output_dir, "per_sample_eval.csv")
    eval_df = export_per_sample_table(meta, y_cls, y_raw, preds, probs, per_sample_path)
    eval_df["p_low_vis"] = probs_cum[:, 1]
    eval_df["p_fog_cum"] = probs_cum[:, 0]
    eval_df.to_csv(per_sample_path, index=False, float_format="%.6f")
    print(f"[5b] Per-sample table saved: {per_sample_path} ({len(eval_df)} rows).")
    eval_df = derive_scenario_columns(eval_df)
    build_confusion_summaries_and_bottleneck_table(eval_df, output_dir)

    setup_paper_style()
    class_names = ["Fog", "Mist", "Clear"]
    plot_confusion_matrix_normalized(
        y_cls,
        preds,
        class_names,
        os.path.join(output_dir, "fig3_confusion_matrix.png"),
    )
    plot_per_class_prf1(report, os.path.join(output_dir, "fig3_prf1_bars.png"))
    plot_pr_curves(probs, y_cls, class_names, os.path.join(output_dir, "fig4_pr_curves.png"))
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

    meta_full = meta.copy()
    meta_full["y_true"] = y_cls
    meta_full["pred"] = preds
    meta_full["p_fog"] = probs[:, 0]
    meta_full["p_mist"] = probs[:, 1]
    scenario_results = plot_scenario_bars(
        meta_full,
        os.path.join(output_dir, "fig7_scenario_robustness.png"),
        fog_th=fog_th,
        mist_th=low_vis_th,
        threshold_rule="default",
    )
    save_forecast_init_metrics_table(
        scenario_results, os.path.join(output_dir, "metrics_by_forecast_init.csv")
    )
    plot_forecast_init_comparison(
        scenario_results, os.path.join(output_dir, "fig7b_forecast_init_comparison.png")
    )
    meta_for_hour = derive_scenario_columns(meta_full)
    save_metrics_by_valid_hour(
        y_cls, preds, meta_for_hour, os.path.join(output_dir, "metrics_by_valid_hour.csv")
    )

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

    with open(os.path.join(output_dir, "ordinal_thresholds_used.txt"), "w") as f:
        f.write(f"fog_th={fog_th:.6f}\n")
        f.write(f"low_vis_th={low_vis_th:.6f}\n")

    print()
    print("=" * 60)
    print("Evaluation complete.")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
