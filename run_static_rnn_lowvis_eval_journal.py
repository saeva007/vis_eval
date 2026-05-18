#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Static-MLP + RNN low-visibility checkpoints with journal figures.

This script is the evaluation companion for
``train_static_rnn_lowvis.py``. It reuses the paper-evaluation metric and
figure helpers from ``run_paper_eval_pm10_pm25_journal.py`` while loading the
compact Static-MLP + GRU/LSTM checkpoints produced in the training repository.

Typical remote usage
--------------------
Main model:
  python paper_eval/run_static_rnn_lowvis_eval_journal.py \
    --mode main \
    --main_run_id exp_113669104_static_mlp_gru_main

Matrix ablations:
  python paper_eval/run_static_rnn_lowvis_eval_journal.py \
    --mode matrix \
    --matrix_run_prefix exp_static_rnn_matrix_from_113669104

Class definitions are identical to the project guide:
  0: 0 <= visibility < 500 m
  1: 500 <= visibility < 1000 m
  2: visibility >= 1000 m
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
VIS_EVAL_DIR = SCRIPT_PATH.parent

for _p in (str(VIS_EVAL_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from paper_eval_config import DEFAULT_CONFIG_NAME, apply_paper_eval_config
    from metrics_core import pred_from_thresholds_mutual
    from run_paper_eval_pm10_pm25_journal import (
        Manifest,
        TIME_OF_DAY_LOCAL_ORDER,
        REGION_DEFS,
        add_scenario_columns,
        aggregate_station_metrics,
        build_scenario_metrics,
        classification_metrics,
        load_main_data,
        plot_csi_recall_pmst_vs_ifs,
        plot_confusion_pmst_vs_ifs,
        plot_region_detail,
        plot_scenario_split,
        plot_station_metric_map,
        plot_time_of_day_detail,
        read_shapefile,
        save_fig_pair,
        setup_journal_style,
        write_report,
        export_per_sample,
    )
except Exception as exc:  # pragma: no cover - import failure must be explicit remotely.
    raise RuntimeError(f"Cannot import journal evaluation helpers from {VIS_EVAL_DIR}") from exc


DEFAULT_BASE = Path("/public/home/putianshu/vis_mlp")
DEFAULT_TRAIN_DIR = DEFAULT_BASE / "train"
DEFAULT_DATA_DIR = "ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2"
DEFAULT_OUT_DIR = "static_rnn_eval_results"
DEFAULT_STAGE_TAG = "S2_PhaseB"


@dataclass(frozen=True)
class VariantSpec:
    exp_id: int
    name: str
    encoder: str = "gru"
    hidden_dim: int = 256
    static_hidden_dim: int = 96
    fe_hidden_dim: int = 128
    fusion_hidden_dim: int = 256
    veg_emb_dim: int = 16
    rnn_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: str = "mean"
    use_fe: bool = True
    use_pm: bool = True


VARIANTS: Dict[int, VariantSpec] = {
    0: VariantSpec(0, "static_mlp_gru_main"),
    1: VariantSpec(1, "static_mlp_lstm", encoder="lstm"),
    2: VariantSpec(2, "static_mlp_gru_no_fe", use_fe=False),
    3: VariantSpec(3, "static_mlp_gru_no_pm", use_pm=False),
    4: VariantSpec(4, "static_mlp_gru_aux_reg"),
    5: VariantSpec(5, "static_mlp_gru_attention", pooling="attention"),
    6: VariantSpec(6, "static_mlp_bigru", hidden_dim=192, bidirectional=True),
    7: VariantSpec(7, "static_mlp_gru_csi_select"),
}
DEFAULT_MATRIX_IDS = [1, 2, 3, 4, 5, 6, 7]


@dataclass
class EvalTarget:
    label: str
    run_id: str
    checkpoint: Path
    variant: VariantSpec
    scaler_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Static-MLP + RNN checkpoints and draw journal figures.")
    p.add_argument("--config_json", default=os.environ.get("PAPER_EVAL_CONFIG", str(VIS_EVAL_DIR / DEFAULT_CONFIG_NAME)))
    p.add_argument("--mode", choices=["main", "matrix", "single"], default="main")
    p.add_argument("--base", default=str(DEFAULT_BASE))
    p.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--stage_tag", default=DEFAULT_STAGE_TAG)

    p.add_argument("--main_run_id", default="")
    p.add_argument("--main_ckpt", default="")
    p.add_argument("--matrix_run_prefix", default="")
    p.add_argument("--matrix_experiments", default="1:2:3:4:5:6:7")
    p.add_argument("--ckpt_path", default="", help="Single-checkpoint path for --mode single.")
    p.add_argument("--run_id", default="", help="Run id for --mode single; defaults to checkpoint stem.")
    p.add_argument("--variant_id", type=int, default=0, help="Variant id for --mode single.")

    p.add_argument("--threshold_source", choices=["checkpoint", "cli", "argmax"], default="checkpoint")
    p.add_argument("--fog_th", type=float, default=0.5)
    p.add_argument("--mist_th", type=float, default=0.5)
    p.add_argument("--plots", choices=["none", "core", "all"], default="core")
    p.add_argument("--shp_path", default="", help="Optional China boundary shapefile for station maps when --plots all.")
    p.add_argument("--allow_missing", action="store_true", help="Skip missing matrix checkpoints instead of failing.")
    return p.parse_args()


def as_abs_under(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 1000):
        cand = path.with_name(f"{path.name}_r{idx:02d}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not allocate non-overwriting output directory near {path}")


def parse_id_list(value: str) -> List[int]:
    value = (value or "").replace(",", ":").replace(" ", ":")
    out = []
    for token in value.split(":"):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out or list(DEFAULT_MATRIX_IDS)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def infer_layout_from_x(x_path: Path, window_size: int):
    shape = np.load(x_path, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{x_path} must be 2D [N,D], got {shape}")
    rest = int(shape[1]) - 6
    for dyn in (27, 26, 25, 24):
        fe = rest - dyn * int(window_size)
        if 20 <= fe <= 64:
            return dyn, fe
    raise ValueError(f"Cannot infer static-RNN layout from {x_path}: shape={shape}, window={window_size}")


def load_static_rnn_module(train_dir: Path):
    model_py = train_dir / "train_static_rnn_lowvis.py"
    if not model_py.exists():
        raise FileNotFoundError(f"Missing static-RNN training script: {model_py}")
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    spec = importlib.util.spec_from_file_location("static_rnn_lowvis_for_eval", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint_payload(path: Path, device):
    import torch

    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        meta = payload.get("metadata", {})
    elif isinstance(payload, dict):
        state = payload
        meta = {}
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(payload)} from {path}")
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    return state, meta if isinstance(meta, dict) else {}


def resolve_scaler_path(base: Path, ckpt_dir: Path, run_id: str, layout, use_pm: bool, explicit: Optional[Path]) -> Path:
    if explicit:
        path = explicit if explicit.is_absolute() else base / explicit
        if not path.exists():
            raise FileNotFoundError(f"Explicit scaler path not found: {path}")
        return path
    tag = "pm" if use_pm else "nopm"
    exact = ckpt_dir / f"robust_scaler_{run_id}_s2_w{layout.window_size}_dyn{layout.dyn_vars}_{tag}.pkl"
    if exact.exists():
        return exact
    candidates = sorted(ckpt_dir.glob(f"robust_scaler_{run_id}_s2_w{layout.window_size}_dyn{layout.dyn_vars}_*.pkl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot find scaler for run_id={run_id} in {ckpt_dir}")


def prepare_static_rows(rows: np.ndarray, scaler, layout, mod, spec: VariantSpec) -> np.ndarray:
    log_mask = mod.build_dyn_log_mask(layout)
    core = rows[:, : layout.core_dim].astype(np.float32, copy=True)
    core = mod.apply_core_transform(core, layout, spec.use_pm, log_mask)
    if scaler is not None:
        if len(scaler.center_) != core.shape[1]:
            raise ValueError(f"Scaler dim={len(scaler.center_)} but core dim={core.shape[1]}")
        core = (core - scaler.center_) / (scaler.scale_ + 1e-6)
    core = np.clip(core, -10.0, 10.0)
    veg = rows[:, layout.split_dyn + 5 : layout.split_dyn + 6].astype(np.float32, copy=False)
    parts = [core, veg]
    if spec.use_fe:
        fe = rows[:, layout.split_dyn + 6 : layout.split_dyn + 6 + layout.fe_dim].astype(np.float32, copy=True)
        parts.append(np.clip(fe, -10.0, 10.0))
    final = np.concatenate(parts, axis=1)
    return np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def run_static_inference(
    x_path: Path,
    scaler,
    model,
    device,
    batch_size: int,
    layout,
    mod,
    spec: VariantSpec,
    limit_samples: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    X = np.load(x_path, mmap_mode="r")
    n = len(X) if not limit_samples or limit_samples <= 0 else min(int(limit_samples), len(X))
    out = []
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = np.asarray(X[start:end], dtype=np.float32)
        final = prepare_static_rows(rows, scaler, layout, mod, spec)
        bx = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits, _ = model(bx)
            probs = F.softmax(logits, dim=1)
        out.append(probs.detach().cpu().numpy())
        if start == 0 or end == n or (start // max(batch_size, 1)) % 20 == 0:
            print(f"  [inference:{spec.name}] {end}/{n}", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def threshold_from_checkpoint(meta: Dict[str, object]) -> Optional[Tuple[float, float]]:
    th = meta.get("thresholds", {}) if isinstance(meta, dict) else {}
    if isinstance(th, dict) and "fog" in th and "mist" in th:
        return float(th["fog"]), float(th["mist"])
    return None


def predict_from_probs(args: argparse.Namespace, probs: np.ndarray, meta: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
    if args.threshold_source == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64), {"threshold_source": "argmax"}
    ckpt_th = threshold_from_checkpoint(meta)
    if args.threshold_source == "checkpoint" and ckpt_th is not None:
        fog_th, mist_th = ckpt_th
        source = "checkpoint"
    else:
        fog_th, mist_th = float(args.fog_th), float(args.mist_th)
        source = "cli_fallback" if args.threshold_source == "checkpoint" else "cli"
    pred = pred_from_thresholds_mutual(probs, fog_th, mist_th).astype(np.int64)
    return pred, {"threshold_source": source, "threshold_rule": "mutual", "fog_th": fog_th, "mist_th": mist_th}


def make_model(mod, layout, spec: VariantSpec, device):
    return mod.StaticRNNLowVisNet(
        layout=layout,
        encoder=spec.encoder,
        hidden_dim=spec.hidden_dim,
        static_hidden_dim=spec.static_hidden_dim,
        fe_hidden_dim=spec.fe_hidden_dim,
        fusion_hidden_dim=spec.fusion_hidden_dim,
        veg_emb_dim=spec.veg_emb_dim,
        rnn_layers=spec.rnn_layers,
        dropout=spec.dropout,
        bidirectional=spec.bidirectional,
        pooling=spec.pooling,
        use_fe=spec.use_fe,
    ).to(device)


def find_main_checkpoint(ckpt_dir: Path, stage_tag: str, main_run_id: str, main_ckpt: str, base: Path) -> Tuple[str, Path]:
    if main_ckpt:
        path = as_abs_under(base, main_ckpt)
        return (main_run_id or path.name.replace(f"_{stage_tag}_best_score.pt", "")), path
    if main_run_id:
        return main_run_id, ckpt_dir / f"{main_run_id}_{stage_tag}_best_score.pt"
    candidates = sorted(ckpt_dir.glob(f"*static_mlp_gru_main_{stage_tag}_best_score.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No main static-RNN checkpoint found; pass --main_run_id or --main_ckpt.")
    path = candidates[0]
    return path.name.replace(f"_{stage_tag}_best_score.pt", ""), path


def build_targets(args: argparse.Namespace, base: Path, ckpt_dir: Path) -> List[EvalTarget]:
    targets: List[EvalTarget] = []
    if args.mode == "main":
        run_id, path = find_main_checkpoint(ckpt_dir, args.stage_tag, args.main_run_id, args.main_ckpt, base)
        targets.append(EvalTarget("main", run_id, path, VARIANTS[0]))
    elif args.mode == "single":
        if not args.ckpt_path:
            raise ValueError("--mode single requires --ckpt_path")
        path = as_abs_under(base, args.ckpt_path)
        run_id = args.run_id or path.name.replace(f"_{args.stage_tag}_best_score.pt", "")
        targets.append(EvalTarget(VARIANTS[args.variant_id].name, run_id, path, VARIANTS[args.variant_id]))
    else:
        if not args.matrix_run_prefix:
            raise ValueError("--mode matrix requires --matrix_run_prefix")
        for exp_id in parse_id_list(args.matrix_experiments):
            if exp_id == 0:
                raise ValueError("Matrix evaluation excludes experiment 0; evaluate the main model with --mode main.")
            spec = VARIANTS[exp_id]
            run_id = f"{args.matrix_run_prefix}_{exp_id}_{spec.name}"
            path = ckpt_dir / f"{run_id}_{args.stage_tag}_best_score.pt"
            if not path.exists() and args.allow_missing:
                print(f"[skip] missing checkpoint: {path}", flush=True)
                continue
            targets.append(EvalTarget(spec.name, run_id, path, spec))
    for target in targets:
        if not target.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {target.label}: {target.checkpoint}")
    return targets


def plot_summary_bar(summary: pd.DataFrame, out_dir: Path, manifest: Manifest, sources: Sequence[str]) -> None:
    if summary.empty:
        return
    setup_journal_style()
    metrics = [
        ("Fog_CSI", "Fog CSI"),
        ("Fog_R", "Fog recall"),
        ("Mist_CSI", "Mist CSI"),
        ("Mist_R", "Mist recall"),
        ("low_vis_precision", "Low-vis precision"),
        ("false_positive_rate", "False alarm rate"),
    ]
    labels = summary["label"].astype(str).tolist()
    x = np.arange(len(labels))
    width = min(0.12, 0.78 / len(metrics))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.82 * len(labels)), 4.6))
    for i, (key, label) in enumerate(metrics):
        vals = summary[key].astype(float).to_numpy()
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width * 0.96, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Static-RNN checkpoint comparison on the S2 test set")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig_pair(fig, out_dir, "fig_static_rnn_model_matrix_summary", manifest, sources, notes="Static-RNN main/ablation checkpoint comparison.")


def evaluate_target(
    args: argparse.Namespace,
    target: EvalTarget,
    base: Path,
    train_dir: Path,
    data_dir: Path,
    out_root: Path,
    mod,
    device,
) -> Dict[str, object]:
    import joblib

    print(f"\n=== Evaluating {target.label}: {target.run_id} ===", flush=True)
    x_path, y_cls, y_raw, meta = load_main_data(data_dir, args.limit_samples)
    dyn, fe = infer_layout_from_x(x_path, args.window_size)
    layout = mod.Layout(window_size=args.window_size, dyn_vars=dyn, fe_dim=fe)
    ckpt_dir = as_abs_under(base, args.ckpt_dir)
    scaler_path = resolve_scaler_path(base, ckpt_dir, target.run_id, layout, target.variant.use_pm, target.scaler_path)
    scaler = joblib.load(scaler_path)

    state, ckpt_meta = load_checkpoint_payload(target.checkpoint, device)
    model = make_model(mod, layout, target.variant, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing keys: {len(missing)} first={missing[:5]}", flush=True)
    if unexpected:
        print(f"  [ckpt] unexpected keys: {len(unexpected)} first={unexpected[:5]}", flush=True)

    probs = run_static_inference(
        x_path,
        scaler,
        model,
        device,
        args.batch_size,
        layout,
        mod,
        target.variant,
        args.limit_samples,
    )
    pred, decision_meta = predict_from_probs(args, probs, ckpt_meta)
    metrics = classification_metrics(y_cls, pred, probs=probs)

    out_dir = unique_dir(out_root / target.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir)
    sources = [str(x_path), str(data_dir / "y_test.npy"), str(data_dir / "meta_test.csv"), str(target.checkpoint), str(scaler_path)]

    pd.DataFrame([{"source": target.label, **decision_meta, **metrics}]).to_csv(out_dir / "overall_metrics.csv", index=False)
    np.save(out_dir / "probs.npy", probs.astype(np.float32))
    eval_df = export_per_sample(out_dir / "per_sample_eval.csv", meta, y_cls, y_raw, pred, probs)
    write_report(out_dir / "rare_event_report.txt", y_cls, pred, metrics)

    scenario_df = build_scenario_metrics(eval_df)
    scenario_df.to_csv(out_dir / "scenario_metrics.csv", index=False)
    station_df = aggregate_station_metrics(eval_df, "pmst_pred")
    station_df.to_csv(out_dir / "station_metrics.csv", index=False)

    if args.plots != "none":
        plot_confusion_pmst_vs_ifs(y_cls, pred, None, None, out_dir, manifest, sources)
        plot_csi_recall_pmst_vs_ifs(metrics, None, out_dir, manifest, sources, n=len(y_cls), matched_ifs=None)
        for split, order in (
            ("time_of_day", TIME_OF_DAY_LOCAL_ORDER),
            ("season", ["DJF", "MAM", "JJA", "SON"]),
            ("region", [r[0] for r in REGION_DEFS] + ["Other"]),
        ):
            plot_scenario_split(scenario_df, split, order, out_dir, manifest, [str(out_dir / "scenario_metrics.csv")])
        plot_time_of_day_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")])
        plot_region_detail(eval_df, out_dir, manifest, [str(out_dir / "per_sample_eval.csv")])
        if args.plots == "all":
            shp = read_shapefile(args.shp_path) if args.shp_path else None
            plot_station_metric_map(
                station_df,
                "fog_recall",
                "n_fog",
                5,
                "Station Fog Recall",
                "fig8_station_fog_recall",
                out_dir,
                manifest,
                [str(out_dir / "station_metrics.csv")],
                shp_gdf=shp,
                cmap="cividis",
                vmin=0,
                vmax=1,
            )
            plot_station_metric_map(
                station_df,
                "mist_recall",
                "n_mist",
                5,
                "Station Mist Recall",
                "fig8_station_mist_recall",
                out_dir,
                manifest,
                [str(out_dir / "station_metrics.csv")],
                shp_gdf=shp,
                cmap="cividis",
                vmin=0,
                vmax=1,
            )
            plot_station_metric_map(
                station_df,
                "fpr_fog",
                "n_clear",
                20,
                "Station Low-Visibility False Positive Rate",
                "fig8_station_fpr",
                out_dir,
                manifest,
                [str(out_dir / "station_metrics.csv")],
                shp_gdf=shp,
                cmap="magma_r",
                vmin=0,
                vmax=0.2,
            )
    manifest.write()

    run_config = {
        "target": target.label,
        "run_id": target.run_id,
        "checkpoint": str(target.checkpoint),
        "scaler": str(scaler_path),
        "variant": asdict(target.variant),
        "layout": asdict(layout),
        "decision": decision_meta,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "checkpoint_metadata": ckpt_meta,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    row = {"label": target.label, "run_id": target.run_id, "checkpoint": str(target.checkpoint), "out_dir": str(out_dir), **decision_meta, **metrics}
    return row


def main() -> None:
    args = parse_args()
    args = apply_paper_eval_config(args, "static_rnn_eval", default_dir=VIS_EVAL_DIR)

    base = Path(args.base).expanduser()
    train_dir = Path(args.train_dir).expanduser()
    data_dir = as_abs_under(base, args.data_dir)
    ckpt_dir = as_abs_under(base, args.ckpt_dir)
    out_root = as_abs_under(base, args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = build_targets(args, base, ckpt_dir)
    mod = load_static_rnn_module(train_dir)
    device = resolve_device(args.device)

    print("Static-RNN journal evaluation", flush=True)
    print(f"mode      : {args.mode}", flush=True)
    print(f"base      : {base}", flush=True)
    print(f"train_dir : {train_dir}", flush=True)
    print(f"data_dir  : {data_dir}", flush=True)
    print(f"out_root  : {out_root}", flush=True)
    print(f"device    : {device}", flush=True)
    print(f"targets   : {[t.run_id for t in targets]}", flush=True)

    rows = []
    for target in targets:
        rows.append(evaluate_target(args, target, base, train_dir, data_dir, out_root, mod, device))
    summary = pd.DataFrame(rows)
    summary_path = out_root / "static_rnn_eval_summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}", flush=True)
    if args.plots != "none" and len(summary) > 1:
        manifest = Manifest(out_root)
        plot_summary_bar(summary, out_root, manifest, [str(summary_path)])
        manifest.write()


if __name__ == "__main__":
    main()
