#!/usr/bin/env python3
"""Merge an existing 10-point sampling curve with newly evaluated 5-point gaps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


TARGETS = tuple(range(0, 51, 5))
LABEL_TO_PCT = {f"lowvis_share_{value:02d}": value for value in TARGETS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/public/home/putianshu/vis_mlp"))
    p.add_argument("--legacy-eval-dir", default="auto")
    p.add_argument("--fill-eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--figure-stem", default="fig_static_rnn_sampling_method_ablation")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def add_target_pct(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    out = df.copy()
    if "target_lowvis_share" in out.columns:
        pct = 100.0 * pd.to_numeric(out["target_lowvis_share"], errors="coerce")
    else:
        pct = pd.Series(np.nan, index=out.index, dtype=float)
    if "label" in out.columns:
        for index, label in out["label"].astype(str).items():
            if not np.isfinite(pct.loc[index]):
                pct.loc[index] = LABEL_TO_PCT.get(label, np.nan)
    if not np.isfinite(pct).all():
        raise ValueError(f"{source}: could not resolve target share for every row")
    out["target_lowvis_pct"] = np.rint(pct).astype(int)
    out["target_lowvis_share"] = out["target_lowvis_pct"] / 100.0
    return out


def curve_points(path: Path) -> Tuple[int, ...]:
    table = add_target_pct(pd.read_csv(path), path)
    return tuple(sorted(set(table["target_lowvis_pct"].astype(int))))


def discover_legacy(base: Path) -> Path:
    candidates: List[Path] = []
    patterns = (
        "paper_eval_results*sampling*/sampling_ablation_overall_metrics.csv",
        "static_rnn_sampling_eval_results*/sampling_ablation_overall_metrics.csv",
    )
    for pattern in patterns:
        candidates.extend(base.glob(pattern))
    valid = []
    for path in candidates:
        try:
            if curve_points(path) == (0, 10, 20, 30, 40, 50):
                valid.append(path)
        except (KeyError, TypeError, ValueError):
            continue
    if not valid:
        raise FileNotFoundError(
            "Could not auto-discover the completed 0/10/20/30/40/50 evaluation; pass --legacy-eval-dir."
        )
    return max(valid, key=lambda path: path.stat().st_mtime).parent


def resolve_legacy(base: Path, value: str) -> Path:
    if str(value).strip().lower() == "auto":
        return discover_legacy(base)
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def require_columns(df: pd.DataFrame, names: Iterable[str], source: Path) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def merge_overall(legacy_dir: Path, fill_dir: Path, out_dir: Path) -> pd.DataFrame:
    paths = [legacy_dir / "sampling_ablation_overall_metrics.csv", fill_dir / "sampling_ablation_overall_metrics.csv"]
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = add_target_pct(pd.read_csv(path), path)
        require_columns(frame, ["label", "low_vis_csi", "low_vis_recall", "low_vis_precision"], path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    duplicated = merged[merged.duplicated("target_lowvis_pct", keep=False)]
    if not duplicated.empty:
        raise ValueError(f"Duplicate target shares across legacy/fill evaluations: {sorted(duplicated['target_lowvis_pct'].unique())}")
    present = tuple(sorted(merged["target_lowvis_pct"].astype(int).tolist()))
    if present != TARGETS:
        raise ValueError(f"Merged curve is incomplete: expected {TARGETS}, found {present}")
    merged = merged.sort_values("target_lowvis_pct").reset_index(drop=True)
    merged["experiment_id_original"] = merged.get("experiment_id", np.nan)
    merged["experiment_id"] = merged["target_lowvis_pct"].astype(int)
    merged.to_csv(out_dir / "sampling_ablation_overall_metrics.csv", index=False, float_format="%.8f")
    return merged


def merge_optional_table(filename: str, legacy_dir: Path, fill_dir: Path, out_dir: Path) -> None:
    paths = [legacy_dir / filename, fill_dir / filename]
    if not all(path.is_file() for path in paths):
        return
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "label" in frame.columns:
            frame["target_lowvis_pct"] = frame["label"].astype(str).map(LABEL_TO_PCT)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "target_lowvis_pct" in merged.columns:
        merged = merged.sort_values(["target_lowvis_pct"] + (["class_id"] if "class_id" in merged.columns else []))
    merged.to_csv(out_dir / filename, index=False, float_format="%.8f")


def main() -> None:
    args = parse_args()
    base = args.base.expanduser().resolve()
    fill_dir = args.fill_eval_dir.expanduser()
    fill_dir = fill_dir if fill_dir.is_absolute() else base / fill_dir
    fill_dir = fill_dir.resolve()
    out_dir = args.out_dir.expanduser()
    out_dir = out_dir if out_dir.is_absolute() else base / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    legacy_dir = resolve_legacy(base, args.legacy_eval_dir).resolve()

    merged = merge_overall(legacy_dir, fill_dir, out_dir)
    merge_optional_table("sampling_ablation_per_class_metrics.csv", legacy_dir, fill_dir, out_dir)
    merge_optional_table("sampling_ablation_confusion_counts.csv", legacy_dir, fill_dir, out_dir)
    config = {
        "legacy_eval_dir": str(legacy_dir),
        "fill_eval_dir": str(fill_dir),
        "out_dir": str(out_dir),
        "target_percentages": list(TARGETS),
        "reused_percentages": [0, 10, 20, 30, 40, 50],
        "newly_trained_percentages": [5, 15, 25, 35, 45],
    }
    (out_dir / "sampling_curve_merge_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    plot_script = Path(__file__).resolve().with_name("plot_static_rnn_sampling_ablation.py")
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--eval_dir",
            str(out_dir),
            "--out_dir",
            str(out_dir),
            "--figure_stem",
            args.figure_stem,
            "--dpi",
            str(args.dpi),
        ],
        check=True,
    )
    print(f"legacy_eval_dir={legacy_dir}")
    print(f"fill_eval_dir={fill_dir}")
    print(f"merged_rows={len(merged)}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
