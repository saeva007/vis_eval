#!/usr/bin/env python3
"""Prepare a best-effort mean-softmax result for static-RNN journal plots.

The source-full best-effort evaluator writes per-sample probabilities on the
common member intersection. The static-RNN journal evaluator can reuse
probabilities from ``probs.npy``, but it still expects a data directory with
``X_test.npy``, ``y_test.npy`` and ``meta_test.csv`` in the same row order.

This helper aligns the best-effort per-sample table to the main test metadata,
validates observed labels/raw visibility, writes an aligned test-only data
directory, and writes a reuse-inference directory containing ``probs.npy``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample_csv", required=True, help="per_sample_tianji_t2nd_ifs_mean_softmax.csv from best-effort eval.")
    p.add_argument("--main_data_dir", required=True, help="Main static-RNN data dir used by paper_eval.")
    p.add_argument("--out_data_dir", required=True, help="Aligned test-only data dir to create.")
    p.add_argument("--out_reuse_dir", required=True, help="Directory to receive probs.npy and provenance JSON.")
    p.add_argument("--source_name", default="tianji_t2nd_ifs_mean_softmax")
    p.add_argument("--prob_cols", default="p_fog,p_mist,p_clear")
    p.add_argument("--time_col", default="time")
    p.add_argument("--station_col", default="station_id")
    p.add_argument("--main_time_shift_hours", type=float, default=0.0)
    p.add_argument("--sample_time_shift_hours", type=float, default=0.0)
    p.add_argument("--raw_vis_tolerance_m", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def reset_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def normalize_station(values: Iterable[object]) -> pd.Series:
    raw = pd.Series(values)
    text = raw.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    ok = numeric.notna()
    if ok.any():
        text.loc[ok] = numeric.loc[ok].astype(np.int64).astype(str)
    return text


def add_alignment_keys(
    frame: pd.DataFrame,
    *,
    time_col: str,
    station_col: str,
    time_shift_hours: float,
    row_name: str,
) -> pd.DataFrame:
    missing = [c for c in (time_col, station_col) if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing alignment column(s) {missing} in frame with columns={list(frame.columns)[:20]}")
    out = frame.copy()
    time = pd.to_datetime(out[time_col], errors="coerce")
    if float(time_shift_hours or 0.0) != 0.0:
        time = time + pd.to_timedelta(float(time_shift_hours), unit="h")
    if time.isna().any():
        raise ValueError(f"{time_col} contains {int(time.isna().sum())} unparseable timestamps.")
    out["__time_key"] = time.dt.floor("s").astype("datetime64[ns]")
    out["__station_key"] = normalize_station(out[station_col])
    out["__dup"] = out.groupby(["__time_key", "__station_key"]).cumcount()
    out[row_name] = np.arange(len(out), dtype=np.int64)
    return out


def visibility_to_class(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_raw = np.asarray(raw, dtype=np.float64).copy()
    finite = np.isfinite(y_raw)
    if finite.any() and float(np.nanmax(y_raw[finite])) < 100.0:
        y_raw *= 1000.0
    y_cls = np.full(y_raw.shape, 2, dtype=np.int64)
    y_cls[y_raw < 1000.0] = 1
    y_cls[y_raw < 500.0] = 0
    y_cls[~np.isfinite(y_raw)] = -1
    return y_cls, y_raw


def copy_optional_metadata(main_data_dir: Path, out_data_dir: Path) -> List[str]:
    copied: List[str] = []
    for name in ("dataset_build_config.json", "dataset_metadata.json"):
        src = main_data_dir / name
        if src.is_file():
            dst = out_data_dir / name
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def save_subset_npy(src_path: Path, dst_path: Path, rows: np.ndarray, chunk_rows: int = 200_000) -> None:
    src = np.load(src_path, mmap_mode="r")
    rows = np.asarray(rows, dtype=np.int64)
    shape = (len(rows),) + tuple(src.shape[1:])
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=shape)
    for start in range(0, len(rows), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(rows))
        dst[start:end] = src[rows[start:end]]
    dst.flush()


def main() -> None:
    args = parse_args()
    sample_csv = Path(args.sample_csv)
    main_data_dir = Path(args.main_data_dir)
    out_data_dir = Path(args.out_data_dir)
    out_reuse_dir = Path(args.out_reuse_dir)

    require_file(sample_csv, "best-effort per-sample CSV")
    require_file(main_data_dir / "X_test.npy", "main X_test.npy")
    require_file(main_data_dir / "y_test.npy", "main y_test.npy")
    require_file(main_data_dir / "meta_test.csv", "main meta_test.csv")

    reset_output_dir(out_data_dir, args.overwrite)
    reset_output_dir(out_reuse_dir, args.overwrite)

    sample = pd.read_csv(sample_csv)
    main_meta = pd.read_csv(main_data_dir / "meta_test.csv")
    sample = add_alignment_keys(
        sample,
        time_col=args.time_col,
        station_col=args.station_col,
        time_shift_hours=args.sample_time_shift_hours,
        row_name="__sample_row",
    )
    main_keyed = add_alignment_keys(
        main_meta,
        time_col=args.time_col,
        station_col=args.station_col,
        time_shift_hours=args.main_time_shift_hours,
        row_name="__main_row",
    )

    joined = sample.merge(
        main_keyed[["__time_key", "__station_key", "__dup", "__main_row"]],
        on=["__time_key", "__station_key", "__dup"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    missing = joined["__main_row"].isna()
    if missing.any():
        examples = joined.loc[missing, [args.time_col, args.station_col]].head(5).to_dict("records")
        raise RuntimeError(
            f"{int(missing.sum())} best-effort rows did not match main metadata. "
            f"Examples: {examples}"
        )
    main_rows = joined["__main_row"].to_numpy(dtype=np.int64)

    prob_cols = [c.strip() for c in str(args.prob_cols).split(",") if c.strip()]
    if len(prob_cols) != 3:
        raise ValueError("--prob_cols must name exactly three probability columns.")
    missing_prob = [c for c in prob_cols if c not in joined.columns]
    if missing_prob:
        raise KeyError(f"Missing probability column(s) in sample CSV: {missing_prob}")
    probs = joined[prob_cols].to_numpy(dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probs)) or np.any(row_sums <= 0):
        raise ValueError("Probability columns contain non-finite values or non-positive row sums.")
    probs = (probs / row_sums).astype(np.float32)

    y_main_raw_all = np.load(main_data_dir / "y_test.npy", mmap_mode="r")
    y_main_raw = np.asarray(y_main_raw_all[main_rows])
    y_main_cls, y_main_m = visibility_to_class(y_main_raw)
    if "y_cls" in joined.columns:
        y_sample_cls = joined["y_cls"].to_numpy(dtype=np.int64)
        bad = y_sample_cls != y_main_cls
        if bad.any():
            raise RuntimeError(f"Sample y_cls differs from main y_test-derived labels in {int(bad.sum())} rows.")
    if "vis_raw_m" in joined.columns:
        sample_raw = pd.to_numeric(joined["vis_raw_m"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(sample_raw) & np.isfinite(y_main_m)
        if finite.any():
            max_diff = float(np.max(np.abs(sample_raw[finite] - y_main_m[finite])))
            if max_diff > float(args.raw_vis_tolerance_m):
                raise RuntimeError(
                    "Sample vis_raw_m differs from main y_test raw visibility: "
                    f"max_diff={max_diff:.6g} m."
                )

    save_subset_npy(main_data_dir / "X_test.npy", out_data_dir / "X_test.npy", main_rows)
    np.save(out_data_dir / "y_test.npy", y_main_raw)
    main_meta.iloc[main_rows].reset_index(drop=True).to_csv(out_data_dir / "meta_test.csv", index=False)
    copied_meta = copy_optional_metadata(main_data_dir, out_data_dir)

    np.save(out_reuse_dir / "probs.npy", probs)
    joined.drop(columns=["__time_key", "__station_key", "__dup"], errors="ignore").to_csv(
        out_reuse_dir / "aligned_best_effort_sample.csv",
        index=False,
    )
    pd.DataFrame({"sample_row": joined["__sample_row"], "main_row": main_rows}).to_csv(
        out_reuse_dir / "alignment_rows.csv",
        index=False,
    )

    config = {
        "source_name": args.source_name,
        "sample_csv": str(sample_csv),
        "main_data_dir": str(main_data_dir),
        "out_data_dir": str(out_data_dir),
        "out_reuse_dir": str(out_reuse_dir),
        "rows": int(len(joined)),
        "main_rows_total": int(len(main_meta)),
        "prob_cols": prob_cols,
        "decision": "mean post-softmax probabilities, then argmax in downstream evaluator",
        "alignment_keys": ["time", "station_id", "duplicate_index_within_time_station"],
        "main_time_shift_hours": float(args.main_time_shift_hours or 0.0),
        "sample_time_shift_hours": float(args.sample_time_shift_hours or 0.0),
        "copied_metadata_files": copied_meta,
        "x_test_mode": "aligned_subset_copy",
    }
    with open(out_reuse_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(out_data_dir / "prepared_reuse_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[OK] prepared static-RNN reuse inputs", flush=True)
    print(f"  rows          : {len(joined)} / {len(main_meta)} main test rows", flush=True)
    print(f"  out_data_dir  : {out_data_dir}", flush=True)
    print(f"  out_reuse_dir : {out_reuse_dir}", flush=True)


if __name__ == "__main__":
    main()
