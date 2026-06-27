#!/usr/bin/env python3
"""Decompose Static-RNN Clear false alarms by visibility and meteorological regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_FEATURES = "RH2M,DPD,WSPD10,PM10_ugm3,PM25_ugm3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-sample-csv", required=True)
    p.add_argument("--x-npy", default="")
    p.add_argument("--dataset-config", default="")
    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--quantile-bins", type=int, default=5)
    p.add_argument("--narrow-soft-trigger", type=float, default=0.35)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def split_tokens(value: str) -> List[str]:
    return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]


def feature_order(config: Dict[str, object]) -> List[str]:
    candidates = [
        config.get("dynamic_feature_order"),
        config.get("dyn_vars"),
        (config.get("layout") or {}).get("dynamic_feature_order") if isinstance(config.get("layout"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return list(value)
    raise KeyError("dataset config has no dynamic_feature_order list; refusing dyn-width inference")


def add_last_step_features(df: pd.DataFrame, x_path: Path, config_path: Path, window_size: int, requested: Iterable[str]) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    order = feature_order(config)
    x = np.load(x_path, mmap_mode="r")
    if len(x) < len(df):
        raise ValueError(f"X rows={len(x)} smaller than per-sample rows={len(df)}")
    expected_dynamic = int(window_size) * len(order)
    if x.shape[1] < expected_dynamic:
        raise ValueError(f"X width={x.shape[1]} cannot contain {window_size}x{len(order)} dynamic values")
    out = df.copy()
    aliases = {name.lower(): name for name in order}
    for requested_name in requested:
        actual = aliases.get(requested_name.lower())
        if actual is None:
            print(f"[skip] dynamic feature unavailable: {requested_name}")
            continue
        idx = (int(window_size) - 1) * len(order) + order.index(actual)
        out[requested_name] = np.asarray(x[: len(out), idx], dtype=np.float64)
    return out


def season_from_month(month: pd.Series) -> pd.Series:
    m = pd.to_numeric(month, errors="coerce")
    return pd.Series(
        np.select(
            [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
            ["DJF", "MAM", "JJA", "SON"],
            default="unknown",
        ),
        index=month.index,
    )


def false_alarm_rows(df: pd.DataFrame, group_name: str, group: pd.Series) -> List[Dict[str, object]]:
    work = df.copy()
    work["__group"] = group.astype(str)
    rows: List[Dict[str, object]] = []
    for value, sub in work.groupby("__group", dropna=False):
        clear = sub["y_true"] == 2
        n_clear = int(clear.sum())
        fog_fp = int(np.sum(clear & (sub["pmst_pred"] == 0)))
        mist_fp = int(np.sum(clear & (sub["pmst_pred"] == 1)))
        rows.append(
            {
                "group_name": group_name,
                "group_value": value,
                "n": int(len(sub)),
                "clear_support": n_clear,
                "clear_to_ultra_fp": fog_fp,
                "clear_to_moderate_fp": mist_fp,
                "clear_to_lowvis_fp": fog_fp + mist_fp,
                "clear_to_ultra_fpr": fog_fp / max(n_clear, 1),
                "clear_to_moderate_fpr": mist_fp / max(n_clear, 1),
                "clear_to_lowvis_fpr": (fog_fp + mist_fp) / max(n_clear, 1),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source = Path(args.per_sample_csv)
    df = pd.read_csv(source)
    required = {"y_true", "vis_raw_m", "pmst_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Per-sample CSV is missing columns: {sorted(missing)}")
    requested = split_tokens(args.features)
    if args.x_npy or args.dataset_config:
        if not (args.x_npy and args.dataset_config):
            raise ValueError("--x-npy and --dataset-config must be provided together")
        df = add_last_step_features(df, Path(args.x_npy), Path(args.dataset_config), args.window_size, requested)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df["visibility_band"] = pd.cut(
        pd.to_numeric(df["vis_raw_m"], errors="coerce"),
        bins=[-np.inf, 500.0, 1000.0, 1200.0, 3000.0, np.inf],
        labels=["<500", "500-1000", "1000-1200", "1200-3000", ">3000"],
        right=False,
    )
    if "month" not in df:
        for col in ("time", "time_utc", "time_analysis"):
            if col in df:
                df["month"] = pd.to_datetime(df[col], errors="coerce").dt.month
                break
    if "month" in df:
        df["season"] = season_from_month(df["month"])

    group_rows: List[Dict[str, object]] = []
    group_rows.extend(false_alarm_rows(df, "visibility_band", df["visibility_band"]))
    for col in ("season", "hour", "hour_utc", "region", "lead_hour"):
        if col in df:
            group_rows.extend(false_alarm_rows(df, col, df[col]))
    for feature in requested:
        if feature not in df:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        try:
            bins = pd.qcut(values, q=max(2, args.quantile_bins), duplicates="drop")
        except ValueError:
            continue
        group_rows.extend(false_alarm_rows(df, f"{feature}_quantile", bins))

    groups = pd.DataFrame(group_rows)
    groups_path = out_dir / "false_alarm_group_metrics.csv"
    groups.to_csv(groups_path, index=False, float_format="%.8f")

    clear_mist_fp = (df["y_true"] == 2) & (df["pmst_pred"] == 1)
    near_boundary = pd.to_numeric(df["vis_raw_m"], errors="coerce").between(1000.0, 1200.0, inclusive="left")
    total_clear_mist_fp = int(clear_mist_fp.sum())
    near_clear_mist_fp = int((clear_mist_fp & near_boundary).sum())
    share = near_clear_mist_fp / max(total_clear_mist_fp, 1)

    cm = np.zeros((3, 3), dtype=np.int64)
    truth = pd.to_numeric(df["y_true"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    pred = pd.to_numeric(df["pmst_pred"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid = (truth >= 0) & (truth <= 2) & (pred >= 0) & (pred <= 2)
    np.add.at(cm, (truth[valid], pred[valid]), 1)
    cm_rows = [
        {"true_class": i, "pred_class": j, "count": int(cm[i, j])}
        for i in range(3)
        for j in range(3)
    ]
    cm_path = out_dir / "false_alarm_confusion_counts.csv"
    pd.DataFrame(cm_rows).to_csv(cm_path, index=False)

    decision = {
        "source": str(source),
        "clear_to_moderate_fp": total_clear_mist_fp,
        "clear_to_moderate_fp_1000_1200m": near_clear_mist_fp,
        "share_1000_1200m": share,
        "narrow_soft_trigger": args.narrow_soft_trigger,
        "enable_p4_narrow_soft": bool(share >= args.narrow_soft_trigger),
        "rule": "enable P4 only when share_1000_1200m >= narrow_soft_trigger",
        "outputs": {"groups": str(groups_path), "confusion": str(cm_path)},
    }
    decision_path = out_dir / "narrow_soft_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"[table] {groups_path}")
    print(f"[table] {cm_path}")
    print(f"[json] {decision_path}")


if __name__ == "__main__":
    main()
