#!/usr/bin/env python3
"""Prepare a strictly aligned multi-seed probability mean for paper evaluation.

The member probabilities must come from one invocation of
``run_static_rnn_precision_candidate_eval.py`` (one output directory per
dataset).  Members are located from the evaluator's metrics table rather than
from guessed file ordering.  Samples are aligned by station, valid time, and a
within-key duplicate index before post-softmax probabilities are averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import run_static_rnn_lowvis_eval_journal as journal


PROBABILITY_COLUMNS = ("pmst_p_fog", "pmst_p_mist", "pmst_p_clear")
IDENTITY_TIME_COLUMNS = ("time", "time_utc", "time_analysis")
IDENTITY_STRING_COLUMNS = (
    "station_id",
    "time",
    "time_utc",
    "time_utc_original",
    "time_analysis",
    "init_time",
    "forecast_reference_time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Full-mode precision-loss TSV manifest.")
    parser.add_argument("--main-eval-dir", "--eval-dir", dest="main_eval_dir", required=True)
    parser.add_argument(
        "--forecast48-eval-dir",
        default="",
        help="Optional member-evaluation directory produced on the 48 h dataset.",
    )
    parser.add_argument("--candidate-id", default="p13")
    parser.add_argument("--expected-seeds", default="42,314,2718")
    parser.add_argument("--required-stage", choices=("full", "any"), default="full")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base", default="/public/home/putianshu/vis_mlp")
    parser.add_argument(
        "--data-48h-dir",
        default="ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime",
    )
    parser.add_argument("--ifs-48h-nc", default="IFS_VIS_0_48h_stations_2025_00_12.nc")
    parser.add_argument("--ifs-48h-var", default="VIS_ifs")
    parser.add_argument("--local-time-offset-hours", type=float, default=8.0)
    parser.add_argument("--meta-time-shift-hours", type=float, default=0.0)
    parser.add_argument("--probability-atol", type=float, default=2.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seed_list(value: str) -> List[str]:
    tokens = str(value or "").replace(":", ",").replace(" ", ",").split(",")
    seeds = [token.strip() for token in tokens if token.strip()]
    if not seeds:
        raise ValueError("--expected-seeds is empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate expected seeds: {seeds}")
    return seeds


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def canonical_station_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": "", "None": ""})
    )


def parse_mixed_datetime(values: pd.Series, name: str = "time") -> pd.Series:
    """Parse ISO, compact, and epoch timestamps without format-order inference.

    Large 48 h per-sample CSVs are appended from many forecast cycles.  Pandas
    can otherwise infer one format from the first chunk and coerce valid rows
    from later chunks to NaT.  Returned timestamps are timezone-naive UTC.
    """

    source = pd.Series(values, index=getattr(values, "index", None))
    raw = source.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns, UTC]")

    # Parse compact forecast-cycle encodings before considering epoch units.
    for length, fmt in ((8, "%Y%m%d"), (10, "%Y%m%d%H"), (12, "%Y%m%d%H%M"), (14, "%Y%m%d%H%M%S")):
        mask = parsed.isna() & raw.str.fullmatch(rf"\d{{{length}}}", na=False)
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(
                raw.loc[mask], format=fmt, errors="coerce", utc=True
            )

    remaining = parsed.isna() & raw.notna() & ~raw.isin(("", "NaT", "nan", "None", "<NA>"))
    if remaining.any():
        # ``format='mixed'`` is available in modern pandas.  The second pass
        # preserves compatibility with older cluster pandas versions.
        try:
            generic = pd.to_datetime(
                raw.loc[remaining], format="mixed", errors="coerce", utc=True
            )
        except (TypeError, ValueError):
            generic = pd.to_datetime(raw.loc[remaining], errors="coerce", utc=True)
        fallback_mask = generic.isna()
        if fallback_mask.any():
            generic.loc[fallback_mask] = pd.to_datetime(
                raw.loc[generic.index[fallback_mask]], errors="coerce", utc=True
            )
        parsed.loc[remaining] = generic

    # Recover serialized Unix epochs only for values still unparsed.  Compact
    # YYYYMMDD[HH[MM[SS]]] values were already consumed above.
    numeric = pd.to_numeric(raw, errors="coerce")
    epoch_specs = (
        ("ns", 1.0e17, np.inf),
        ("us", 1.0e14, 1.0e17),
        ("ms", 1.0e11, 1.0e14),
        ("s", 1.0e8, 1.0e11),
    )
    for unit, lower, upper in epoch_specs:
        magnitude = numeric.abs()
        mask = parsed.isna() & numeric.notna() & magnitude.ge(lower) & magnitude.lt(upper)
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(
                numeric.loc[mask], unit=unit, errors="coerce", utc=True
            )

    return parsed.dt.tz_convert(None).rename(name)


def canonical_valid_times(frame: pd.DataFrame) -> Tuple[pd.Series, str]:
    time_col = next((column for column in IDENTITY_TIME_COLUMNS if column in frame), "")
    if not time_col:
        raise KeyError(f"per_sample_eval.csv has none of {IDENTITY_TIME_COLUMNS}")
    times = parse_mixed_datetime(frame[time_col], time_col)

    # The 48 h dataset has an independent, physically exact valid-time
    # contract.  Use it only to fill unparseable rows and require agreement
    # wherever both representations exist.
    derived = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if "init_time" in frame and "lead_hour" in frame:
        init_time = parse_mixed_datetime(frame["init_time"], "init_time")
        lead_hour = pd.to_numeric(frame["lead_hour"], errors="coerce")
        shift_hour = pd.Series(0.0, index=frame.index, dtype=float)
        if "meta_time_shift_hours" in frame:
            shift_hour = pd.to_numeric(
                frame["meta_time_shift_hours"], errors="coerce"
            ).fillna(0.0)
        valid = init_time.notna() & lead_hour.notna() & shift_hour.notna()
        if valid.any():
            derived.loc[valid] = (
                init_time.loc[valid]
                + pd.to_timedelta(lead_hour.loc[valid], unit="h")
                + pd.to_timedelta(shift_hour.loc[valid], unit="h")
            )
        both = times.notna() & derived.notna()
        if both.any():
            delta_seconds = (times.loc[both] - derived.loc[both]).abs().dt.total_seconds()
            mismatch = delta_seconds > 1.0
            if mismatch.any():
                sample = frame.loc[delta_seconds.index[mismatch][:5], [time_col, "init_time", "lead_hour"]]
                raise ValueError(
                    "Valid time disagrees with init_time + lead_hour for "
                    f"{int(mismatch.sum())} rows; examples={sample.to_dict('records')}"
                )
        times = times.fillna(derived)

    if times.isna().any():
        bad_mask = times.isna()
        bad = int(bad_mask.sum())
        example_columns = [
            column
            for column in (time_col, "init_time", "lead_hour")
            if column in frame
        ]
        examples = frame.loc[bad_mask, example_columns].head(5).to_dict("records")
        raise ValueError(
            f"Cannot recover {bad} sample times from {time_col}; examples={examples}"
        )
    return times, time_col


def identity_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.MultiIndex]:
    if "station_id" not in frame:
        raise KeyError("per_sample_eval.csv has no station_id column")
    times, _ = canonical_valid_times(frame)
    identity = pd.DataFrame(
        {
            "station_id": canonical_station_id(frame["station_id"]),
            "valid_time_ns": times.astype("int64"),
        }
    )
    for source_name, output_name in (
        ("init_time", "init_time_ns"),
        ("forecast_reference_time", "forecast_reference_time_ns"),
    ):
        if source_name in frame:
            values = parse_mixed_datetime(frame[source_name], source_name)
            if values.notna().any():
                if values.isna().any():
                    raise ValueError(f"Cannot parse all identity values from {source_name}")
                identity[output_name] = values.astype("int64")
    for source_name in ("lead_hour", "init_hour"):
        if source_name in frame:
            values = pd.to_numeric(frame[source_name], errors="coerce")
            if values.notna().any():
                if values.isna().any():
                    raise ValueError(f"Cannot parse all identity values from {source_name}")
                identity[source_name] = values.astype(float)
    identity["duplicate_index"] = identity.groupby(
        list(identity.columns), sort=False
    ).cumcount()
    key = pd.MultiIndex.from_frame(identity, names=list(identity.columns))
    if not key.is_unique:
        raise ValueError("Composite sample identity is not unique after duplicate indexing")
    return identity, key


def identity_sha256(identity: pd.DataFrame, frame: pd.DataFrame) -> str:
    columns = [identity.reset_index(drop=True)]
    for name in ("y_true", "vis_raw_m"):
        if name in frame:
            columns.append(frame[[name]].reset_index(drop=True))
    payload = pd.concat(columns, axis=1)
    hashed = pd.util.hash_pandas_object(payload, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def validate_probabilities(path: Path, probs: np.ndarray, atol: float) -> None:
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"{path}: expected probability shape (N, 3), got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError(f"{path}: probabilities contain NaN or infinity")
    if float(np.min(probs)) < -atol or float(np.max(probs)) > 1.0 + atol:
        raise ValueError(
            f"{path}: probabilities outside [0, 1], min={float(np.min(probs))}, "
            f"max={float(np.max(probs))}"
        )
    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=atol):
        max_error = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"{path}: probability row sums differ from 1 (max error={max_error})")


def find_metrics_table(eval_dir: Path) -> Path:
    candidates = sorted(eval_dir.glob("precision_candidates_*_overall_metrics.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one precision candidate overall table in {eval_dir}; found {candidates}"
        )
    return candidates[0]


def locate_member_files(eval_dir: Path, experiment_id: int, label: str) -> Tuple[Path, Path]:
    exact_prob = eval_dir / f"{experiment_id}_{label}_probs.npy"
    exact_sample = eval_dir / f"{experiment_id}_{label}_event_eval" / "per_sample_eval.csv"
    if exact_prob.is_file() and exact_sample.is_file():
        return exact_prob, exact_sample
    prob_candidates = sorted(eval_dir.glob(f"{experiment_id}_*_probs.npy"))
    sample_candidates = sorted(eval_dir.glob(f"{experiment_id}_*_event_eval/per_sample_eval.csv"))
    if len(prob_candidates) != 1 or len(sample_candidates) != 1:
        raise FileNotFoundError(
            f"Cannot uniquely locate member experiment_id={experiment_id}, label={label!r}: "
            f"probs={prob_candidates}, samples={sample_candidates}"
        )
    return prob_candidates[0], sample_candidates[0]


def manifest_members(
    manifest_path: Path,
    candidate_id: str,
    expected_seeds: Sequence[str],
    required_stage: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    required = {
        "candidate_id",
        "candidate_label",
        "seed",
        "stage",
        "run_id",
        "s2_checkpoint",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    selected = manifest.loc[manifest["candidate_id"] == candidate_id].copy()
    if selected.empty:
        raise ValueError(f"Manifest has no candidate_id={candidate_id!r}")
    actual_seeds = selected["seed"].astype(str).tolist()
    if set(actual_seeds) != set(expected_seeds) or len(actual_seeds) != len(expected_seeds):
        raise ValueError(f"Seed mismatch: expected={list(expected_seeds)}, actual={actual_seeds}")
    if required_stage != "any" and set(selected["stage"].astype(str)) != {required_stage}:
        raise ValueError(
            f"Manifest stage mismatch: required={required_stage}, "
            f"actual={sorted(set(selected['stage'].astype(str)))}"
        )
    order = {seed: index for index, seed in enumerate(expected_seeds)}
    selected["_seed_order"] = selected["seed"].map(order)
    return selected.sort_values("_seed_order").drop(columns="_seed_order").reset_index(drop=True)


def evaluation_members(eval_dir: Path, candidate_id: str, expected_seeds: Sequence[str]) -> pd.DataFrame:
    table_path = find_metrics_table(eval_dir)
    metrics = pd.read_csv(table_path, dtype={"seed": str, "candidate_id": str})
    required = {
        "candidate_id",
        "seed",
        "experiment_id",
        "label",
        "run_id",
        "checkpoint",
        "scaler",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise KeyError(f"{table_path} is missing columns: {sorted(missing)}")
    selected = metrics.loc[metrics["candidate_id"].astype(str) == candidate_id].copy()
    selected["seed"] = selected["seed"].astype(str)
    actual_seeds = selected["seed"].tolist()
    if set(actual_seeds) != set(expected_seeds) or len(actual_seeds) != len(expected_seeds):
        raise ValueError(
            f"{table_path}: expected seeds={list(expected_seeds)}, actual={actual_seeds}"
        )
    order = {seed: index for index, seed in enumerate(expected_seeds)}
    selected["_seed_order"] = selected["seed"].map(order)
    selected = selected.sort_values("_seed_order").drop(columns="_seed_order").reset_index(drop=True)
    selected.attrs["metrics_path"] = str(table_path)
    return selected


def verify_eval_config(eval_dir: Path, manifest_path: Path, base: Path) -> Dict[str, object]:
    config_path = eval_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing evaluator provenance: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_manifest = str(config.get("manifest", "") or "")
    if not configured_manifest:
        raise ValueError(f"Evaluator run_config has no manifest: {config_path}")
    if Path(configured_manifest).expanduser().resolve() != manifest_path.expanduser().resolve():
        raise ValueError(
            f"Evaluator manifest mismatch: config={configured_manifest}, requested={manifest_path}"
        )
    configured_manifest_sha256 = str(config.get("manifest_sha256", "") or "").strip()
    if not configured_manifest_sha256 or configured_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError(f"Evaluator manifest hash mismatch: {config_path}")
    if str(config.get("threshold_source", "")) != "argmax":
        raise ValueError(f"Evaluator did not use argmax: {config_path}")
    if str(config.get("eval_split", "")) != "test":
        raise ValueError(f"Seed-mean reuse must be prepared from the frozen test split: {config_path}")
    data_value = str(config.get("data_dir", "") or "").strip()
    dataset_provenance = (
        config.get("dataset_provenance", {})
        if isinstance(config.get("dataset_provenance", {}), dict)
        else {}
    )
    if not data_value or not dataset_provenance:
        raise ValueError(f"Evaluator run_config has no dataset provenance: {config_path}")
    data_dir = resolve_path(base, data_value)
    for filename, hash_key in (("y_test.npy", "y_sha256"), ("meta_test.csv", "meta_sha256")):
        path = data_dir / filename
        expected_hash = str(dataset_provenance.get(hash_key, "") or "")
        if not path.is_file() or not expected_hash or sha256_file(path) != expected_hash:
            raise ValueError(f"Evaluator dataset provenance mismatch for {path}")
    return config


def verify_frame_against_dataset(
    frame: pd.DataFrame,
    data_dir: Path,
    meta_time_shift_hours: float,
) -> Dict[str, object]:
    _, y_cls, y_raw, meta = journal.load_main_data(
        data_dir,
        limit_samples=0,
        meta_time_shift_hours=meta_time_shift_hours,
        split="test",
    )
    if len(frame) != len(y_cls):
        raise ValueError(
            f"Dataset/sample row mismatch for {data_dir}: frame={len(frame)}, dataset={len(y_cls)}"
        )
    frame_identity, frame_key = identity_frame(frame)
    data_identity, data_key = identity_frame(meta)
    if not frame_key.equals(data_key):
        raise ValueError(
            f"Sample order or identity differs from the evaluator dataset: {data_dir}. "
            "Reusable probabilities cannot be consumed by row position."
        )
    if not np.array_equal(frame["y_true"].to_numpy(dtype=np.int64), y_cls.astype(np.int64)):
        raise ValueError(f"y_true differs from the evaluator dataset: {data_dir}")
    if not np.allclose(
        frame["vis_raw_m"].to_numpy(dtype=float),
        np.asarray(y_raw, dtype=float),
        equal_nan=True,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(f"vis_raw_m differs from the evaluator dataset: {data_dir}")
    return {
        "data_dir": str(data_dir.resolve()),
        "n_samples": int(len(frame)),
        "sample_identity_sha256": identity_sha256(data_identity, frame),
        "y_test_sha256": sha256_file(data_dir / "y_test.npy"),
        "meta_test_sha256": sha256_file(data_dir / "meta_test.csv"),
    }


def average_eval_directory(
    eval_dir: Path,
    base: Path,
    manifest_rows: pd.DataFrame,
    candidate_id: str,
    expected_seeds: Sequence[str],
    probability_atol: float,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    eval_config = verify_eval_config(eval_dir, Path(manifest_rows.attrs["manifest_path"]), base)
    eval_rows = evaluation_members(eval_dir, candidate_id, expected_seeds)
    merged = eval_rows.merge(
        manifest_rows[["seed", "candidate_label", "run_id", "s2_checkpoint"]],
        on="seed",
        how="left",
        suffixes=("_eval", "_manifest"),
        validate="one_to_one",
    )
    if not np.all(merged["run_id_eval"].astype(str) == merged["run_id_manifest"].astype(str)):
        raise ValueError("Evaluator run_id values do not match the manifest")
    if not np.all(merged["label"].astype(str) == merged["candidate_label"].astype(str)):
        raise ValueError("Evaluator labels do not match the manifest candidate labels")

    config_members = eval_config.get("members", [])
    if not isinstance(config_members, list) or len(config_members) != len(expected_seeds):
        raise ValueError(f"Evaluator run_config has no exact member provenance set: {eval_dir}")
    config_by_seed: Dict[str, Dict[str, object]] = {}
    for item in config_members:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed evaluator member provenance: {eval_dir}")
        if str(item.get("candidate_id", "")) != candidate_id:
            continue
        seed = str(item.get("seed", ""))
        if seed in config_by_seed:
            raise ValueError(f"Duplicate evaluator provenance for seed={seed}")
        config_by_seed[seed] = item
    if set(config_by_seed) != set(expected_seeds):
        raise ValueError(
            f"Evaluator provenance seed mismatch: expected={list(expected_seeds)}, actual={sorted(config_by_seed)}"
        )

    reference_frame: pd.DataFrame | None = None
    reference_identity: pd.DataFrame | None = None
    reference_key: pd.MultiIndex | None = None
    accumulator: np.ndarray | None = None
    member_records: List[Dict[str, object]] = []

    for row in merged.to_dict("records"):
        experiment_id = int(row["experiment_id"])
        label = str(row["label"])
        prob_path, sample_path = locate_member_files(eval_dir, experiment_id, label)
        eval_checkpoint = resolve_path(base, str(row["checkpoint"]))
        manifest_checkpoint = resolve_path(base, str(row["s2_checkpoint"]))
        if eval_checkpoint.resolve() != manifest_checkpoint.resolve():
            raise ValueError(
                f"Evaluator checkpoint differs from manifest for seed={row['seed']}: "
                f"eval={eval_checkpoint}, manifest={manifest_checkpoint}"
            )
        scaler_path = resolve_path(base, str(row["scaler"]))
        for required_path in (eval_checkpoint, scaler_path):
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        provenance = config_by_seed[str(row["seed"])]
        expected_fields = {
            "experiment_id": experiment_id,
            "label": label,
            "run_id": str(row["run_id_eval"]),
            "checkpoint": str(eval_checkpoint.resolve()),
            "scaler": str(scaler_path.resolve()),
            "probability_file": str(prob_path.resolve()),
            "per_sample_file": str(sample_path.resolve()),
        }
        for field, expected_value in expected_fields.items():
            actual_value = provenance.get(field)
            if str(actual_value) != str(expected_value):
                raise ValueError(
                    f"Evaluator provenance mismatch for seed={row['seed']}, field={field}: "
                    f"config={actual_value}, current={expected_value}"
                )
        hash_contract = {
            "checkpoint_sha256": eval_checkpoint,
            "scaler_sha256": scaler_path,
            "probability_sha256": prob_path,
            "per_sample_sha256": sample_path,
        }
        for hash_field, path in hash_contract.items():
            actual_hash = sha256_file(path)
            if str(provenance.get(hash_field, "")) != actual_hash:
                raise ValueError(
                    f"Evaluator provenance hash mismatch for seed={row['seed']}: {hash_field}"
                )
        member_probs = np.asarray(np.load(prob_path, mmap_mode="r"), dtype=np.float64)
        validate_probabilities(prob_path, member_probs, probability_atol)
        member_frame = pd.read_csv(
            sample_path,
            dtype={column: "string" for column in IDENTITY_STRING_COLUMNS},
        )
        if len(member_frame) != len(member_probs):
            raise ValueError(
                f"Row mismatch for seed={row['seed']}: samples={len(member_frame)}, probs={len(member_probs)}"
            )
        member_identity, member_key = identity_frame(member_frame)
        # Preserve a canonical valid-time column in the reusable output even
        # when the original appended CSV used mixed timestamp encodings.
        member_frame["time"] = pd.to_datetime(
            member_identity["valid_time_ns"], unit="ns", errors="raise"
        )

        if reference_frame is None:
            reference_frame = member_frame.reset_index(drop=True)
            reference_identity = member_identity.reset_index(drop=True)
            reference_key = member_key
            order_index = np.arange(len(member_frame), dtype=np.int64)
            accumulator = np.zeros(member_probs.shape, dtype=np.float64)
        else:
            assert reference_key is not None
            order_index = member_key.get_indexer(reference_key)
            if np.any(order_index < 0) or len(np.unique(order_index)) != len(order_index):
                raise ValueError(f"Cannot one-to-one align samples for seed={row['seed']}")
            aligned = member_frame.iloc[order_index].reset_index(drop=True)
            for column in ("y_true", "vis_raw_m"):
                if column not in reference_frame or column not in aligned:
                    raise KeyError(f"Missing alignment check column: {column}")
            if not np.array_equal(
                reference_frame["y_true"].to_numpy(), aligned["y_true"].to_numpy()
            ):
                raise ValueError(f"y_true mismatch after alignment for seed={row['seed']}")
            if not np.allclose(
                reference_frame["vis_raw_m"].to_numpy(dtype=float),
                aligned["vis_raw_m"].to_numpy(dtype=float),
                equal_nan=True,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(f"vis_raw_m mismatch after alignment for seed={row['seed']}")

        aligned_probs = member_probs[order_index]
        if all(column in member_frame for column in PROBABILITY_COLUMNS):
            csv_probs = member_frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)[order_index]
            if not np.allclose(csv_probs, aligned_probs, rtol=0.0, atol=max(probability_atol, 1.0e-6)):
                raise ValueError(f"Probability CSV/NPY mismatch for seed={row['seed']}")
        assert accumulator is not None
        accumulator += aligned_probs
        member_records.append(
            {
                "candidate_id": candidate_id,
                "seed": str(row["seed"]),
                "experiment_id": experiment_id,
                "label": label,
                "run_id": str(row["run_id_eval"]),
                "checkpoint": str(eval_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(eval_checkpoint),
                "scaler": str(scaler_path.resolve()),
                "scaler_sha256": sha256_file(scaler_path),
                "probability_file": str(prob_path.resolve()),
                "probability_sha256": sha256_file(prob_path),
                "per_sample_file": str(sample_path.resolve()),
                "per_sample_sha256": sha256_file(sample_path),
            }
        )

    assert accumulator is not None
    assert reference_frame is not None
    assert reference_identity is not None
    mean_probs64 = accumulator / float(len(member_records))
    row_sums = mean_probs64.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Mean probabilities contain a non-positive row sum")
    mean_probs = (mean_probs64 / row_sums).astype(np.float32)
    validate_probabilities(Path("<seed mean>"), mean_probs, probability_atol)

    pred = np.argmax(mean_probs, axis=1).astype(np.int16)
    ensemble_frame = reference_frame.copy()
    ensemble_frame["pmst_pred"] = pred
    for index, column in enumerate(PROBABILITY_COLUMNS):
        ensemble_frame[column] = mean_probs[:, index]
    ensemble_frame["pmst_correct"] = pred == ensemble_frame["y_true"].to_numpy()
    sample_identity_hash = identity_sha256(reference_identity, ensemble_frame)
    metadata = {
        "eval_dir": str(eval_dir.resolve()),
        "eval_run_config": eval_config,
        "sample_identity_sha256": sample_identity_hash,
        "n_samples": int(len(ensemble_frame)),
        "n_members": int(len(member_records)),
    }
    return mean_probs, ensemble_frame, pd.DataFrame(member_records), metadata


def write_seed_summaries(
    eval_dir: Path,
    candidate_id: str,
    out_dir: Path,
    ensemble_frame: pd.DataFrame,
    ensemble_probs: np.ndarray,
) -> Dict[str, str]:
    metrics_path = find_metrics_table(eval_dir)
    member_metrics = pd.read_csv(metrics_path)
    member_metrics = member_metrics.loc[
        member_metrics["candidate_id"].astype(str) == candidate_id
    ].copy()
    member_out = out_dir / "seed_member_overall_metrics.csv"
    member_metrics.to_csv(member_out, index=False, float_format="%.8f")

    numeric = member_metrics.select_dtypes(include=[np.number]).drop(
        columns=[column for column in ("experiment_id", "seed") if column in member_metrics],
        errors="ignore",
    )
    rows = []
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "member_mean": float(values.mean()),
                "member_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "n_seeds": int(len(values)),
            }
        )
    mean_sd_out = out_dir / "seed_member_metrics_mean_sd.csv"
    pd.DataFrame(rows).to_csv(mean_sd_out, index=False, float_format="%.8f")

    y_true = ensemble_frame["y_true"].to_numpy(dtype=np.int64)
    pred = np.argmax(ensemble_probs, axis=1).astype(np.int64)
    ensemble_metrics = journal.classification_metrics(y_true, pred, probs=ensemble_probs)
    ensemble_metrics.update(
        {
            "candidate_id": candidate_id,
            "source": "seed_probability_mean",
            "decision_rule": "mean_post_softmax_then_argmax",
            "n_seeds": int(member_metrics["seed"].nunique()),
        }
    )
    ensemble_out = out_dir / "ensemble_overall_metrics.csv"
    pd.DataFrame([ensemble_metrics]).to_csv(ensemble_out, index=False, float_format="%.8f")
    return {
        "seed_member_overall_metrics": str(member_out),
        "seed_member_metrics_mean_sd": str(mean_sd_out),
        "ensemble_overall_metrics": str(ensemble_out),
    }


def write_48h_tables(
    args: argparse.Namespace,
    base: Path,
    out_dir: Path,
    probs: np.ndarray,
) -> Dict[str, str]:
    data_dir = resolve_path(base, args.data_48h_dir)
    x_path, y_cls, _, meta = journal.load_main_data(
        data_dir,
        limit_samples=0,
        meta_time_shift_hours=args.meta_time_shift_hours,
        split="test",
    )
    data_identity, _ = identity_frame(meta)
    meta = meta.copy()
    meta["time"] = pd.to_datetime(
        data_identity["valid_time_ns"], unit="ns", errors="raise"
    )
    if len(y_cls) != len(probs):
        raise ValueError(
            f"48 h probability/data mismatch: probs={len(probs)}, targets={len(y_cls)}"
        )
    if "lead_hour" not in meta:
        raise KeyError(f"{data_dir / 'meta_test.csv'} has no lead_hour")
    pred = np.argmax(probs, axis=1).astype(np.int64)
    lead = pd.to_numeric(meta["lead_hour"], errors="coerce").to_numpy(dtype=float)
    init_cycle_hour, init_cycle_source = journal.infer_init_cycle_hour(
        meta, args.local_time_offset_hours
    )
    if init_cycle_hour.isna().all():
        raise ValueError("Cannot infer 00Z/12Z initialization cycles for 48 h evaluation")
    mask00 = journal.init_cycle_mask(init_cycle_hour, 0)
    mask12 = journal.init_cycle_mask(init_cycle_hour, 12)

    pooled = journal.lead_metrics_table(y_cls, pred, probs, lead)
    lead00 = journal.lead_metrics_table(y_cls, pred, probs, lead, mask=mask00)
    lead12 = journal.lead_metrics_table(y_cls, pred, probs, lead, mask=mask12)
    tables: Dict[str, pd.DataFrame] = {
        "metrics_by_lead_hour_48h_model.csv": pooled,
        "metrics_by_lead_hour_init00Z.csv": lead00,
        "metrics_by_lead_hour_init12Z.csv": lead12,
        "metrics_by_display_lead_hour_48h_model.csv": journal.build_display_lead_table(
            pooled, pooled, "pooled_previous_init_12_24h"
        ),
        "metrics_by_display_lead_hour_init00Z.csv": journal.build_display_lead_table(
            lead00, lead12, "previous_12Z_init_12_24h"
        ),
        "metrics_by_display_lead_hour_init12Z.csv": journal.build_display_lead_table(
            lead12, lead00, "previous_00Z_init_12_24h"
        ),
    }

    ifs_path = resolve_path(base, args.ifs_48h_nc)
    if ifs_path.is_file():
        ifs_pred, _, ifs_valid, ifs_diag = journal.load_ifs_48h_diagnostic(
            meta, ifs_path, args.ifs_48h_var
        )
        matched = np.asarray(ifs_valid, dtype=bool)
        if int(matched.sum()) < 50:
            raise ValueError(f"Only {int(matched.sum())} rows match the 48 h IFS diagnostic")
        model_matched = journal.lead_metrics_table(y_cls, pred, probs, lead, mask=matched)
        ifs_lead = journal.lead_metrics_table(y_cls, ifs_pred, None, lead, mask=matched)
        comparison = model_matched.merge(
            ifs_lead,
            on="lead_hour",
            how="inner",
            suffixes=("_model", "_ifs"),
        ).sort_values("lead_hour").reset_index(drop=True)
        for metric in (
            "Fog_CSI",
            "Fog_R",
            "Mist_CSI",
            "Mist_R",
            "low_vis_csi",
            "low_vis_recall",
        ):
            model_column = f"{metric}_model"
            ifs_column = f"{metric}_ifs"
            if model_column in comparison and ifs_column in comparison:
                comparison[f"{metric}_diff_model_minus_ifs"] = (
                    comparison[model_column] - comparison[ifs_column]
                )
        tables.update(
            {
                "model_metrics_by_lead_hour_48h_ifs_matched.csv": model_matched,
                "ifs_metrics_by_lead_hour_48h.csv": ifs_lead,
                "model_vs_ifs_metrics_by_lead_hour_48h.csv": comparison,
                "model_vs_ifs_metrics_by_display_lead_hour_48h.csv": journal.build_display_lead_table(
                    comparison, comparison, "matched_previous_init_12_24h"
                ),
                "lead_eval_alignment_diagnostics_48h_ifs.csv": pd.DataFrame([ifs_diag]),
            }
        )

    outputs: Dict[str, str] = {}
    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path, index=False, float_format="%.8f")
        outputs[name] = str(path)
    outputs["data_48h_dir"] = str(data_dir)
    outputs["x_test_48h"] = str(x_path)
    outputs["init_cycle_source"] = str(init_cycle_source)
    outputs["n_init00"] = str(int(mask00.sum()))
    outputs["n_init12"] = str(int(mask12.sum()))
    return outputs


def main() -> None:
    args = parse_args()
    base = Path(args.base).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_file():
        candidate = base / "train" / manifest_path
        if candidate.is_file():
            manifest_path = candidate
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    expected_seeds = parse_seed_list(args.expected_seeds)
    manifest_rows = manifest_members(
        manifest_path,
        args.candidate_id,
        expected_seeds,
        args.required_stage,
    )
    manifest_rows.attrs["manifest_path"] = str(manifest_path.resolve())

    out_dir = resolve_path(base, args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}; choose a new formal-run directory")
        if any(out_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to merge with a non-empty output directory: {out_dir}; choose a new directory"
            )

    main_eval_dir = resolve_path(base, args.main_eval_dir)
    main_probs, main_frame, main_members, main_meta = average_eval_directory(
        main_eval_dir,
        base,
        manifest_rows,
        args.candidate_id,
        expected_seeds,
        args.probability_atol,
    )
    main_data_value = str(main_meta["eval_run_config"].get("data_dir", "") or "").strip()
    if not main_data_value:
        raise ValueError(f"Main evaluator run_config has no data_dir: {main_eval_dir}")
    main_data_dir = resolve_path(base, main_data_value)
    main_meta["dataset_verification"] = verify_frame_against_dataset(
        main_frame,
        main_data_dir,
        args.meta_time_shift_hours,
    )
    forecast48_meta: Dict[str, object] = {}
    forecast48_outputs: Dict[str, str] = {}
    probs48: np.ndarray | None = None
    frame48: pd.DataFrame | None = None
    members48: pd.DataFrame | None = None
    if str(args.forecast48_eval_dir or "").strip():
        forecast48_eval_dir = resolve_path(base, args.forecast48_eval_dir)
        probs48, frame48, members48, forecast48_meta = average_eval_directory(
            forecast48_eval_dir,
            base,
            manifest_rows,
            args.candidate_id,
            expected_seeds,
            args.probability_atol,
        )
        forecast48_data_value = str(
            forecast48_meta["eval_run_config"].get("data_dir", "") or ""
        ).strip()
        if not forecast48_data_value:
            raise ValueError(f"48 h evaluator run_config has no data_dir: {forecast48_eval_dir}")
        forecast48_data_dir = resolve_path(base, forecast48_data_value)
        expected48_data_dir = resolve_path(base, args.data_48h_dir)
        if forecast48_data_dir.resolve() != expected48_data_dir.resolve():
            raise ValueError(
                "48 h member evaluation used a different dataset than --data-48h-dir: "
                f"eval={forecast48_data_dir.resolve()}, expected={expected48_data_dir.resolve()}"
            )
        forecast48_meta["dataset_verification"] = verify_frame_against_dataset(
            frame48,
            forecast48_data_dir,
            args.meta_time_shift_hours,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "probs.npy", main_probs)
    np.save(out_dir / "pred_argmax.npy", np.argmax(main_probs, axis=1).astype(np.int16))
    main_frame.to_csv(out_dir / "per_sample_eval.csv", index=False, float_format="%.8f")
    main_members.to_csv(out_dir / "ensemble_members.csv", index=False)
    summary_outputs = write_seed_summaries(
        main_eval_dir,
        args.candidate_id,
        out_dir,
        main_frame,
        main_probs,
    )

    if probs48 is not None:
        assert frame48 is not None
        assert members48 is not None
        np.save(out_dir / "probs_48h.npy", probs48)
        np.save(out_dir / "pred_argmax_48h.npy", np.argmax(probs48, axis=1).astype(np.int16))
        frame48.to_csv(out_dir / "per_sample_eval_48h.csv", index=False, float_format="%.8f")
        members48.to_csv(out_dir / "ensemble_members_48h.csv", index=False)
        forecast48_outputs = write_48h_tables(args, base, out_dir, probs48)

    output_paths = {
        "main_probs": out_dir / "probs.npy",
        "main_pred_argmax": out_dir / "pred_argmax.npy",
        "main_per_sample": out_dir / "per_sample_eval.csv",
        "main_members": out_dir / "ensemble_members.csv",
        "forecast48_probs": out_dir / "probs_48h.npy" if forecast48_meta else None,
        "forecast48_pred_argmax": out_dir / "pred_argmax_48h.npy" if forecast48_meta else None,
        "forecast48_per_sample": out_dir / "per_sample_eval_48h.csv" if forecast48_meta else None,
        "forecast48_members": out_dir / "ensemble_members_48h.csv" if forecast48_meta else None,
    }
    output_hashes = {
        key: sha256_file(path)
        for key, path in output_paths.items()
        if path is not None and path.is_file()
    }
    for key, value in {**summary_outputs, **forecast48_outputs}.items():
        path = Path(value)
        if path.is_file():
            output_hashes[key] = sha256_file(path)

    config = {
        "schema_version": 1,
        "experiment_status": "formal_three_seed_candidate",
        "replaces_mainline": False,
        "candidate_id": args.candidate_id,
        "seeds": expected_seeds,
        "ensemble_size": len(expected_seeds),
        "probability_combination": "equal_weight_mean_of_post_softmax_probabilities",
        "decision_rule": "argmax_of_mean_probabilities",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "main": main_meta,
        "forecast48": forecast48_meta,
        "outputs": {
            **{key: str(path) if path is not None else "" for key, path in output_paths.items()},
            **summary_outputs,
            **forecast48_outputs,
        },
        "output_sha256": output_hashes,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ensemble] candidate={args.candidate_id} seeds={expected_seeds}", flush=True)
    print(f"[ensemble] decision=mean post-softmax probabilities, then argmax", flush=True)
    print(f"[ensemble] main rows={len(main_probs)} out={out_dir / 'probs.npy'}", flush=True)
    if forecast48_meta:
        print(f"[ensemble] 48h rows={forecast48_meta['n_samples']} out={out_dir / 'probs_48h.npy'}", flush=True)
    print(f"[config] {out_dir / 'run_config.json'}", flush=True)


if __name__ == "__main__":
    main()
