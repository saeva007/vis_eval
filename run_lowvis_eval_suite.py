#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dispatch journal-style evaluation for the current main model or RNN matrix.

The current paper main model keeps using ``run_paper_eval_pm10_pm25_journal.py``.
The compact Static-MLP + RNN experiment matrix uses
``run_static_rnn_lowvis_eval_journal.py``, which reuses the same metric and
figure helpers for checkpoint comparison.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from paper_eval_config import DEFAULT_CONFIG_NAME, apply_paper_eval_config


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run low-visibility paper evaluation for main model and/or RNN matrix.")
    p.add_argument("--config_json", default=os.environ.get("PAPER_EVAL_CONFIG", str(SCRIPT_DIR / DEFAULT_CONFIG_NAME)))
    p.add_argument("--target", choices=["main", "matrix", "both"], default="main")
    p.add_argument("--main_script", default="run_paper_eval_pm10_pm25_journal.py")
    p.add_argument("--matrix_script", default="run_static_rnn_lowvis_eval_journal.py")
    p.add_argument("--main_mode", default="all")
    p.add_argument("--main_out_dir", default="")
    p.add_argument("--matrix_run_prefix", default="")
    p.add_argument("--matrix_experiments", default="1:2:3:4:5:6:7")
    p.add_argument("--matrix_out_dir", default="")
    p.add_argument("--plots", choices=["none", "core", "all"], default="core")
    p.add_argument("--device", default="")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--extra_main_args", default="")
    p.add_argument("--extra_matrix_args", default="")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def resolve_script(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def append_optional(cmd: List[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, int):
        if value <= 0:
            return
        cmd.extend([flag, str(value)])
        return
    text = str(value)
    if text:
        cmd.extend([flag, text])


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("Command:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_main_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, str(resolve_script(args.main_script)), "--config_json", str(args.config_json)]
    append_optional(cmd, "--mode", args.main_mode)
    append_optional(cmd, "--out_dir", args.main_out_dir)
    append_optional(cmd, "--device", args.device)
    append_optional(cmd, "--limit_samples", args.limit_samples)
    if args.extra_main_args:
        cmd.extend(shlex.split(args.extra_main_args))
    return cmd


def build_matrix_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(resolve_script(args.matrix_script)),
        "--config_json",
        str(args.config_json),
        "--mode",
        "matrix",
    ]
    append_optional(cmd, "--matrix_run_prefix", args.matrix_run_prefix)
    append_optional(cmd, "--matrix_experiments", args.matrix_experiments)
    append_optional(cmd, "--out_dir", args.matrix_out_dir)
    append_optional(cmd, "--device", args.device)
    append_optional(cmd, "--limit_samples", args.limit_samples)
    append_optional(cmd, "--plots", args.plots)
    if args.extra_matrix_args:
        cmd.extend(shlex.split(args.extra_matrix_args))
    return cmd


def main() -> None:
    args = parse_args()
    args = apply_paper_eval_config(args, "lowvis_eval_suite", default_dir=SCRIPT_DIR)

    if args.target in {"matrix", "both"} and not args.matrix_run_prefix:
        raise ValueError("--matrix_run_prefix is required when --target is matrix or both.")

    print("Low-visibility evaluation suite", flush=True)
    print(f"target     : {args.target}", flush=True)
    print(f"config_json: {args.config_json}", flush=True)

    if args.target in {"main", "both"}:
        print("\n=== Main model: original journal evaluation ===", flush=True)
        run_cmd(build_main_cmd(args), args.dry_run)

    if args.target in {"matrix", "both"}:
        print("\n=== Static-RNN experiment matrix evaluation ===", flush=True)
        run_cmd(build_matrix_cmd(args), args.dry_run)


if __name__ == "__main__":
    main()
