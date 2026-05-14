#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared JSON configuration support for paper evaluation scripts."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set


DEFAULT_CONFIG_NAME = "paper_eval_config.json"
READ_ONLY_MODES = {"tables", "plot"}


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _cli_options(argv: Optional[Sequence[str]]) -> Set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    out: Set[str] = set()
    for token in tokens:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].replace("-", "_")
        out.add(key)
    return out


def _candidate_config_path(args: Namespace, default_dir: Path) -> Optional[Path]:
    value = getattr(args, "config_json", "") or os.environ.get("PAPER_EVAL_CONFIG", "")
    if not value:
        return None
    if str(value).strip().lower() in {"none", "off", "false", "0"}:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = default_dir / path
    return path


def _load_raw_config(path: Path, explicit: bool) -> Dict[str, Any]:
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"Configured paper-eval JSON not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Paper-eval config must be a JSON object: {path}")
    return data


def _collect_scalar_context(config: Mapping[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            context[key] = value
    for section_name in ("common", "model", "outputs"):
        section = config.get(section_name, {})
        if isinstance(section, Mapping):
            for key, value in section.items():
                if isinstance(value, (str, int, float, bool)):
                    context[key] = value
    return context


def _render_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_SafeFormatDict(context))
    if isinstance(value, list):
        return [_render_value(v, context) for v in value]
    if isinstance(value, dict):
        return {k: _render_value(v, context) for k, v in value.items()}
    return value


def _render_config(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = deepcopy(dict(raw_config))
    for _ in range(4):
        context = _collect_scalar_context(config)
        rendered = _render_value(config, context)
        if rendered == config:
            return rendered
        config = rendered
    return config


def _abs_under_base(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.name}_r{idx:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a non-overwriting output path near {path}")


def _should_uniquify(args: Namespace, cli_seen: Set[str]) -> bool:
    if "out_dir" in cli_seen:
        return False
    mode = str(getattr(args, "mode", "") or "").lower()
    return mode not in READ_ONLY_MODES


def apply_paper_eval_config(
    args: Namespace,
    section: str,
    *,
    argv: Optional[Sequence[str]] = None,
    default_dir: Optional[Path] = None,
) -> Namespace:
    """Apply central JSON defaults to an argparse Namespace.

    Explicit command-line options always win over JSON values. Relative output
    paths are interpreted under ``args.base`` when prevent_overwrite is enabled.
    """

    config_dir = Path(default_dir) if default_dir is not None else Path(__file__).resolve().parent
    cli_seen = _cli_options(argv)
    config_path = _candidate_config_path(args, config_dir)
    explicit_config = "config_json" in cli_seen or bool(os.environ.get("PAPER_EVAL_CONFIG"))
    if config_path is None:
        return args

    raw_config = _load_raw_config(config_path, explicit=explicit_config)
    if not raw_config:
        return args
    config = _render_config(raw_config)

    values: Dict[str, Any] = {}
    for name in ("common", "model", section):
        item = config.get(name, {})
        if isinstance(item, Mapping):
            values.update(item)

    outputs = config.get("outputs", {})
    if isinstance(outputs, Mapping):
        output_key = f"{section}_out_dir"
        if output_key in outputs:
            values["out_dir"] = outputs[output_key]
        elif section == "journal" and "out_dir" in outputs:
            values["out_dir"] = outputs["out_dir"]
        elif "out_dir" in outputs and section not in {"journal"}:
            values["out_dir"] = outputs["out_dir"]

    for key, value in values.items():
        if key in cli_seen:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
        elif key == "s2_run_id":
            setattr(args, "config_s2_run_id", value)

    if isinstance(outputs, Mapping):
        if "run_tag" in outputs:
            setattr(args, "config_run_tag", outputs["run_tag"])
        prevent = bool(outputs.get("prevent_overwrite", False))
        setattr(args, "config_prevent_overwrite", prevent)
        if prevent and hasattr(args, "out_dir") and _should_uniquify(args, cli_seen):
            base = Path(getattr(args, "base", ".")).expanduser()
            out_path = _abs_under_base(base, str(getattr(args, "out_dir")))
            unique = _unique_path(out_path)
            if unique != out_path:
                print(f"[config] output exists; using non-overwriting path: {unique}", flush=True)
            setattr(args, "out_dir", str(unique))

    setattr(args, "config_json", str(config_path))
    setattr(args, "config_loaded", True)
    return args
