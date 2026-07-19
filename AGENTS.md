# Agent Rules for Paper Evaluation Scripts

## Central Run Configuration

- Treat `paper_eval_config.json` as the primary source for evaluation run settings: model run id, checkpoint, scaler, season-threshold file, data directories, IFS input files, thresholds, and output directories.
- The main journal evaluation, feature-importance evaluation, and feature-engineering ablation evaluation scripts all accept `--config_json`. Their Slurm launchers should pass this JSON path and avoid duplicating model paths or thresholds in shell defaults.
- Command-line options explicitly passed to a script override JSON values. Use `--config_json none` only when a one-off run intentionally needs the script's built-in fallback defaults.
- Keep `outputs.prevent_overwrite` enabled for paper runs. If a configured output directory already exists, the loader will append `_r02`, `_r03`, and so on to avoid overwriting older figures.
- When a new model run, output root, or figure family is introduced, update `paper_eval_config.json` first, then keep script changes limited to genuinely new behavior.

## Factual Boundaries

- Keep class definitions aligned with the project guide: Fog is `0 <= visibility < 500 m`, Mist is `500 <= visibility < 1000 m`, and Clear is `visibility >= 1000 m`.
- Before changing thresholds, model paths, data versions, IFS baseline paths, or reported metrics, verify the current values in the relevant source files or result artifacts.

## SSH-Resilient Slurm Submission

- Do not deliver a multi-step paper-evaluation chain as a sequence that must
  remain inside one interactive SSH session. Put checkpoint checks, exports,
  manifest construction, and all dependent `sbatch` calls in a tracked launcher;
  run its worker with `nohup ... </dev/null >log 2>&1 &`. Prefer a public
  launcher that performs this detachment itself so the user runs one command.
- A successful `test -s` is silent and can be mistaken for a hung terminal.
  Long-form launchers must print explicit preflight `checkpoint=OK` messages.
- Persist the bundle ID, output paths, and every returned JobID after each
  submission in a sourceable state file. If SSH disconnects, recovery must use
  that file instead of reconstructing state from shell history or resubmitting
  the whole chain blindly.

## Figure/Layout Regression Pitfalls

- Do not treat a visible overlap in one PNG as a one-line fontsize problem. First inspect every function that writes the same figure family, shared legend/colorbar helpers, and any merge-only or rerun scripts that can redraw the same output from cached CSVs.
- After label or font-size changes, check for stale visible strings across all related figures: obsolete class names, redundant axis labels on shared panels, old event wording such as `true peak`, old legend entries such as `IFS missing`, and notes that no longer match the rendered line style.
- Event figures have two different concepts of order: selection priority and chronological display. Paper-facing event files and titles should use the display order intended by the manuscript; if chronological order is required, sort by `peak_time` and renumber display `event_rank` instead of relying on old CSV rank values.
- For `fig9_events_ultralow_lowvis_counts_1x3`, do not trust the hourly filename rank as the display order. Sort panels by the timestamp where `hour_offset == 0`, and keep both footprint scales in the panel: `*_low_vis_count` for the overall Low-vis event and `*_ultralow_count`/legacy `*_fog_count` for the Ultra-low core.
- For colorbars with direction labels such as `IFS better` and `PMST better`, never place those labels on the same side as the colorbar xlabel. Put endpoint labels above the bar or reserve a separate axis, then run a bounding-box or rendered-image check. If the labels still collide after one layout attempt, delete the optional endpoint labels rather than repeatedly moving them around.
- For 48 h lead-time figures, do not mark the first 0-12 h segment as stitched unless the current manuscript explicitly wants that visual encoding. Keep `fig11_48h_lead_init_00Z_12Z` and model-vs-IFS lead curves continuous when the stitched segment should be hidden.
- Reuse completed inference whenever possible. `REUSE_INFERENCE_DIR` should skip main test inference via `probs.npy`; if a figure is driven by derived CSVs such as 48 h display-lead tables, add a CSV-reuse path instead of silently recomputing model outputs.
- Verification should include both syntax checks and semantic grep/AST assertions for removed labels, ordering, and output paths. If local `py_compile` writes fail because of a locked or permission-restricted `__pycache__`, use no-write AST parsing rather than deleting cache files.
