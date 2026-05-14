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
