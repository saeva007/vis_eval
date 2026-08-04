# AGU 2026 aerosol-conditioned low-visibility experiment

## Scientific question

This experiment tests whether PM10/PM2.5 sequences provide predictive
information beyond the meteorological and station inputs in the current
Static-MLP + GRU framework.  It does not estimate a causal aerosol effect,
separate anthropogenic and natural aerosol sources, or attribute an event to a
single mechanism.

The three manuscript events can be used in the AGU presentation as contrasting
compound environments.  Their roles are descriptive:

- 2025-02-27 22:00 UTC;
- 2025-09-28 23:00 UTC;
- 2025-10-30 22:00 UTC.

The analysis reports humidity, month-relative particulate loading, weak-wind
frequency, precipitation occurrence, and Full-versus-No-PM skill for each
event.  A fog, haze, precipitation, anthropogenic, or natural-source label is
allowed only if independent evidence supports it.

## Controlled design

- Arms: `full` (27 dynamic channels) and `no_pm` (the same 25 non-PM dynamic
  channels, with no zero PM placeholders).
- Seeds: `42`, `314`, and `2718`.
- Training: independent S1 -> S2 training for every arm and seed.
- Loss control: both arms explicitly use `--aerosol-hard-weight 0`; therefore
  PM-dependent sample weighting cannot confound the input ablation.
- Ensemble: arithmetic mean of the three post-softmax probability arrays,
  followed by one decision rule.
- Selection: each arm's combined low-visibility threshold is selected only on
  validation data under the pre-registered FPR target of 0.04.
- Test: frozen thresholds are applied once to the aligned test rows.
- Uncertainty: Full-minus-No-PM differences use paired UTC-date block
  bootstrap with 1,000 iterations.
- Conditional analysis: RH uses physical bands (`<70%`, `70-90%`, `>=90%`).
  PM10/PM2.5 use validation-referenced within-month ranks because the historical
  mainline files retain a legacy numeric PM scale.

## One-command launch

Run from the remote evaluation repository after pulling the pushed commit:

```bash
cd /public/home/putianshu/vis_mlp/paper_eval
git pull --ff-only origin main
bash submit_aerosol_ablation_chain.sh
```

The public launcher immediately submits a short Slurm controller and prints a
controller log and a sourceable state-file path. This keeps chain submission
alive when the originating SSH/login-node session disappears. The controller
persists every data-build, training, evaluation, ensemble, and analysis JobID.
After submission:

```bash
source /public/home/putianshu/vis_mlp/paper_eval/logs/<bundle>.state.sh
squeue -j "${ALL_JOB_IDS//:/,}"
```

The launcher overrides legacy generic Slurm names with explicit roles such as
`aero_full_s1_42`, `aero_nopm_s2_314`, and `aero_final_analysis`. The historical
`airport25` filenames denote the 25-variable No-PM layout; they do not restrict
the experiment to airport stations.

S1 readiness follows the established trainer contract and requires the four
train/validation arrays. S2 additionally requires validation/test row metadata.
For legacy Full S2 builds that predate `dataset_build_config.json`, the analysis
accepts the feature order only after the array width proves the fixed 27-variable
PM10+PM2.5 mainline layout; this fallback is recorded in the run configuration.

Do not resubmit a partially built No-PM dataset.  The launcher stops on a
partial directory so recovery can use the recorded JobIDs and logs.

## Primary outputs

The final analysis directory is recorded as `ANALYSIS_OUT_DIR` in the state
file and contains:

- `aerosol_ablation_overall_metrics.csv`;
- `aerosol_ablation_metric_differences.csv`;
- `aerosol_ablation_date_block_bootstrap.csv`;
- `rh_pm_conditional_skill.csv`;
- `three_event_aerosol_environment_summary.csv`;
- `aerosol_ablation_report.md`;
- `aerosol_ablation_run_config.json`.

## Claim gate

An abstract may say that aerosol information improves predictive skill only if
the Full-minus-No-PM direction is positive and the paired confidence interval
supports that direction for the metric being cited.  Conditional or event
claims must match `rh_pm_conditional_skill.csv` and
`three_event_aerosol_environment_summary.csv`.  The experiment never supports
the statements “aerosols caused the event” or “anthropogenic aerosols dominate
the event.”
