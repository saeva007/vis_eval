#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
PAPER_EVAL="${PAPER_EVAL:-${BASE}/paper_eval}"
IFS_BASELINE="${IFS_BASELINE:-${BASE}/ifs_baseline}"

EVAL_DIR="${EVAL_DIR:-${BASE}/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal}"
IFS_OUT_DIR="${IFS_OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/ifs_event_grid_inference}"
IFS_SAMPLE_CSV="${IFS_SAMPLE_CSV:-}"
if [[ -z "${IFS_SAMPLE_CSV}" ]]; then
    EXISTING_BEST_EFFORT_IFS="${BASE}/paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/figure1_all_sources/per_sample_ifs.csv"
    if [[ -s "${EXISTING_BEST_EFFORT_IFS}" ]]; then
        IFS_SAMPLE_CSV="${EXISTING_BEST_EFFORT_IFS}"
    else
        IFS_SAMPLE_CSV="${IFS_OUT_DIR}/per_sample_ifs.csv"
    fi
fi
IFS_SLURM="${IFS_SLURM:-${IFS_BASELINE}/sub_static_rnn_source_full_argmax_eval.slurm}"
PLOT_SLURM="${PLOT_SLURM:-${PAPER_EVAL}/sub_rerun_static_rnn_event_figures.slurm}"

for required in "${EVAL_DIR}/per_sample_eval.csv" "${IFS_SLURM}" "${PLOT_SLURM}"; do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: missing required file: ${required}" >&2
        exit 2
    fi
done

IFS_JOB=""
DEPENDENCY_ARGS=()
if [[ -s "${IFS_SAMPLE_CSV}" ]]; then
    echo "Reusing best-effort IFS inference: ${IFS_SAMPLE_CSV}"
else
    IFS_SUBMISSION="$(
        sbatch --parsable \
            --export=ALL,EVAL_SCENARIO=figure1_all_sources,SOURCE_SUBSET=ifs,OUT_DIR="${IFS_OUT_DIR}",NO_PER_SAMPLE_CSV=0,NO_FIGURES=1,SKIP_IFS_FORECAST_BASELINE=1,REQUIRE_BEST_EFFORT_ENSEMBLE=0,SKIP_OPERATIONAL_ENSEMBLE_ANALYSIS=1 \
            "${IFS_SLURM}"
    )"
    IFS_JOB="${IFS_SUBMISSION%%;*}"
    DEPENDENCY_ARGS=(--dependency="afterok:${IFS_JOB}")
fi

PLOT_SUBMISSION="$(
    sbatch --parsable \
        "${DEPENDENCY_ARGS[@]+"${DEPENDENCY_ARGS[@]}"}" \
        --export=ALL,EVAL_DIR="${EVAL_DIR}",WINDOW_HOURS=3,EVENT_ENV_MAX_EVENTS=3,EVENT_ENV_INCLUDE_CSI=1,EVENT_ENV_WITH_SOURCE_MODELS=1,EVENT_ENV_OVERLAP_EVAL="${IFS_SAMPLE_CSV}",EVENT_ENV_WITH_PANGU=1,ENVIRONMENT_GRIDS_ONLY=1 \
        "${PLOT_SLURM}"
)"
PLOT_JOB="${PLOT_SUBMISSION%%;*}"

echo "IFS_SAMPLE_CSV=${IFS_SAMPLE_CSV}"
echo "IFS_JOB=${IFS_JOB:-reused_existing_output}"
echo "PLOT_JOB=${PLOT_JOB}"
