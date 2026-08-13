#!/bin/bash
#SBATCH -J viscast_main_figs
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/viscast_main_figs_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/viscast_main_figs_%j.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
EVAL_DIR="${BASE}/paper_eval"
IFS_DIR="${BASE}/ifs_baseline"
QCORE_ANALYSIS_DIR="${QCORE_ANALYSIS_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/qcore_t925_mhtpw_formal_v2_20260728/analysis}"
QCORE_ENDPOINT_DIR="${QCORE_ENDPOINT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/qcore_t925_fair_formal_v1_20260721/analysis}"
MAIN_EVAL_DIR="${MAIN_EVAL_DIR:-${BASE}/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal}"
S1_CSV="${S1_CSV:-auto}"
SAMPLING_CSV="${SAMPLING_CSV:-auto}"
LOSS_CSV="${LOSS_CSV:-auto}"
UPPER_AIR_DIR="${UPPER_AIR_DIR:-auto}"
PAIRED_QUALITY_DIR="${PAIRED_QUALITY_DIR:-auto}"
TAIL_ANALYSIS_DIR="${TAIL_ANALYSIS_DIR:-auto}"
IFS_COMPARISON_DIR="${IFS_COMPARISON_DIR:-auto}"
SOURCE_FULL_METRICS_CSV="${SOURCE_FULL_METRICS_CSV:-auto}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/manuscript_main_figures_redraw_${SLURM_JOB_ID}}"
SHP_PATH="${SHP_PATH:-/public/home/putianshu/中华人民共和国/中华人民共和国.shp}"

latest_file() {
    local name="$1"
    find "${BASE}" -maxdepth 7 -type f -name "${name}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

if [[ "${MAIN_EVAL_DIR}" == "auto" ]]; then
    station_file="$(latest_file station_model_vs_ifs_metrics.csv)"
    [[ -n "${station_file}" ]] || { echo "ERROR: could not find station_model_vs_ifs_metrics.csv" >&2; exit 2; }
    MAIN_EVAL_DIR="$(dirname "${station_file}")"
fi
if [[ "${S1_CSV}" == "auto" ]]; then
    S1_CSV="$(latest_file static_rnn_month_group_split_metric_deltas.csv)"
    if [[ -z "${S1_CSV}" ]]; then
        S1_CSV="$(latest_file static_rnn_month_group_split_comparison_summary.csv)"
    fi
fi
if [[ "${SAMPLING_CSV}" == "auto" ]]; then
    SAMPLING_CSV="$(latest_file sampling_ablation_overall_metrics.csv)"
fi
if [[ "${LOSS_CSV}" == "auto" ]]; then
    LOSS_CSV="$(latest_file loss_ablation_overall_metrics.csv)"
fi
if [[ "${UPPER_AIR_DIR}" == "auto" ]]; then
    UPPER_AIR_DIR="${QCORE_ANALYSIS_DIR}"
fi
if [[ "${PAIRED_QUALITY_DIR}" == "auto" ]]; then
    paired_quality_file="$(latest_file pressure_level_paired_rmse_utc_date_bootstrap_ci.csv)"
    [[ -n "${paired_quality_file}" ]] || { echo "ERROR: could not find pressure_level_paired_rmse_utc_date_bootstrap_ci.csv" >&2; exit 2; }
    PAIRED_QUALITY_DIR="$(dirname "${paired_quality_file}")"
fi
if [[ "${TAIL_ANALYSIS_DIR}" == "auto" ]]; then
    tail_file="$(latest_file joint_tail_placement_utc_date_bootstrap_ci.csv)"
    [[ -n "${tail_file}" ]] || { echo "ERROR: could not find joint_tail_placement_utc_date_bootstrap_ci.csv" >&2; exit 2; }
    TAIL_ANALYSIS_DIR="$(dirname "${tail_file}")"
fi
if [[ "${IFS_COMPARISON_DIR}" == "auto" ]]; then
    if [[ -s "${MAIN_EVAL_DIR}/ifs_diagnostic_matched_metrics.csv" ]]; then
        IFS_COMPARISON_DIR="${MAIN_EVAL_DIR}"
    elif [[ -s "${MAIN_EVAL_DIR}/overlap_forecast_source/ifs_diagnostic_matched_metrics.csv" ]]; then
        IFS_COMPARISON_DIR="${MAIN_EVAL_DIR}/overlap_forecast_source"
    else
        ifs_matched_file="$(latest_file ifs_diagnostic_matched_metrics.csv)"
        [[ -n "${ifs_matched_file}" ]] || { echo "ERROR: could not find ifs_diagnostic_matched_metrics.csv" >&2; exit 2; }
        IFS_COMPARISON_DIR="$(dirname "${ifs_matched_file}")"
    fi
fi
if [[ "${SOURCE_FULL_METRICS_CSV}" == "auto" ]]; then
    SOURCE_FULL_METRICS_CSV="$({
        find "${BASE}" -maxdepth 8 -type f -name overall_metrics.csv -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | cut -d' ' -f2- \
            | while IFS= read -r candidate; do
                if grep -q 'tianji_t2nd_ifs_mean_softmax' "${candidate}"; then
                    printf '%s\n' "${candidate}"
                    break
                fi
            done
    } || true)"
    [[ -n "${SOURCE_FULL_METRICS_CSV}" ]] || { echo "ERROR: could not find source-full overall_metrics.csv" >&2; exit 2; }
fi

for path in \
    "${EVAL_DIR}/plot_viscast_baseline_manuscript_composite.py" \
    "${EVAL_DIR}/plot_viscast_long_tail_manuscript_composite.py" \
    "${IFS_DIR}/plot_qcore_t925_manuscript_composite.py" \
    "${IFS_DIR}/plot_viscast_controlled_attribution_composite.py" \
    "${IFS_DIR}/plot_viscast_source_full_metrics.py" \
    "${IFS_DIR}/plot_common_variable_error_regimes.py" \
    "${MAIN_EVAL_DIR}/station_model_vs_ifs_metrics.csv" \
    "${QCORE_ANALYSIS_DIR}/hybrid_exact_shapley_effects.csv" \
    "${QCORE_ENDPOINT_DIR}/qcore_t925_metrics_by_seed.csv" \
    "${QCORE_ENDPOINT_DIR}/qcore_t925_bootstrap_gap_draws.csv" \
    "${PAIRED_QUALITY_DIR}/pressure_level_paired_rmse_utc_date_bootstrap_ci.csv" \
    "${PAIRED_QUALITY_DIR}/surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv" \
    "${PAIRED_QUALITY_DIR}/surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv" \
    "${TAIL_ANALYSIS_DIR}/joint_tail_placement_utc_date_bootstrap_ci.csv" \
    "${TAIL_ANALYSIS_DIR}/joint_tail_placement_metrics.csv" \
    "${IFS_COMPARISON_DIR}/ifs_diagnostic_matched_metrics.csv" \
    "${SOURCE_FULL_METRICS_CSV}" \
    "${S1_CSV}" \
    "${SAMPLING_CSV}" \
    "${LOSS_CSV}"; do
    test -s "${path}" || { echo "ERROR: missing or empty ${path}" >&2; exit 2; }
done

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:${LD_LIBRARY_PATH:-}
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${OUT_DIR}" "${EVAL_DIR}/logs"

echo "viscast_main_figures_preflight=OK"
echo "main_eval_dir=${MAIN_EVAL_DIR}"
echo "qcore_analysis_dir=${QCORE_ANALYSIS_DIR}"
echo "qcore_endpoint_dir=${QCORE_ENDPOINT_DIR}"
echo "paired_quality_dir=${PAIRED_QUALITY_DIR}"
echo "tail_analysis_dir=${TAIL_ANALYSIS_DIR}"
echo "ifs_comparison_dir=${IFS_COMPARISON_DIR}"
echo "source_full_metrics_csv=${SOURCE_FULL_METRICS_CSV}"
echo "s1_csv=${S1_CSV}"
echo "sampling_csv=${SAMPLING_CSV}"
echo "loss_csv=${LOSS_CSV}"
echo "output_dir=${OUT_DIR}"

python "${EVAL_DIR}/plot_viscast_baseline_manuscript_composite.py" \
  --eval-dir "${MAIN_EVAL_DIR}" \
  --shp-path "${SHP_PATH}" \
  --out-dir "${OUT_DIR}"

python "${EVAL_DIR}/plot_viscast_baseline_manuscript_composite.py" \
  --eval-dir "${MAIN_EVAL_DIR}" \
  --shp-path "${SHP_PATH}" \
  --include-precision \
  --figure-stem fig_viscast_vs_ifs_main_composite_with_precision \
  --out-dir "${OUT_DIR}"

python "${IFS_DIR}/plot_qcore_t925_manuscript_composite.py" \
  --analysis-dir "${QCORE_ANALYSIS_DIR}" \
  --endpoint-dir "${QCORE_ENDPOINT_DIR}" \
  --upper-air-dir "${UPPER_AIR_DIR}" \
  --paired-quality-dir "${PAIRED_QUALITY_DIR}" \
  --tail-analysis-dir "${TAIL_ANALYSIS_DIR}" \
  --out-dir "${OUT_DIR}"

python "${IFS_DIR}/plot_viscast_controlled_attribution_composite.py" \
  --ifs-comparison-dir "${IFS_COMPARISON_DIR}" \
  --endpoint-dir "${QCORE_ENDPOINT_DIR}" \
  --out-dir "${OUT_DIR}"

python "${IFS_DIR}/plot_viscast_source_full_metrics.py" \
  --metrics-csv "${SOURCE_FULL_METRICS_CSV}" \
  --out-dir "${OUT_DIR}"

python "${EVAL_DIR}/plot_viscast_long_tail_manuscript_composite.py" \
  --s1-csv "${S1_CSV}" \
  --sampling-csv "${SAMPLING_CSV}" \
  --loss-csv "${LOSS_CSV}" \
  --out-dir "${OUT_DIR}"

echo "viscast_main_figures=COMPLETE"
echo "output_dir=${OUT_DIR}"
