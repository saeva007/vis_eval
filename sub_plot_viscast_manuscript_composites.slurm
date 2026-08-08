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
MAIN_EVAL_DIR="${MAIN_EVAL_DIR:-auto}"
S1_CSV="${S1_CSV:-auto}"
SAMPLING_CSV="${SAMPLING_CSV:-auto}"
LOSS_CSV="${LOSS_CSV:-auto}"
UPPER_AIR_DIR="${UPPER_AIR_DIR:-auto}"
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

for path in \
    "${EVAL_DIR}/plot_viscast_baseline_manuscript_composite.py" \
    "${EVAL_DIR}/plot_viscast_long_tail_manuscript_composite.py" \
    "${IFS_DIR}/plot_qcore_t925_manuscript_composite.py" \
    "${MAIN_EVAL_DIR}/station_model_vs_ifs_metrics.csv" \
    "${QCORE_ANALYSIS_DIR}/hybrid_exact_shapley_effects.csv" \
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
echo "s1_csv=${S1_CSV}"
echo "sampling_csv=${SAMPLING_CSV}"
echo "loss_csv=${LOSS_CSV}"
echo "output_dir=${OUT_DIR}"

python "${EVAL_DIR}/plot_viscast_baseline_manuscript_composite.py" \
  --eval-dir "${MAIN_EVAL_DIR}" \
  --shp-path "${SHP_PATH}" \
  --out-dir "${OUT_DIR}"

python "${IFS_DIR}/plot_qcore_t925_manuscript_composite.py" \
  --analysis-dir "${QCORE_ANALYSIS_DIR}" \
  --upper-air-dir "${UPPER_AIR_DIR}" \
  --out-dir "${OUT_DIR}"

python "${EVAL_DIR}/plot_viscast_long_tail_manuscript_composite.py" \
  --s1-csv "${S1_CSV}" \
  --sampling-csv "${SAMPLING_CSV}" \
  --loss-csv "${LOSS_CSV}" \
  --out-dir "${OUT_DIR}"

echo "viscast_main_figures=COMPLETE"
echo "output_dir=${OUT_DIR}"
