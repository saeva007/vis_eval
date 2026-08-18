#!/bin/bash
#SBATCH -J static_rnn_event_rerun
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_event_rerun_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_event_rerun_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Static-RNN event-figure rerun started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p /public/home/putianshu/vis_mlp/paper_eval/logs

SCRIPT_PATH="${SCRIPT_PATH:-/public/home/putianshu/vis_mlp/paper_eval/rerun_static_rnn_event_figures.py}"
BASE="${BASE:-/public/home/putianshu/vis_mlp}"
EVAL_DIR="${EVAL_DIR:?Set EVAL_DIR to the existing Static-RNN evaluation directory.}"
OUT_DIR="${OUT_DIR:-${EVAL_DIR}/event_rerun_compute_${SLURM_JOB_ID}}"
EVENT_SUMMARY="${EVENT_SUMMARY:-}"
REPLACE_EVENT="${REPLACE_EVENT:-}"
WINDOW_HOURS="${WINDOW_HOURS:-3}"
EVENT_ENV_SOURCE="${EVENT_ENV_SOURCE:-grid}"
EVENT_ENV_MAX_EVENTS="${EVENT_ENV_MAX_EVENTS:-3}"
EVENT_ENV_INCLUDE_CSI="${EVENT_ENV_INCLUDE_CSI:-1}"
EVENT_ENV_WITH_PANGU="${EVENT_ENV_WITH_PANGU:-0}"
EVENT_ENV_WITH_SOURCE_MODELS="${EVENT_ENV_WITH_SOURCE_MODELS:-0}"
EVENT_ENV_OVERLAP_EVAL="${EVENT_ENV_OVERLAP_EVAL:-${BASE}/paper_eval_results_pm10_pm25_journal/overlap_forecast_source/per_sample_paired_eval.csv}"
EVENT_ENV_PANGU_EVAL_ROOT="${EVENT_ENV_PANGU_EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/qcore_t925_fair_formal_v1_20260721}"
EVENT_ENV_PANGU_SEEDS="${EVENT_ENV_PANGU_SEEDS:-42:2025:20260702}"
EVENT_ENV_PANGU_TAG="${EVENT_ENV_PANGU_TAG:-pangu2025_q_core_t925_no_rh2m}"
ENVIRONMENT_GRID_ONLY="${ENVIRONMENT_GRID_ONLY:-0}"
ENVIRONMENT_GRIDS_ONLY="${ENVIRONMENT_GRIDS_ONLY:-0}"
ENVIRONMENT_EVENT_RANK="${ENVIRONMENT_EVENT_RANK:-1}"
SHP_PATH="${SHP_PATH:-}"
EVENT_ENV_TIANJI_TEMPLATE="${EVENT_ENV_TIANJI_TEMPLATE:-}"
if [[ -z "${EVENT_ENV_TIANJI_TEMPLATE}" ]]; then
    EVENT_ENV_TIANJI_TEMPLATE='/tj01/sd3op/userpp/pp_data/{init_yyyymmddhh}/stage26Q/multi_model_sources/{init_yyyymmddhh}/{variable}.nc'
fi
EVENT_ENV_RH2M_VAR="${EVENT_ENV_RH2M_VAR:-rh2m}"
EVENT_ENV_PM10_DIR="${EVENT_ENV_PM10_DIR:-pm10_data}"
EVENT_ENV_PM10_VAR="${EVENT_ENV_PM10_VAR:-pm10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --eval_dir "${EVAL_DIR}"
    --out_dir "${OUT_DIR}"
    --base "${BASE}"
    --window_hours "${WINDOW_HOURS}"
    --event_env_source "${EVENT_ENV_SOURCE}"
    --event_env_max_events "${EVENT_ENV_MAX_EVENTS}"
    --event_env_tianji_template "${EVENT_ENV_TIANJI_TEMPLATE}"
    --event_env_rh2m_var "${EVENT_ENV_RH2M_VAR}"
    --event_env_pm10_dir "${EVENT_ENV_PM10_DIR}"
    --event_env_pm10_var "${EVENT_ENV_PM10_VAR}"
)

if [[ -n "${EVENT_SUMMARY}" ]]; then
    CMD+=(--event_summary "${EVENT_SUMMARY}")
fi
if [[ -n "${SHP_PATH}" ]]; then
    CMD+=(--shp_path "${SHP_PATH}")
fi
if [[ -n "${REPLACE_EVENT}" ]]; then
    CMD+=(--replace_event "${REPLACE_EVENT}")
fi
if [[ "${EVENT_ENV_INCLUDE_CSI}" == "1" ]]; then
    CMD+=(--event_env_include_csi)
fi
if [[ "${EVENT_ENV_WITH_PANGU}" == "1" ]]; then
    CMD+=(
        --event_env_with_pangu
        --event_env_pangu_eval_root "${EVENT_ENV_PANGU_EVAL_ROOT}"
        --event_env_pangu_seeds "${EVENT_ENV_PANGU_SEEDS}"
        --event_env_pangu_tag "${EVENT_ENV_PANGU_TAG}"
    )
fi
if [[ "${EVENT_ENV_WITH_SOURCE_MODELS}" == "1" ]]; then
    CMD+=(--event_env_with_source_models)
    if [[ -n "${EVENT_ENV_OVERLAP_EVAL}" ]]; then
        CMD+=(--event_env_overlap_eval "${EVENT_ENV_OVERLAP_EVAL}")
    fi
fi
if [[ "${ENVIRONMENT_GRID_ONLY}" == "1" ]]; then
    CMD+=(--environment_grid_only --environment_event_rank "${ENVIRONMENT_EVENT_RANK}")
fi
if [[ "${ENVIRONMENT_GRIDS_ONLY}" == "1" ]]; then
    CMD+=(--environment_grids_only)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "EVAL_DIR=${EVAL_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "REPLACE_EVENT=${REPLACE_EVENT:-<none>}"
echo "EVENT_ENV_SOURCE=${EVENT_ENV_SOURCE}"
echo "EVENT_ENV_MAX_EVENTS=${EVENT_ENV_MAX_EVENTS}"
echo "EVENT_ENV_INCLUDE_CSI=${EVENT_ENV_INCLUDE_CSI}"
echo "EVENT_ENV_WITH_PANGU=${EVENT_ENV_WITH_PANGU}"
echo "EVENT_ENV_WITH_SOURCE_MODELS=${EVENT_ENV_WITH_SOURCE_MODELS}"
echo "EVENT_ENV_OVERLAP_EVAL=${EVENT_ENV_OVERLAP_EVAL:-<default>}"
echo "EVENT_ENV_PANGU_EVAL_ROOT=${EVENT_ENV_PANGU_EVAL_ROOT}"
echo "ENVIRONMENT_GRID_ONLY=${ENVIRONMENT_GRID_ONLY}"
echo "ENVIRONMENT_GRIDS_ONLY=${ENVIRONMENT_GRIDS_ONLY}"
echo "ENVIRONMENT_EVENT_RANK=${ENVIRONMENT_EVENT_RANK}"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Static-RNN event-figure rerun finished at: $(date)"
