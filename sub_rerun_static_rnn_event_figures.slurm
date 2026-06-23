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
echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Static-RNN event-figure rerun finished at: $(date)"
