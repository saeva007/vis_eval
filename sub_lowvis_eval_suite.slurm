#!/bin/bash
#SBATCH -J lowvis_eval_suite
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/lowvis_eval_suite_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/lowvis_eval_suite_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Low-vis eval suite job started at: $(date)"
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
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

mkdir -p /public/home/putianshu/vis_mlp/paper_eval/logs

SCRIPT_PATH="${SCRIPT_PATH:-/public/home/putianshu/vis_mlp/paper_eval/run_lowvis_eval_suite.py}"
CONFIG_JSON="${CONFIG_JSON:-/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json}"
TARGET="${TARGET:-main}"
MAIN_MODE="${MAIN_MODE:-all}"
MAIN_OUT_DIR="${MAIN_OUT_DIR:-}"
MATRIX_RUN_PREFIX="${MATRIX_RUN_PREFIX:-}"
MATRIX_EXPERIMENTS="${MATRIX_EXPERIMENTS:-}"
MATRIX_OUT_DIR="${MATRIX_OUT_DIR:-}"
DEVICE="${DEVICE:-}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
PLOTS="${PLOTS:-core}"
EXTRA_MAIN_ARGS="${EXTRA_MAIN_ARGS:-}"
EXTRA_MATRIX_ARGS="${EXTRA_MATRIX_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
    python "${SCRIPT_PATH}"
    --config_json "${CONFIG_JSON}"
    --target "${TARGET}"
    --main_mode "${MAIN_MODE}"
    --plots "${PLOTS}"
)

if [[ -n "${MAIN_OUT_DIR}" ]]; then
    CMD+=(--main_out_dir "${MAIN_OUT_DIR}")
fi
if [[ -n "${MATRIX_RUN_PREFIX}" ]]; then
    CMD+=(--matrix_run_prefix "${MATRIX_RUN_PREFIX}")
fi
if [[ -n "${MATRIX_EXPERIMENTS}" ]]; then
    CMD+=(--matrix_experiments "${MATRIX_EXPERIMENTS}")
fi
if [[ -n "${MATRIX_OUT_DIR}" ]]; then
    CMD+=(--matrix_out_dir "${MATRIX_OUT_DIR}")
fi
if [[ -n "${DEVICE}" ]]; then
    CMD+=(--device "${DEVICE}")
fi
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ -n "${EXTRA_MAIN_ARGS}" ]]; then
    CMD+=(--extra_main_args "${EXTRA_MAIN_ARGS}")
fi
if [[ -n "${EXTRA_MATRIX_ARGS}" ]]; then
    CMD+=(--extra_matrix_args "${EXTRA_MATRIX_ARGS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    CMD+=(--dry_run)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Low-vis eval suite job finished at: $(date)"
