#!/bin/bash
#SBATCH -J static_rnn_eval
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_eval_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_eval_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Static-RNN evaluation job started at: $(date)"
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

SCRIPT_PATH="${SCRIPT_PATH:-/public/home/putianshu/vis_mlp/paper_eval/run_static_rnn_lowvis_eval_journal.py}"
CONFIG_JSON="${CONFIG_JSON:-/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json}"
MODE="${MODE:-main}"
BASE="${BASE:-/public/home/putianshu/vis_mlp}"
TRAIN_DIR="${TRAIN_DIR:-/public/home/putianshu/vis_mlp/train}"
OUT_DIR="${OUT_DIR:-}"
MAIN_RUN_ID="${MAIN_RUN_ID:-}"
MAIN_CKPT="${MAIN_CKPT:-}"
MATRIX_RUN_PREFIX="${MATRIX_RUN_PREFIX:-}"
MATRIX_EXPERIMENTS="${MATRIX_EXPERIMENTS:-}"
CKPT_PATH="${CKPT_PATH:-}"
VARIANT_ID="${VARIANT_ID:-}"
DEVICE="${DEVICE:-auto}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
PLOTS="${PLOTS:-all}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --config_json "${CONFIG_JSON}"
    --mode "${MODE}"
    --base "${BASE}"
    --train_dir "${TRAIN_DIR}"
    --device "${DEVICE}"
    --plots "${PLOTS}"
)

if [[ -n "${OUT_DIR}" ]]; then
    CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${MAIN_RUN_ID}" ]]; then
    CMD+=(--main_run_id "${MAIN_RUN_ID}")
fi
if [[ -n "${MAIN_CKPT}" ]]; then
    CMD+=(--main_ckpt "${MAIN_CKPT}")
fi
if [[ -n "${MATRIX_RUN_PREFIX}" ]]; then
    CMD+=(--matrix_run_prefix "${MATRIX_RUN_PREFIX}")
fi
if [[ -n "${MATRIX_EXPERIMENTS}" ]]; then
    CMD+=(--matrix_experiments "${MATRIX_EXPERIMENTS}")
fi
if [[ -n "${CKPT_PATH}" ]]; then
    CMD+=(--ckpt_path "${CKPT_PATH}")
fi
if [[ -n "${VARIANT_ID}" ]]; then
    CMD+=(--variant_id "${VARIANT_ID}")
fi
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Static-RNN evaluation job finished at: $(date)"
