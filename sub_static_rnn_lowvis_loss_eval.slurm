#!/bin/bash
#SBATCH -J static_rnn_loss_eval
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_loss_eval_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_loss_eval_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Static-RNN loss-function evaluation job started at: $(date)"
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

SCRIPT_PATH="${SCRIPT_PATH:-/public/home/putianshu/vis_mlp/paper_eval/run_static_rnn_lowvis_loss_eval.py}"
CONFIG_JSON="${CONFIG_JSON:-/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json}"
BASE="${BASE:-/public/home/putianshu/vis_mlp}"
TRAIN_DIR="${TRAIN_DIR:-/public/home/putianshu/vis_mlp/train}"
OUT_DIR="${OUT_DIR:-static_rnn_loss_eval_results}"
LOSS_RUN_PREFIX="${LOSS_RUN_PREFIX:-${LOWVIS_RNN_RUN_PREFIX:-}}"
EXPERIMENTS="${EXPERIMENTS:-${LOWVIS_RNN_EXPERIMENTS:-0 1 2}}"
DEVICE="${DEVICE:-cpu}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-checkpoint}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --config_json "${CONFIG_JSON}"
    --base "${BASE}"
    --train_dir "${TRAIN_DIR}"
    --out_dir "${OUT_DIR}"
    --experiments "${EXPERIMENTS}"
    --device "${DEVICE}"
    --threshold_source "${THRESHOLD_SOURCE}"
)

if [[ -n "${LOSS_RUN_PREFIX}" ]]; then
    CMD+=(--loss_run_prefix "${LOSS_RUN_PREFIX}")
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

echo "Static-RNN loss-function evaluation job finished at: $(date)"
