#!/bin/bash
#SBATCH -J static_rnn_precision_eval
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_precision_eval_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_precision_eval_%j.err
#SBATCH --exclusive

set -euo pipefail

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/run_static_rnn_precision_candidate_eval.py}"
MANIFEST="${MANIFEST:?Set MANIFEST to the precision-loss TSV manifest}"
SPLIT="${SPLIT:-val}"
OUT_DIR="${OUT_DIR:-${BASE}/static_rnn_precision_candidate_eval/${SPLIT}_${SLURM_JOB_ID}}"
DEVICE="${DEVICE:-cpu}"
RUN_EVENT_EVAL="${RUN_EVENT_EVAL:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --manifest "${MANIFEST}"
    --split "${SPLIT}"
    --base "${BASE}"
    --train_dir "${BASE}/train"
    --out_dir "${OUT_DIR}"
    --device "${DEVICE}"
)
if [[ "${RUN_EVENT_EVAL}" == "1" ]]; then
    CMD+=(--run_event_eval)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
