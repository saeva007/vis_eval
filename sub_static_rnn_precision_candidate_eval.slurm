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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:${LD_LIBRARY_PATH:-}
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic

RUNTIME_BOOTSTRAP="${RUNTIME_BOOTSTRAP:-${BASE}/ifs_baseline/activate_eval_torch_runtime.sh}"
if [[ ! -f "${RUNTIME_BOOTSTRAP}" ]]; then
    echo "ERROR: eval torch runtime bootstrap not found: ${RUNTIME_BOOTSTRAP}" >&2
    exit 2
fi
source "${RUNTIME_BOOTSTRAP}"

SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/run_static_rnn_precision_candidate_eval.py}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MANIFEST_RAW="${MANIFEST:?Set MANIFEST to the precision-loss TSV manifest}"
if [[ "${MANIFEST_RAW}" = /* ]]; then
    MANIFEST="${MANIFEST_RAW}"
elif [[ -f "${BASE}/train/${MANIFEST_RAW}" ]]; then
    MANIFEST="${BASE}/train/${MANIFEST_RAW}"
elif [[ -f "${MANIFEST_RAW}" ]]; then
    MANIFEST="$(readlink -f "${MANIFEST_RAW}")"
else
    echo "ERROR: manifest not found: ${MANIFEST_RAW}" >&2
    echo "Checked: ${BASE}/train/${MANIFEST_RAW}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
SPLIT="${SPLIT:-val}"
OUT_DIR="${OUT_DIR:-${BASE}/static_rnn_precision_candidate_eval/${SPLIT}_${SLURM_JOB_ID}}"
DEVICE="${DEVICE:-cpu}"
RUN_EVENT_EVAL="${RUN_EVENT_EVAL:-1}"
DATA_DIR="${DATA_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    "${PYTHON_BIN}" "${SCRIPT_PATH}"
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
if [[ -n "${DATA_DIR}" ]]; then
    CMD+=(--data_dir "${DATA_DIR}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
