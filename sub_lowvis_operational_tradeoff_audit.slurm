#!/bin/bash
#SBATCH -J lowvis_op_audit
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/lowvis_op_audit_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/lowvis_op_audit_%j.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
PAPER_EVAL="${PAPER_EVAL:-${BASE}/paper_eval}"
EVAL_DIR="${EVAL_DIR:?Set EVAL_DIR to the frozen test paper-evaluation directory}"
SELECTION_DIR="${SELECTION_DIR:-}"
OUT_DIR="${OUT_DIR:-${EVAL_DIR}/operational_tradeoff_audit}"

mkdir -p "${PAPER_EVAL}/logs"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

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
# shellcheck disable=SC1090
source "${RUNTIME_BOOTSTRAP}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

test -s "${EVAL_DIR}/per_sample_eval.csv" || {
    echo "ERROR: missing ${EVAL_DIR}/per_sample_eval.csv" >&2
    exit 2
}

args=(
    --eval-dir "${EVAL_DIR}"
    --out-dir "${OUT_DIR}"
    --threshold-points "${THRESHOLD_POINTS:-1001}"
    --calibration-bins "${CALIBRATION_BINS:-15}"
    --activity-fraction "${ACTIVITY_FRACTION:-0.20}"
    --precision-targets "${PRECISION_TARGETS:-0.20,0.25,0.30,0.40}"
)

if [ -n "${SELECTION_DIR}" ]; then
    test -s "${SELECTION_DIR}/per_sample_eval.csv" || {
        echo "ERROR: missing ${SELECTION_DIR}/per_sample_eval.csv" >&2
        exit 2
    }
    args+=(--selection-dir "${SELECTION_DIR}")
fi

echo "EVAL_DIR=${EVAL_DIR}"
echo "SELECTION_DIR=${SELECTION_DIR:-none}"
echo "OUT_DIR=${OUT_DIR}"
"${PYTHON_BIN}" "${PAPER_EVAL}/analyze_lowvis_operational_tradeoffs.py" "${args[@]}"
