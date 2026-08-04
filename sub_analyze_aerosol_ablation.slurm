#!/bin/bash
#SBATCH -J aerosol_ablation_analysis
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/aerosol_ablation_analysis_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/aerosol_ablation_analysis_%j.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/analyze_aerosol_ablation.py}"
FULL_VAL_DIR="${FULL_VAL_DIR:?Set FULL_VAL_DIR}"
NO_PM_VAL_DIR="${NO_PM_VAL_DIR:?Set NO_PM_VAL_DIR}"
FULL_TEST_DIR="${FULL_TEST_DIR:?Set FULL_TEST_DIR}"
NO_PM_TEST_DIR="${NO_PM_TEST_DIR:?Set NO_PM_TEST_DIR}"
FULL_DATA_DIR="${FULL_DATA_DIR:?Set FULL_DATA_DIR}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR}"
TARGET_FPR="${TARGET_FPR:-0.04}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260804}"
EVENT_TIMES="${EVENT_TIMES:-2025-02-27 22:00:00;2025-09-28 23:00:00;2025-10-30 22:00:00}"
EVENT_WINDOW_HOURS="${EVENT_WINDOW_HOURS:-3}"
MIN_CELL_LOWVIS="${MIN_CELL_LOWVIS:-30}"

mkdir -p "${BASE}/paper_eval/logs"
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:${LD_LIBRARY_PATH:-}

RUNTIME_BOOTSTRAP="${RUNTIME_BOOTSTRAP:-${BASE}/ifs_baseline/activate_eval_torch_runtime.sh}"
if [[ ! -f "${RUNTIME_BOOTSTRAP}" ]]; then
    echo "ERROR: runtime bootstrap not found: ${RUNTIME_BOOTSTRAP}" >&2
    exit 2
fi
source "${RUNTIME_BOOTSTRAP}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
    "${PYTHON_BIN}" "${SCRIPT_PATH}"
    --full-val-dir "${FULL_VAL_DIR}"
    --no-pm-val-dir "${NO_PM_VAL_DIR}"
    --full-test-dir "${FULL_TEST_DIR}"
    --no-pm-test-dir "${NO_PM_TEST_DIR}"
    --full-data-dir "${FULL_DATA_DIR}"
    --out-dir "${OUT_DIR}"
    --target-fpr "${TARGET_FPR}"
    --bootstrap-iters "${BOOTSTRAP_ITERS}"
    --bootstrap-seed "${BOOTSTRAP_SEED}"
    --event-times "${EVENT_TIMES}"
    --event-window-hours "${EVENT_WINDOW_HOURS}"
    --min-cell-lowvis "${MIN_CELL_LOWVIS}"
)

echo "aerosol_ablation_analysis_preflight=OK"
echo "full_val_dir=${FULL_VAL_DIR}"
echo "no_pm_val_dir=${NO_PM_VAL_DIR}"
echo "full_test_dir=${FULL_TEST_DIR}"
echo "no_pm_test_dir=${NO_PM_TEST_DIR}"
echo "full_data_dir=${FULL_DATA_DIR}"
echo "out_dir=${OUT_DIR}"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
