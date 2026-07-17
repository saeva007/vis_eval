#!/bin/bash
#SBATCH -J p13_seed_mean
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/p13_seed_mean_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/p13_seed_mean_%j.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
MANIFEST="${MANIFEST:?Set MANIFEST to the full P13 manifest}"
MAIN_EVAL_DIR="${MAIN_EVAL_DIR:?Set MAIN_EVAL_DIR to the three-member test evaluation directory}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new ensemble reuse directory}"
FORECAST48_EVAL_DIR="${FORECAST48_EVAL_DIR:-}"
CANDIDATE_ID="${CANDIDATE_ID:-p13}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-42,314,2718}"
DATA_48H_DIR="${DATA_48H_DIR:-ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime}"
IFS_48H_NC="${IFS_48H_NC:-IFS_VIS_0_48h_stations_2025_00_12.nc}"
IFS_48H_VAR="${IFS_48H_VAR:-VIS_ifs}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/prepare_static_rnn_seed_mean_for_eval.py}"

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
source "${RUNTIME_BOOTSTRAP}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CMD=(
    "${PYTHON_BIN}" "${SCRIPT_PATH}"
    --manifest "${MANIFEST}"
    --main-eval-dir "${MAIN_EVAL_DIR}"
    --candidate-id "${CANDIDATE_ID}"
    --expected-seeds "${EXPECTED_SEEDS}"
    --required-stage full
    --out-dir "${OUT_DIR}"
    --base "${BASE}"
    --data-48h-dir "${DATA_48H_DIR}"
    --ifs-48h-nc "${IFS_48H_NC}"
    --ifs-48h-var "${IFS_48H_VAR}"
)
if [[ -n "${FORECAST48_EVAL_DIR}" ]]; then
    CMD+=(--forecast48-eval-dir "${FORECAST48_EVAL_DIR}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
