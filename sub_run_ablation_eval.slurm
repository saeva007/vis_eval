#!/bin/bash
#SBATCH -J ablation_eval
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -o logs/ablation_eval_%j.out
#SBATCH -e logs/ablation_eval_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Ablation eval job started at: $(date)"
echo "Node: $(hostname)"

# 1. 环境加载，与 sub_run_eval.slurm 对齐
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

# 2. CPU 数值库线程设置
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

# 3. 脚本路径与参数
SCRIPT_PATH="/public/home/putianshu/vis_mlp/paper_eval/run_feature_engineering_ablation_journal.py"
MODE="${MODE:-all}"
BASE_DIR="${BASE_DIR:-/public/home/putianshu/vis_mlp}"
S2_RUN_ID="${S2_RUN_ID:-exp_1778563813_pm10_more_temp_search_utc}"
SPEC_CSV="${SPEC_CSV:-${BASE_DIR}/feature_engineering_ablation_spec_${S2_RUN_ID}.csv}"
OUT_DIR="${OUT_DIR:-paper_eval_results_pm10_pm25_journal/feature_engineering_ablation}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REFRESH_SPEC="${REFRESH_SPEC:-0}"

if [[ "${REFRESH_SPEC}" == "1" || ! -f "${SPEC_CSV}" ]]; then
    echo "Writing current template: ${SPEC_CSV}"
    python "${SCRIPT_PATH}" --write_template "${SPEC_CSV}"
fi

echo "Running Python script: ${SCRIPT_PATH}"
echo "MODE=${MODE}"
echo "BASE_DIR=${BASE_DIR}"
echo "S2_RUN_ID=${S2_RUN_ID}"
echo "SPEC_CSV=${SPEC_CSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "REFRESH_SPEC=${REFRESH_SPEC}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

python "${SCRIPT_PATH}" \
    --mode "${MODE}" \
    --base "${BASE_DIR}" \
    --spec_csv "${SPEC_CSV}" \
    --out_dir "${OUT_DIR}" \
    ${EXTRA_ARGS}

echo "Job finished at: $(date)"
