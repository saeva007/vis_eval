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

# 3. 脚本路径与集中配置
SCRIPT_PATH="/public/home/putianshu/vis_mlp/paper_eval/run_feature_engineering_ablation_journal.py"
CONFIG_JSON="${CONFIG_JSON:-/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json}"
MODE="${MODE:-}"
OUT_DIR="${OUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REFRESH_SPEC="${REFRESH_SPEC:-0}"

echo "Running Python script: ${SCRIPT_PATH}"
echo "CONFIG_JSON=${CONFIG_JSON}"
echo "MODE override=${MODE:-<from config>}"
echo "OUT_DIR override=${OUT_DIR:-<from config>}"
echo "REFRESH_SPEC=${REFRESH_SPEC}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

CMD=(
    python "${SCRIPT_PATH}"
    --config_json "${CONFIG_JSON}"
)

if [[ -n "${MODE}" ]]; then
    CMD+=(--mode "${MODE}")
fi
if [[ -n "${OUT_DIR}" ]]; then
    CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ "${REFRESH_SPEC}" == "1" ]]; then
    CMD+=(--refresh_spec)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Job finished at: $(date)"
