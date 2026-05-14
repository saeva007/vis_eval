#!/bin/bash
#SBATCH -J eval
#SBATCH -p kshdexclu01          # 使用你指定的队列节点
#SBATCH -N 1                    # 数据处理只需单节点
#SBATCH --ntasks-per-node=1     # 单个任务
#SBATCH --cpus-per-task=32      # 充分利用 32 个 CPU 核心
#SBATCH --gres=dcu:4            # 队列按整节点 4 DCU 申请
#SBATCH --mem=0                 # 0 表示申请节点上的所有可用内存（防止数据量大时 OOM）
#SBATCH -t 48:00:00             # 数据处理时间限制
#SBATCH -o logs/process_%j.out
#SBATCH -e logs/process_%j.err
#SBATCH --exclusive             # 独占节点，避免与其他任务抢占内存和 IO

set -euo pipefail

echo "Data Processing Job started at: $(date)"
echo "Node: $(hostname)"

# 1. 环境加载 (与你的训练环境对齐)
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
#source /public/home/jarvis226/miniconda3/bin/activate
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 2. 核心性能环境变量优化
export PYTHONUNBUFFERED=1
# 限制底层线性代数库的线程数，对齐申请的 32 核心，极大加速 numpy 矩阵运算
export OMP_NUM_THREADS=32       
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

# 3. 脚本路径与集中配置
SCRIPT_PATH="/public/home/putianshu/vis_mlp/paper_eval/run_paper_eval_pm10_pm25_journal.py"
CONFIG_JSON="${CONFIG_JSON:-/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json}"
EVAL_MODE="${EVAL_MODE:-}"
OUT_DIR="${OUT_DIR:-}"
HISTORY_PATHS="${HISTORY_PATHS:-}"
HISTORY_LABELS="${HISTORY_LABELS:-}"
DEVICE="${DEVICE:-}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# 4. 执行数据处理
echo "Running Python script: ${SCRIPT_PATH}"
echo "Config JSON: ${CONFIG_JSON}"
echo "Eval mode override: ${EVAL_MODE:-<from config>}"
echo "Output dir override: ${OUT_DIR:-<from config>}"

CMD=(
    python "${SCRIPT_PATH}"
    --config_json "${CONFIG_JSON}"
)

if [[ -n "${EVAL_MODE}" ]]; then
    CMD+=(--mode "${EVAL_MODE}")
fi
if [[ -n "${OUT_DIR}" ]]; then
    CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${DEVICE}" ]]; then
    CMD+=(--device "${DEVICE}")
fi
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ -n "${HISTORY_PATHS}" ]]; then
    CMD+=(--history_paths "${HISTORY_PATHS}")
fi
if [[ -n "${HISTORY_LABELS}" ]]; then
    CMD+=(--history_labels "${HISTORY_LABELS}")
fi
if [[ "${SKIP_OVERLAP_BOOTSTRAP:-0}" == "1" ]]; then
    CMD+=(--skip_overlap_bootstrap)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Job finished at: $(date)"
