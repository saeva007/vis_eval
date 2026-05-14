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

# 3. 脚本路径
SCRIPT_PATH="/public/home/putianshu/vis_mlp/paper_eval/run_paper_eval_pm10_pm25_journal.py"
EVAL_MODE="${EVAL_MODE:-all}"
S2_RUN_ID="${S2_RUN_ID:-exp_1778563813_pm10_more_temp_search_utc}"
OUT_DIR="${OUT_DIR:-paper_eval_results_pm10_pm25_journal_utc}"
HISTORY_PATHS="${HISTORY_PATHS:-}"
HISTORY_LABELS="${HISTORY_LABELS:-No FE values,Full FE}"
DATA_DIR="${DATA_DIR:-ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
DATA_48H_DIR="${DATA_48H_DIR:-ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime}"
CKPT_PATH="${CKPT_PATH:-checkpoints/${S2_RUN_ID}_S2_PhaseB_best_score.pt}"
SCALER_PATH="${SCALER_PATH:-checkpoints/robust_scaler_${S2_RUN_ID}_w12_dyn27_s2_48h_pm10.pkl}"
SEASON_TH_PATH="${SEASON_TH_PATH:-checkpoints/${S2_RUN_ID}_season_thresholds.pt}"
MODEL_PY="${MODEL_PY:-PMST_net_test_11_s2_pm10.py}"
IFS_VIS_NC="${IFS_VIS_NC:-VIS_IDW_KDTree_20250101_20251231.nc}"
LOCAL_TIME_OFFSET_HOURS="${LOCAL_TIME_OFFSET_HOURS:-8}"
FOG_TH="${FOG_TH:-0.34}"
MIST_TH="${MIST_TH:-0.56}"
LEAD_FOG_TH="${LEAD_FOG_TH:-${FOG_TH}}"
LEAD_MIST_TH="${LEAD_MIST_TH:-${MIST_TH}}"

# 4. 执行数据处理
echo "Running Python script: ${SCRIPT_PATH}"
echo "Eval mode: ${EVAL_MODE}"
echo "S2 run id: ${S2_RUN_ID}"
echo "Output dir: ${OUT_DIR}"
echo "Thresholds: fog=${FOG_TH}, mist=${MIST_TH}, lead_fog=${LEAD_FOG_TH}, lead_mist=${LEAD_MIST_TH}"

CMD=(
    python "${SCRIPT_PATH}"
    --mode "${EVAL_MODE}"
    --out_dir "${OUT_DIR}"
    --fog_th "${FOG_TH}"
    --mist_th "${MIST_TH}"
    --lead_fog_th "${LEAD_FOG_TH}"
    --lead_mist_th "${LEAD_MIST_TH}"
)

if [[ -n "${DATA_DIR}" ]]; then
    CMD+=(--data_dir "${DATA_DIR}")
fi
if [[ -n "${DATA_48H_DIR}" ]]; then
    CMD+=(--data_48h_dir "${DATA_48H_DIR}")
fi
if [[ -n "${CKPT_PATH}" ]]; then
    CMD+=(--ckpt_path "${CKPT_PATH}")
fi
if [[ -n "${SCALER_PATH}" ]]; then
    CMD+=(--scaler_path "${SCALER_PATH}")
fi
if [[ -n "${SEASON_TH_PATH}" ]]; then
    CMD+=(--season_th_path "${SEASON_TH_PATH}")
fi
if [[ -n "${MODEL_PY}" ]]; then
    CMD+=(--model_py "${MODEL_PY}")
fi
if [[ -n "${IFS_VIS_NC}" ]]; then
    CMD+=(--ifs_vis_nc "${IFS_VIS_NC}")
fi
if [[ -n "${LOCAL_TIME_OFFSET_HOURS}" ]]; then
    CMD+=(--local_time_offset_hours "${LOCAL_TIME_OFFSET_HOURS}")
fi

if [[ "${EVAL_MODE}" == "tables" && -n "${HISTORY_PATHS}" ]]; then
    CMD+=(--history_paths "${HISTORY_PATHS}" --history_labels "${HISTORY_LABELS}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Job finished at: $(date)"
