#!/bin/bash
#SBATCH -J lowvis_traj_eval
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail
BASE="${LOWVIS_TRAJ_BASE:-/public/home/putianshu/vis_mlp}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
EVAL_DIR="${EVAL_DIR:-${BASE}/paper_eval}"
DATA_DIR="${LOWVIS_TRAJ_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_trajectory_1_48h_pm10_pm25_v1}"
: "${LOWVIS_TRAJ_CHECKPOINT:?Set LOWVIS_TRAJ_CHECKPOINT to a trained diffusion or Gaussian checkpoint}"
RUN_ID="${LOWVIS_TRAJ_EVAL_RUN_ID:-$(basename "${LOWVIS_TRAJ_CHECKPOINT}" .pt)}"
OUT_DIR="${LOWVIS_TRAJ_OUT_DIR:-${BASE}/trajectory_diffusion_results/${RUN_ID}}"
mkdir -p "${EVAL_DIR}/logs" "${OUT_DIR}"
cd "${EVAL_DIR}"
export LOWVIS_DIFFUSION_USE_DCU=1
export LOWVIS_DIFFUSION_TORCH_ENV="${LOWVIS_DIFFUSION_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
source "${TRAIN_DIR}/activate_lowvis_diffusion_runtime.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

python "${EVAL_DIR}/evaluate_lowvis_trajectory_diffusion.py" \
    --train-dir "${TRAIN_DIR}" \
    --data-dir "${DATA_DIR}" \
    --checkpoint "${LOWVIS_TRAJ_CHECKPOINT}" \
    --out-dir "${OUT_DIR}" \
    --members "${LOWVIS_TRAJ_MEMBERS:-50}" \
    --ddim-steps "${LOWVIS_TRAJ_DDIM_STEPS:-50}" \
    ${LOWVIS_TRAJ_EVAL_EXTRA_ARGS:-}
