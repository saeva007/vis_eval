#!/bin/bash
#SBATCH -J sampling_curve_merge
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/sampling_curve_merge_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/sampling_curve_merge_%j.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/merge_static_rnn_sampling_curve_results.py}"
LEGACY_EVAL_DIR="${LEGACY_EVAL_DIR:-auto}"
FILL_EVAL_DIR="${FILL_EVAL_DIR:?FILL_EVAL_DIR is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

module purge
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${BASE}/paper_eval/logs"

echo "sampling_curve_merge_preflight=OK"
echo "legacy_eval_dir=${LEGACY_EVAL_DIR}"
echo "fill_eval_dir=${FILL_EVAL_DIR}"
echo "out_dir=${OUT_DIR}"

python "${SCRIPT_PATH}" \
  --base "${BASE}" \
  --legacy-eval-dir "${LEGACY_EVAL_DIR}" \
  --fill-eval-dir "${FILL_EVAL_DIR}" \
  --out-dir "${OUT_DIR}"
