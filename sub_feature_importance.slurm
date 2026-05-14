#!/bin/bash
#SBATCH -J feat_imp
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -o logs/feature_importance_%j.out
#SBATCH -e logs/feature_importance_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Feature-importance job started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

SCRIPT_PATH="/public/home/putianshu/vis_mlp/paper_eval/analyze_feature_importance_pm10_pm25.py"
CONFIG_JSON="${CONFIG_JSON:-/public/home/putianshu/vis_mlp/paper_eval/paper_eval_config.json}"

OUT_DIR="${OUT_DIR:-}"
DEVICE="${DEVICE:-auto}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Config JSON: ${CONFIG_JSON}"
echo "Output dir override: ${OUT_DIR:-<from config>}"

CMD=(
  python "${SCRIPT_PATH}"
  --config_json "${CONFIG_JSON}"
  --device "${DEVICE}"
)

if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${LIMIT_SAMPLES}" ]]; then
  CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ "${USE_CALIBRATION:-0}" == "1" ]]; then
  CMD+=(--use_calibration)
fi
if [[ "${CATALOG_ONLY:-0}" == "1" ]]; then
  CMD+=(--catalog_only)
fi
if [[ "${NO_PLOT:-0}" == "1" ]]; then
  CMD+=(--no_plot)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGV=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Job finished at: $(date)"
