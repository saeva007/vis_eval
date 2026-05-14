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

OUT_DIR="${OUT_DIR:-paper_eval_results_pm10_pm25_journal_utc/feature_importance}"
DATA_DIR="${DATA_DIR:-ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
CKPT_PATH="${CKPT_PATH:-checkpoints/exp_1778563813_pm10_more_temp_search_utc_S2_PhaseB_best_score.pt}"
SCALER_PATH="${SCALER_PATH:-checkpoints/robust_scaler_exp_1778563813_pm10_more_temp_search_utc_w12_dyn27_s2_48h_pm10.pkl}"
SEASON_TH_PATH="${SEASON_TH_PATH:-checkpoints/exp_1778563813_pm10_more_temp_search_utc_season_thresholds.pt}"
MODEL_PY="${MODEL_PY:-}"
DEVICE="${DEVICE:-auto}"
REPEATS="${REPEATS:-3}"
MAX_FOG="${MAX_FOG:-8000}"
MAX_MIST="${MAX_MIST:-8000}"
MAX_CLEAR="${MAX_CLEAR:-20000}"
SORT_METRIC="${SORT_METRIC:-low_vis_f2}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
FOG_TH="${FOG_TH:-0.34}"
MIST_TH="${MIST_TH:-0.56}"

CMD=(
  python "${SCRIPT_PATH}"
  --out_dir "${OUT_DIR}"
  --data_dir "${DATA_DIR}"
  --ckpt_path "${CKPT_PATH}"
  --scaler_path "${SCALER_PATH}"
  --season_th_path "${SEASON_TH_PATH}"
  --fog_th "${FOG_TH}"
  --mist_th "${MIST_TH}"
  --device "${DEVICE}"
  --repeats "${REPEATS}"
  --max_fog "${MAX_FOG}"
  --max_mist "${MAX_MIST}"
  --max_clear "${MAX_CLEAR}"
  --sort_metric "${SORT_METRIC}"
  --limit_samples "${LIMIT_SAMPLES}"
)

if [[ -n "${MODEL_PY}" ]]; then
  CMD+=(--model_py "${MODEL_PY}")
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

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Job finished at: $(date)"
