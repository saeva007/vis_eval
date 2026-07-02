#!/bin/bash
#SBATCH -J static_rnn_event_story
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/paper_eval
#SBATCH -o /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_event_story_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/paper_eval/logs/static_rnn_event_story_%j.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/putianshu/miniconda3/envs/torch/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/plot_static_rnn_event_story.py}"
PER_SAMPLE_CSV="${PER_SAMPLE_CSV:?Set PER_SAMPLE_CSV to per_sample_eval.csv}"
EVENT_SUMMARY_CSV="${EVENT_SUMMARY_CSV:?Set EVENT_SUMMARY_CSV to event_case_summary.csv}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for the figure bundle}"
SHP_PATH="${SHP_PATH:-/public/home/putianshu/中华人民共和国/中华人民共和国.shp}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-Static MLP + GRU}"
DECISION_RULE="${DECISION_RULE:-argmax}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LOWVIS_GATE="${LOWVIS_GATE:-0.5}"
WINDOW_HOURS="${WINDOW_HOURS:-3}"

for path in "${PYTHON_BIN}" "${SCRIPT_PATH}" "${PER_SAMPLE_CSV}" "${EVENT_SUMMARY_CSV}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path not found: ${path}" >&2
        exit 2
    fi
done

mkdir -p "${OUT_DIR}" "${BASE}/paper_eval/logs"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
    "${PYTHON_BIN}" "${SCRIPT_PATH}"
    --per-sample-csv "${PER_SAMPLE_CSV}"
    --event-summary-csv "${EVENT_SUMMARY_CSV}"
    --out-dir "${OUT_DIR}"
    --shp-path "${SHP_PATH}"
    --candidate-label "${CANDIDATE_LABEL}"
    --decision-rule "${DECISION_RULE}"
    --temperature "${TEMPERATURE}"
    --lowvis-gate "${LOWVIS_GATE}"
    --window-hours "${WINDOW_HOURS}"
)

echo "Command: ${CMD[*]}"
"${CMD[@]}"
