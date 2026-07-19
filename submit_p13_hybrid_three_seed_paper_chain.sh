#!/bin/bash
# Submit the hybrid P13 three-seed inference -> mean-softmax -> paper-figure
# chain.  The public entrypoint detaches a worker through nohup so an SSH
# disconnect cannot interrupt the short sequence of sbatch calls or discard
# its exported variables.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
PAPER_EVAL="${PAPER_EVAL:-${BASE}/paper_eval}"

if [ "${P13_CHAIN_WORKER:-0}" != "1" ]; then
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    BUNDLE_ID="${BUNDLE_ID:-p13_hybrid_three_seed_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${PAPER_EVAL}/logs"
    LAUNCH_LOG="${LAUNCH_LOG:-${PAPER_EVAL}/logs/${BUNDLE_ID}_launcher.log}"
    HANDLE_FILE="${PAPER_EVAL}/logs/${BUNDLE_ID}_launcher.env"
    STATE_FILE="${STATE_FILE:-${PAPER_EVAL}/logs/${BUNDLE_ID}_submission.env}"

    nohup env \
        P13_CHAIN_WORKER=1 \
        BASE="${BASE}" \
        PAPER_EVAL="${PAPER_EVAL}" \
        BUNDLE_ID="${BUNDLE_ID}" \
        STATE_FILE="${STATE_FILE}" \
        bash "${SCRIPT_PATH}" </dev/null >"${LAUNCH_LOG}" 2>&1 &
    WORKER_PID=$!

    HANDLE_TMP="${HANDLE_FILE}.tmp.$$"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'WORKER_PID=%q\n' "${WORKER_PID}"
        printf 'LAUNCH_LOG=%q\n' "${LAUNCH_LOG}"
        printf 'STATE_FILE=%q\n' "${STATE_FILE}"
    } > "${HANDLE_TMP}"
    mv -f "${HANDLE_TMP}" "${HANDLE_FILE}"

    echo "Detached P13 paper chain; an SSH disconnect is now safe."
    echo "BUNDLE_ID=${BUNDLE_ID}"
    echo "WORKER_PID=${WORKER_PID}"
    echo "LAUNCH_LOG=${LAUNCH_LOG}"
    echo "STATE_FILE=${STATE_FILE}"
    echo "HANDLE_FILE=${HANDLE_FILE}"
    exit 0
fi

CKPT_DIR="${CKPT_DIR:-${BASE}/checkpoints}"
RUN42="${RUN42:-exp_20260717_203259_p13_sampling_calibration_full_p13_seed42_2_proposed_rare_event_focal}"
RUN314="${RUN314:-exp_20260717_203259_p13_sampling_calibration_full_p13_seed314_2_proposed_rare_event_focal}"
RUN2718="${RUN2718:-exp_20260718_195035_p13_sampling_calibration_full_p13_seed2718_2_proposed_rare_event_focal}"

CKPT42="${CKPT42:-${CKPT_DIR}/${RUN42}_S2_PhaseD_best_score.pt}"
CKPT314="${CKPT314:-${CKPT_DIR}/${RUN314}_S2_PhaseD_best_score.pt}"
CKPT2718="${CKPT2718:-${CKPT_DIR}/${RUN2718}_S2_PhaseD_best_score.pt}"

BUNDLE_ID="${BUNDLE_ID:-p13_hybrid_three_seed_$(date +%Y%m%d_%H%M%S)}"
HYBRID_MANIFEST="${HYBRID_MANIFEST:-${BASE}/train/logs/${BUNDLE_ID}_manifest.tsv}"
MAIN_TEST_DIR="${MAIN_TEST_DIR:-${BASE}/static_rnn_precision_candidate_eval/${BUNDLE_ID}_test}"
H48_TEST_DIR="${H48_TEST_DIR:-${BASE}/static_rnn_precision_candidate_eval/${BUNDLE_ID}_48h_members}"
ENSEMBLE_REUSE_DIR="${ENSEMBLE_REUSE_DIR:-${BASE}/static_rnn_precision_candidate_eval/${BUNDLE_ID}_mean_argmax}"
PAPER_OUT="${PAPER_OUT:-${BASE}/static_rnn_eval_results/${BUNDLE_ID}_paper_figures}"
STATE_FILE="${STATE_FILE:-${PAPER_EVAL}/logs/${BUNDLE_ID}_submission.env}"

MAIN_TEST_JOB=""
H48_TEST_JOB=""
MEAN_JOB=""
PLOT_JOB=""
SUBMISSION_STATUS="starting"

mkdir -p "${PAPER_EVAL}/logs" "$(dirname "${HYBRID_MANIFEST}")"
cd "${PAPER_EVAL}"

write_state() {
    local tmp="${STATE_FILE}.tmp.$$"
    {
        printf 'SUBMISSION_STATUS=%q\n' "${SUBMISSION_STATUS}"
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'HYBRID_MANIFEST=%q\n' "${HYBRID_MANIFEST}"
        printf 'MAIN_TEST_DIR=%q\n' "${MAIN_TEST_DIR}"
        printf 'H48_TEST_DIR=%q\n' "${H48_TEST_DIR}"
        printf 'ENSEMBLE_REUSE_DIR=%q\n' "${ENSEMBLE_REUSE_DIR}"
        printf 'PAPER_OUT=%q\n' "${PAPER_OUT}"
        printf 'MAIN_TEST_JOB=%q\n' "${MAIN_TEST_JOB}"
        printf 'H48_TEST_JOB=%q\n' "${H48_TEST_JOB}"
        printf 'MEAN_JOB=%q\n' "${MEAN_JOB}"
        printf 'PLOT_JOB=%q\n' "${PLOT_JOB}"
    } > "${tmp}"
    mv -f "${tmp}" "${STATE_FILE}"
}

on_exit() {
    local status=$?
    trap - EXIT
    if [ "${status}" -ne 0 ]; then
        SUBMISSION_STATUS="failed_rc_${status}"
    fi
    write_state || true
    exit "${status}"
}
trap on_exit EXIT

check_checkpoint() {
    local seed="$1"
    local checkpoint="$2"
    printf '[preflight] seed=%s checkpoint=%s\n' "${seed}" "${checkpoint}"
    if [ ! -s "${checkpoint}" ]; then
        printf 'ERROR: missing or empty checkpoint for seed=%s: %s\n' \
            "${seed}" "${checkpoint}" >&2
        return 2
    fi
    printf '[preflight] seed=%s checkpoint=OK\n' "${seed}"
}

write_state
check_checkpoint 42 "${CKPT42}"
check_checkpoint 314 "${CKPT314}"
check_checkpoint 2718 "${CKPT2718}"

for target in \
    "${HYBRID_MANIFEST}" \
    "${MAIN_TEST_DIR}" \
    "${H48_TEST_DIR}" \
    "${ENSEMBLE_REUSE_DIR}" \
    "${PAPER_OUT}"; do
    if [ -e "${target}" ]; then
        echo "ERROR: refusing to overwrite existing target: ${target}" >&2
        exit 2
    fi
done

{
    printf 'candidate_id\tcandidate_label\tseed\tstage\trun_prefix\trun_id\ts2_checkpoint\n'
    printf 'p13\tp3_sampling_calibration_eta030\t42\tfull\t%s\t%s\t%s\n' \
        "${RUN42%_2_proposed_rare_event_focal}" "${RUN42}" "${CKPT42}"
    printf 'p13\tp3_sampling_calibration_eta030\t314\tfull\t%s\t%s\t%s\n' \
        "${RUN314%_2_proposed_rare_event_focal}" "${RUN314}" "${CKPT314}"
    printf 'p13\tp3_sampling_calibration_eta030\t2718\tfull\t%s\t%s\t%s\n' \
        "${RUN2718%_2_proposed_rare_event_focal}" "${RUN2718}" "${CKPT2718}"
} > "${HYBRID_MANIFEST}"

SUBMISSION_STATUS="submitting_member_jobs"
write_state
unset EXTRA_ARGS || true

MAIN_TEST_SUBMISSION="$(
    sbatch --parsable \
        --export=ALL,MANIFEST=${HYBRID_MANIFEST},SPLIT=test,RUN_EVENT_EVAL=0,OUT_DIR=${MAIN_TEST_DIR},DEVICE=cpu \
        sub_static_rnn_precision_candidate_eval.slurm
)"
MAIN_TEST_JOB="${MAIN_TEST_SUBMISSION%%;*}"
write_state

export EXTRA_ARGS="--data_dir ${BASE}/ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime"
H48_TEST_SUBMISSION="$(
    sbatch --parsable \
        --export=ALL,MANIFEST=${HYBRID_MANIFEST},SPLIT=test,RUN_EVENT_EVAL=0,OUT_DIR=${H48_TEST_DIR},DEVICE=cpu \
        sub_static_rnn_precision_candidate_eval.slurm
)"
H48_TEST_JOB="${H48_TEST_SUBMISSION%%;*}"
unset EXTRA_ARGS
write_state

SUBMISSION_STATUS="submitting_mean_job"
write_state
MEAN_SUBMISSION="$(
    sbatch --parsable \
        --dependency=afterok:${MAIN_TEST_JOB}:${H48_TEST_JOB} \
        --export=ALL,MANIFEST=${HYBRID_MANIFEST},MAIN_EVAL_DIR=${MAIN_TEST_DIR},FORECAST48_EVAL_DIR=${H48_TEST_DIR},OUT_DIR=${ENSEMBLE_REUSE_DIR},CANDIDATE_ID=p13,EXPECTED_SEEDS=42:314:2718 \
        sub_prepare_static_rnn_seed_mean_for_eval.slurm
)"
MEAN_JOB="${MEAN_SUBMISSION%%;*}"
write_state

SUBMISSION_STATUS="submitting_plot_job"
write_state
export EXTRA_ARGS="--threshold_source argmax --skip_feature_importance --skip_variable_quality --skip_overlap_source_comparison"
PLOT_SUBMISSION="$(
    sbatch --parsable \
        --dependency=afterok:${MEAN_JOB} \
        --export=ALL,CONFIG_JSON=${PAPER_EVAL}/paper_eval_config.json,MODE=main,DATA_DIR=ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2,OUT_DIR=${PAPER_OUT},MAIN_RUN_ID=${RUN42},MAIN_CKPT=${CKPT42},STAGE_TAG=S2_PhaseD,REUSE_INFERENCE_DIR=${ENSEMBLE_REUSE_DIR},PLOTS=all,DEVICE=cpu \
        sub_static_rnn_lowvis_eval.slurm
)"
PLOT_JOB="${PLOT_SUBMISSION%%;*}"
unset EXTRA_ARGS

SUBMISSION_STATUS="submitted"
write_state

echo "[submitted] MAIN_TEST_JOB=${MAIN_TEST_JOB}"
echo "[submitted] H48_TEST_JOB=${H48_TEST_JOB}"
echo "[submitted] MEAN_JOB=${MEAN_JOB}"
echo "[submitted] PLOT_JOB=${PLOT_JOB}"
echo "[submitted] STATE_FILE=${STATE_FILE}"
echo "[submitted] PAPER_OUT=${PAPER_OUT}"
