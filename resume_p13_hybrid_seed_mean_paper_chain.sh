#!/bin/bash
# Resume only the failed P13 seed-mean and dependent paper-figure jobs after
# member inference has completed. The public entrypoint self-detaches via nohup.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
PAPER_EVAL="${PAPER_EVAL:-${BASE}/paper_eval}"

if [ "${P13_RESUME_WORKER:-0}" != "1" ]; then
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    RECOVERY_ID="${RECOVERY_ID:-p13_seed_mean_timefix_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${PAPER_EVAL}/logs"
    RECOVERY_LOG="${RECOVERY_LOG:-${PAPER_EVAL}/logs/${RECOVERY_ID}_launcher.log}"
    RECOVERY_STATE="${RECOVERY_STATE:-${PAPER_EVAL}/logs/${RECOVERY_ID}_submission.env}"
    HANDLE_FILE="${PAPER_EVAL}/logs/${RECOVERY_ID}_launcher.env"

    nohup env \
        P13_RESUME_WORKER=1 \
        BASE="${BASE}" \
        PAPER_EVAL="${PAPER_EVAL}" \
        RECOVERY_ID="${RECOVERY_ID}" \
        RECOVERY_STATE="${RECOVERY_STATE}" \
        SOURCE_STATE_FILE="${SOURCE_STATE_FILE:-}" \
        bash "${SCRIPT_PATH}" </dev/null >"${RECOVERY_LOG}" 2>&1 &
    WORKER_PID=$!

    HANDLE_TMP="${HANDLE_FILE}.tmp.$$"
    {
        printf 'RECOVERY_ID=%q\n' "${RECOVERY_ID}"
        printf 'WORKER_PID=%q\n' "${WORKER_PID}"
        printf 'RECOVERY_LOG=%q\n' "${RECOVERY_LOG}"
        printf 'RECOVERY_STATE=%q\n' "${RECOVERY_STATE}"
    } > "${HANDLE_TMP}"
    mv -f "${HANDLE_TMP}" "${HANDLE_FILE}"

    echo "Detached P13 seed-mean recovery; an SSH disconnect is safe."
    echo "RECOVERY_ID=${RECOVERY_ID}"
    echo "WORKER_PID=${WORKER_PID}"
    echo "RECOVERY_LOG=${RECOVERY_LOG}"
    echo "RECOVERY_STATE=${RECOVERY_STATE}"
    echo "HANDLE_FILE=${HANDLE_FILE}"
    exit 0
fi

cd "${PAPER_EVAL}"
mkdir -p logs

SOURCE_STATE_FILE="${SOURCE_STATE_FILE:-}"
NEW_MEAN_JOB=""
NEW_PLOT_JOB=""
RECOVERY_STATUS="starting"
SOURCE_BUNDLE_ID=""
RECOVERY_REUSE_DIR=""
RECOVERY_PAPER_OUT=""

slurm_state() {
    local job_id="$1"
    sacct -X -n -P -j "${job_id}" -o State 2>/dev/null | awk -F'|' 'NF && $1!="" {print $1; exit}'
}

discover_failed_source_state() {
    local file mean_job state
    local -a matches=()
    shopt -s nullglob
    for file in "${PAPER_EVAL}"/logs/p13_hybrid_three_seed_*_submission.env; do
        mean_job="$(sed -n 's/^MEAN_JOB=//p' "${file}" | tr -d "'\"")"
        if [[ ! "${mean_job}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        state="$(slurm_state "${mean_job}")"
        if [[ "${state}" == FAILED* ]]; then
            matches+=("${file}")
        fi
    done
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one P13 source state with a FAILED mean job; found ${#matches[@]}." >&2
        printf '  %s\n' "${matches[@]}" >&2
        echo "Set SOURCE_STATE_FILE to the exact original *_submission.env and retry." >&2
        return 2
    fi
    printf '%s\n' "${matches[0]}"
}

if [ -z "${SOURCE_STATE_FILE}" ]; then
    SOURCE_STATE_FILE="$(discover_failed_source_state)"
fi
if [ ! -s "${SOURCE_STATE_FILE}" ]; then
    echo "ERROR: source submission state not found: ${SOURCE_STATE_FILE}" >&2
    exit 2
fi
SOURCE_STATE_FILE="$(readlink -f "${SOURCE_STATE_FILE}")"

write_recovery_state() {
    local tmp="${RECOVERY_STATE}.tmp.$$"
    {
        printf 'RECOVERY_STATUS=%q\n' "${RECOVERY_STATUS}"
        printf 'SOURCE_STATE_FILE=%q\n' "${SOURCE_STATE_FILE}"
        printf 'SOURCE_BUNDLE_ID=%q\n' "${SOURCE_BUNDLE_ID}"
        printf 'RECOVERY_REUSE_DIR=%q\n' "${RECOVERY_REUSE_DIR}"
        printf 'RECOVERY_PAPER_OUT=%q\n' "${RECOVERY_PAPER_OUT}"
        printf 'NEW_MEAN_JOB=%q\n' "${NEW_MEAN_JOB}"
        printf 'NEW_PLOT_JOB=%q\n' "${NEW_PLOT_JOB}"
    } > "${tmp}"
    mv -f "${tmp}" "${RECOVERY_STATE}"
}

on_exit() {
    local status=$?
    trap - EXIT
    if [ "${status}" -ne 0 ]; then
        RECOVERY_STATUS="failed_rc_${status}"
    fi
    write_recovery_state || true
    exit "${status}"
}
trap on_exit EXIT

# shellcheck disable=SC1090
source "${SOURCE_STATE_FILE}"
SOURCE_BUNDLE_ID="${BUNDLE_ID:?source state has no BUNDLE_ID}"
SOURCE_MANIFEST="${HYBRID_MANIFEST:?source state has no HYBRID_MANIFEST}"
SOURCE_MAIN_TEST_DIR="${MAIN_TEST_DIR:?source state has no MAIN_TEST_DIR}"
SOURCE_H48_TEST_DIR="${H48_TEST_DIR:?source state has no H48_TEST_DIR}"
SOURCE_MAIN_TEST_JOB="${MAIN_TEST_JOB:?source state has no MAIN_TEST_JOB}"
SOURCE_H48_TEST_JOB="${H48_TEST_JOB:?source state has no H48_TEST_JOB}"
SOURCE_MEAN_JOB="${MEAN_JOB:?source state has no MEAN_JOB}"
SOURCE_PLOT_JOB="${PLOT_JOB:-}"

for path in \
    "${SOURCE_MANIFEST}" \
    "${SOURCE_MAIN_TEST_DIR}/run_config.json" \
    "${SOURCE_H48_TEST_DIR}/run_config.json"; do
    test -s "${path}" || { echo "ERROR: missing reusable member artifact: ${path}" >&2; exit 2; }
done

for pair in \
    "main:${SOURCE_MAIN_TEST_JOB}" \
    "48h:${SOURCE_H48_TEST_JOB}"; do
    label="${pair%%:*}"
    job_id="${pair#*:}"
    state="$(slurm_state "${job_id}")"
    echo "[preflight] ${label} member job=${job_id} state=${state}"
    [[ "${state}" == COMPLETED* ]] || {
        echo "ERROR: ${label} member inference is not completed; refusing recovery." >&2
        exit 2
    }
done

mean_state="$(slurm_state "${SOURCE_MEAN_JOB}")"
echo "[preflight] failed mean job=${SOURCE_MEAN_JOB} state=${mean_state}"
[[ "${mean_state}" == FAILED* ]] || {
    echo "ERROR: source mean job is not FAILED; refusing ambiguous recovery." >&2
    exit 2
}

grep -q '^def canonical_valid_times' "${PAPER_EVAL}/prepare_static_rnn_seed_mean_for_eval.py" || {
    echo "ERROR: time-identity fix is not present in prepare_static_rnn_seed_mean_for_eval.py" >&2
    exit 2
}

RECOVERY_REUSE_DIR="${BASE}/static_rnn_precision_candidate_eval/${RECOVERY_ID}_mean_argmax"
RECOVERY_PAPER_OUT="${BASE}/static_rnn_eval_results/${RECOVERY_ID}_paper_figures"
test ! -e "${RECOVERY_REUSE_DIR}" || { echo "ERROR: output exists: ${RECOVERY_REUSE_DIR}" >&2; exit 2; }
test ! -e "${RECOVERY_PAPER_OUT}" || { echo "ERROR: output exists: ${RECOVERY_PAPER_OUT}" >&2; exit 2; }

manifest_value() {
    local seed="$1" field="$2"
    awk -F'\t' -v seed="${seed}" -v field="${field}" '
        NR==1 {for (i=1; i<=NF; i++) col[$i]=i; next}
        $col["candidate_id"]=="p13" && $col["seed"]==seed {print $col[field]; exit}
    ' "${SOURCE_MANIFEST}"
}

RUN42="$(manifest_value 42 run_id)"
CKPT42="$(manifest_value 42 s2_checkpoint)"
test -n "${RUN42}" || { echo "ERROR: seed-42 run_id missing from manifest" >&2; exit 2; }
test -s "${CKPT42}" || { echo "ERROR: seed-42 checkpoint missing: ${CKPT42}" >&2; exit 2; }

RECOVERY_STATUS="submitting_mean"
write_recovery_state
MEAN_SUBMISSION="$(
    sbatch --parsable \
        --export=ALL,MANIFEST=${SOURCE_MANIFEST},MAIN_EVAL_DIR=${SOURCE_MAIN_TEST_DIR},FORECAST48_EVAL_DIR=${SOURCE_H48_TEST_DIR},OUT_DIR=${RECOVERY_REUSE_DIR},CANDIDATE_ID=p13,EXPECTED_SEEDS=42:314:2718 \
        sub_prepare_static_rnn_seed_mean_for_eval.slurm
)"
NEW_MEAN_JOB="${MEAN_SUBMISSION%%;*}"
write_recovery_state

RECOVERY_STATUS="submitting_plot"
write_recovery_state
export EXTRA_ARGS="--threshold_source argmax --skip_feature_importance --skip_variable_quality --skip_overlap_source_comparison"
PLOT_SUBMISSION="$(
    sbatch --parsable \
        --dependency=afterok:${NEW_MEAN_JOB} \
        --export=ALL,CONFIG_JSON=${PAPER_EVAL}/paper_eval_config.json,MODE=main,DATA_DIR=ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2,OUT_DIR=${RECOVERY_PAPER_OUT},MAIN_RUN_ID=${RUN42},MAIN_CKPT=${CKPT42},STAGE_TAG=S2_PhaseD,REUSE_INFERENCE_DIR=${RECOVERY_REUSE_DIR},PLOTS=all,DEVICE=cpu \
        sub_static_rnn_lowvis_eval.slurm
)"
NEW_PLOT_JOB="${PLOT_SUBMISSION%%;*}"
unset EXTRA_ARGS

RECOVERY_STATUS="submitted"
write_recovery_state

echo "[submitted] SOURCE_BUNDLE_ID=${SOURCE_BUNDLE_ID}"
echo "[submitted] NEW_MEAN_JOB=${NEW_MEAN_JOB}"
echo "[submitted] NEW_PLOT_JOB=${NEW_PLOT_JOB}"
echo "[submitted] RECOVERY_REUSE_DIR=${RECOVERY_REUSE_DIR}"
echo "[submitted] RECOVERY_PAPER_OUT=${RECOVERY_PAPER_OUT}"
if [[ "${SOURCE_PLOT_JOB}" =~ ^[0-9]+$ ]]; then
    echo "[note] old dependent plot job=${SOURCE_PLOT_JOB}; inspect its terminal/dependency state separately."
fi
