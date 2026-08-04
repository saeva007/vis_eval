#!/bin/bash
# Submit a strict three-seed Full-versus-No-PM experiment and the downstream
# validation-frozen aerosol-conditioned event analysis.
#
# Public mode submits a short Slurm controller and writes a sourceable state
# file.  The controller submits data builds only when the No-PM datasets are
# absent, then six independent S1->S2 chains, validation/test inference, seed
# means, and one final CPU analysis job.  Using Slurm for the controller keeps
# submission alive when the originating SSH/login-node session disappears.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
BASE="${BASE:-/public/home/putianshu/vis_mlp}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
EVAL_DIR="${EVAL_DIR:-${BASE}/paper_eval}"
LOG_DIR="${AEROSOL_ABLATION_LOG_DIR:-${EVAL_DIR}/logs}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BUNDLE_ID="${BUNDLE_ID:-agu2026_aerosol_ablation_${RUN_STAMP}}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/${BUNDLE_ID}.state.sh}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/${BUNDLE_ID}.launcher.log}"

FULL_S1_DATA_DIR="${FULL_S1_DATA_DIR:-${BASE}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25}"
FULL_S2_DATA_DIR="${FULL_S2_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
NO_PM_S1_DATA_DIR="${NO_PM_S1_DATA_DIR:-${BASE}/ml_dataset_pmst_v5_aligned_12h_airport25}"
NO_PM_S2_DATA_DIR="${NO_PM_S2_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_12h_airport25_monthtail_2}"
CKPT_DIR="${CKPT_DIR:-${BASE}/checkpoints}"
SEEDS_RAW="${AEROSOL_ABLATION_SEEDS:-42:314:2718}"
TARGET_FPR="${TARGET_FPR:-0.04}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
EVENT_TIMES="${EVENT_TIMES:-2025-02-27 22:00:00;2025-09-28 23:00:00;2025-10-30 22:00:00}"
CONTROLLER_PARTITION="${AEROSOL_ABLATION_CONTROLLER_PARTITION:-kshcexclu04}"
CONTROLLER_TIME="${AEROSOL_ABLATION_CONTROLLER_TIME:-00:20:00}"

FULL_MANIFEST="${LOG_DIR}/${BUNDLE_ID}_full_manifest.tsv"
NO_PM_MANIFEST="${LOG_DIR}/${BUNDLE_ID}_no_pm_manifest.tsv"
RESULT_ROOT="${BASE}/aerosol_ablation_results/${BUNDLE_ID}"
FULL_VAL_EVAL_DIR="${RESULT_ROOT}/full_seed_members_val"
FULL_TEST_EVAL_DIR="${RESULT_ROOT}/full_seed_members_test"
NO_PM_VAL_EVAL_DIR="${RESULT_ROOT}/no_pm_seed_members_val"
NO_PM_TEST_EVAL_DIR="${RESULT_ROOT}/no_pm_seed_members_test"
FULL_VAL_ENSEMBLE_DIR="${RESULT_ROOT}/full_seed_mean_val"
FULL_TEST_ENSEMBLE_DIR="${RESULT_ROOT}/full_seed_mean_test"
NO_PM_VAL_ENSEMBLE_DIR="${RESULT_ROOT}/no_pm_seed_mean_val"
NO_PM_TEST_ENSEMBLE_DIR="${RESULT_ROOT}/no_pm_seed_mean_test"
ANALYSIS_OUT_DIR="${RESULT_ROOT}/analysis"

mkdir -p "${LOG_DIR}"

if [[ "${AEROSOL_ABLATION_WORKER:-0}" != "1" ]]; then
    if [[ -e "${STATE_FILE}" || -e "${LAUNCH_LOG}" ]]; then
        echo "ERROR: bundle output already exists for ${BUNDLE_ID}" >&2
        echo "Choose a new BUNDLE_ID or inspect ${STATE_FILE}" >&2
        exit 2
    fi
    for required_cmd in sbatch scontrol; do
        if ! command -v "${required_cmd}" >/dev/null 2>&1; then
            echo "ERROR: required Slurm command is unavailable: ${required_cmd}" >&2
            exit 2
        fi
        echo "public_preflight_command=OK name=${required_cmd}"
    done
    for required_file in \
        "${TRAIN_DIR}/sub_s1_data_aerosol_airport25.slurm" \
        "${TRAIN_DIR}/sub_s2_data_aerosol_airport25.slurm" \
        "${TRAIN_DIR}/sub_static_rnn_lowvis_main.slurm" \
        "${EVAL_DIR}/sub_static_rnn_precision_candidate_eval.slurm" \
        "${EVAL_DIR}/sub_prepare_static_rnn_seed_mean_for_eval.slurm" \
        "${EVAL_DIR}/sub_analyze_aerosol_ablation.slurm" \
        "${EVAL_DIR}/analyze_aerosol_ablation.py"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "ERROR: required file missing before controller submission: ${required_file}" >&2
            exit 2
        fi
        echo "public_preflight_file=OK path=${required_file}"
    done
    export AEROSOL_ABLATION_WORKER=1
    export BASE TRAIN_DIR EVAL_DIR LOG_DIR RUN_STAMP BUNDLE_ID STATE_FILE LAUNCH_LOG
    export FULL_S1_DATA_DIR FULL_S2_DATA_DIR NO_PM_S1_DATA_DIR NO_PM_S2_DATA_DIR
    export CKPT_DIR AEROSOL_ABLATION_SEEDS="${SEEDS_RAW}" TARGET_FPR BOOTSTRAP_ITERS EVENT_TIMES
    export CONTROLLER_PARTITION CONTROLLER_TIME

    controller_raw="$(
        sbatch --parsable --hold \
            --job-name=agu_aero_submit \
            --partition="${CONTROLLER_PARTITION}" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 \
            --time="${CONTROLLER_TIME}" \
            --output="${LAUNCH_LOG}" --error="${LAUNCH_LOG}" \
            --export=ALL "${SCRIPT_FILE}" --worker
    )"
    CONTROLLER_JOB="${controller_raw%%;*}"
    export CONTROLLER_JOB

    initial_state="${STATE_FILE}.tmp.$$"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'STATUS=%q\n' "controller_submitted"
        printf 'CONTROLLER_JOB=%q\n' "${CONTROLLER_JOB}"
        printf 'ALL_JOB_IDS=%q\n' ""
        printf 'STATE_FILE=%q\n' "${STATE_FILE}"
        printf 'LAUNCH_LOG=%q\n' "${LAUNCH_LOG}"
        printf 'UPDATED_AT=%q\n' "$(date -Is)"
    } >"${initial_state}"
    mv "${initial_state}" "${STATE_FILE}"

    if ! scontrol release "${CONTROLLER_JOB}"; then
        echo "ERROR: controller job ${CONTROLLER_JOB} was submitted on hold but could not be released." >&2
        echo "Recover with: scontrol release '${CONTROLLER_JOB}'" >&2
        exit 2
    fi

    echo "aerosol_ablation_launcher=SLURM_CONTROLLER"
    echo "bundle_id=${BUNDLE_ID}"
    echo "controller_job=${CONTROLLER_JOB}"
    echo "launcher_log=${LAUNCH_LOG}"
    echo "state_file=${STATE_FILE}"
    echo "Controller status: squeue -j '${CONTROLLER_JOB}' -o '%.18i %.32j %.2t %.10M %.30R'"
    echo "After the controller finishes: source '${STATE_FILE}' && test -n \"\${ALL_JOB_IDS}\" && squeue -j \"\${ALL_JOB_IDS//:/,}\""
    exit 0
fi

normalize_seeds() {
    local value="$1"
    value="${value//,/ }"
    value="${value//:/ }"
    echo "${value}"
}

SEEDS="$(normalize_seeds "${SEEDS_RAW}")"
if [[ -z "${SEEDS}" ]]; then
    echo "ERROR: AEROSOL_ABLATION_SEEDS is empty" >&2
    exit 2
fi
SEEDS_CSV="$(echo "${SEEDS}" | xargs | tr ' ' ',')"

FULL_S1_JOBS=""
FULL_S2_JOBS=""
NO_PM_S1_JOBS=""
NO_PM_S2_JOBS=""
NO_PM_S1_DATA_JOB=""
NO_PM_S2_DATA_JOB=""
FULL_VAL_EVAL_JOB=""
FULL_TEST_EVAL_JOB=""
NO_PM_VAL_EVAL_JOB=""
NO_PM_TEST_EVAL_JOB=""
FULL_VAL_ENSEMBLE_JOB=""
FULL_TEST_ENSEMBLE_JOB=""
NO_PM_VAL_ENSEMBLE_JOB=""
NO_PM_TEST_ENSEMBLE_JOB=""
ANALYSIS_JOB=""
ALL_JOB_IDS=""
CONTROLLER_JOB="${CONTROLLER_JOB:-${SLURM_JOB_ID:-}}"

append_value() {
    local current="$1"
    local value="$2"
    if [[ -z "${current}" ]]; then
        echo "${value}"
    else
        echo "${current}:${value}"
    fi
}

shell_quote() {
    printf '%q' "$1"
}

write_state() {
    local tmp="${STATE_FILE}.tmp.$$"
    {
        echo "BUNDLE_ID=$(shell_quote "${BUNDLE_ID}")"
        echo "STATUS=$(shell_quote "${STATUS:-submitting}")"
        echo "BASE=$(shell_quote "${BASE}")"
        echo "TRAIN_DIR=$(shell_quote "${TRAIN_DIR}")"
        echo "EVAL_DIR=$(shell_quote "${EVAL_DIR}")"
        echo "FULL_MANIFEST=$(shell_quote "${FULL_MANIFEST}")"
        echo "NO_PM_MANIFEST=$(shell_quote "${NO_PM_MANIFEST}")"
        echo "RESULT_ROOT=$(shell_quote "${RESULT_ROOT}")"
        echo "ANALYSIS_OUT_DIR=$(shell_quote "${ANALYSIS_OUT_DIR}")"
        echo "NO_PM_S1_DATA_JOB=$(shell_quote "${NO_PM_S1_DATA_JOB}")"
        echo "NO_PM_S2_DATA_JOB=$(shell_quote "${NO_PM_S2_DATA_JOB}")"
        echo "FULL_S1_JOBS=$(shell_quote "${FULL_S1_JOBS}")"
        echo "FULL_S2_JOBS=$(shell_quote "${FULL_S2_JOBS}")"
        echo "NO_PM_S1_JOBS=$(shell_quote "${NO_PM_S1_JOBS}")"
        echo "NO_PM_S2_JOBS=$(shell_quote "${NO_PM_S2_JOBS}")"
        echo "FULL_VAL_EVAL_JOB=$(shell_quote "${FULL_VAL_EVAL_JOB}")"
        echo "FULL_TEST_EVAL_JOB=$(shell_quote "${FULL_TEST_EVAL_JOB}")"
        echo "NO_PM_VAL_EVAL_JOB=$(shell_quote "${NO_PM_VAL_EVAL_JOB}")"
        echo "NO_PM_TEST_EVAL_JOB=$(shell_quote "${NO_PM_TEST_EVAL_JOB}")"
        echo "FULL_VAL_ENSEMBLE_JOB=$(shell_quote "${FULL_VAL_ENSEMBLE_JOB}")"
        echo "FULL_TEST_ENSEMBLE_JOB=$(shell_quote "${FULL_TEST_ENSEMBLE_JOB}")"
        echo "NO_PM_VAL_ENSEMBLE_JOB=$(shell_quote "${NO_PM_VAL_ENSEMBLE_JOB}")"
        echo "NO_PM_TEST_ENSEMBLE_JOB=$(shell_quote "${NO_PM_TEST_ENSEMBLE_JOB}")"
        echo "ANALYSIS_JOB=$(shell_quote "${ANALYSIS_JOB}")"
        echo "CONTROLLER_JOB=$(shell_quote "${CONTROLLER_JOB}")"
        echo "ALL_JOB_IDS=$(shell_quote "${ALL_JOB_IDS}")"
        echo "UPDATED_AT=$(shell_quote "$(date -Is)")"
    } >"${tmp}"
    mv "${tmp}" "${STATE_FILE}"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file missing: $1" >&2
        exit 2
    fi
    echo "preflight_file=OK path=$1"
}

dataset_state() {
    local directory="$1"
    local split_kind="$2"
    # Match the established trainer contract for S1: only the four train/val
    # arrays are required. Older nationwide Full datasets legitimately predate
    # dataset_build_config.json and meta_val.csv, and the trainer infers their
    # feature layout directly from X_train.npy.
    local required=(X_train.npy y_train.npy X_val.npy y_val.npy)
    if [[ "${split_kind}" == "s2" ]]; then
        # Downstream paired validation/test and RH-PM/event analysis require
        # validation/test row metadata on S2. Legacy Full S2 datasets may
        # predate dataset_build_config.json; the fixed 27-variable mainline
        # layout is audited from X_val.npy by the final analysis instead.
        required+=(meta_val.csv X_test.npy y_test.npy meta_test.csv)
    fi
    if [[ ! -e "${directory}" ]]; then
        echo "absent"
        return
    fi
    local name
    for name in "${required[@]}"; do
        if [[ ! -s "${directory}/${name}" ]]; then
            echo "partial"
            return
        fi
    done
    echo "ready"
}

submit_data_if_needed() {
    local state
    state="$(dataset_state "${NO_PM_S1_DATA_DIR}" s1)"
    if [[ "${state}" == "partial" ]]; then
        echo "ERROR: partial No-PM S1 dataset; inspect before recovery: ${NO_PM_S1_DATA_DIR}" >&2
        exit 2
    elif [[ "${state}" == "absent" ]]; then
        NO_PM_S1_DATA_JOB="$(
            S1_OUTPUT_DATASET_DIR="${NO_PM_S1_DATA_DIR}" \
                sbatch --parsable --job-name=aero_nopm_s1_data --export=ALL \
                "${TRAIN_DIR}/sub_s1_data_aerosol_airport25.slurm"
        )"
        ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${NO_PM_S1_DATA_JOB}")"
        write_state
        echo "submitted_no_pm_s1_data_job=${NO_PM_S1_DATA_JOB}"
    else
        echo "no_pm_s1_dataset=READY path=${NO_PM_S1_DATA_DIR}"
    fi

    state="$(dataset_state "${NO_PM_S2_DATA_DIR}" s2)"
    if [[ "${state}" == "partial" ]]; then
        echo "ERROR: partial No-PM S2 dataset; inspect before recovery: ${NO_PM_S2_DATA_DIR}" >&2
        exit 2
    elif [[ "${state}" == "absent" ]]; then
        NO_PM_S2_DATA_JOB="$(
            S2_OUTPUT_DATASET_DIR="${NO_PM_S2_DATA_DIR}" \
                sbatch --parsable --job-name=aero_nopm_s2_data --export=ALL \
                "${TRAIN_DIR}/sub_s2_data_aerosol_airport25.slurm"
        )"
        ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${NO_PM_S2_DATA_JOB}")"
        write_state
        echo "submitted_no_pm_s2_data_job=${NO_PM_S2_DATA_JOB}"
    else
        echo "no_pm_s2_dataset=READY path=${NO_PM_S2_DATA_DIR}"
    fi
}

submit_train_arm() {
    local arm="$1"
    local variant_id="$2"
    local s1_data="$3"
    local s2_data="$4"
    local s1_data_job="$5"
    local s2_data_job="$6"
    local manifest="$7"
    local arm_s1_jobs=""
    local arm_s2_jobs=""
    local seed

    printf "candidate_id\tcandidate_label\texperiment_status\treplaces_mainline\tvariant_id\tseed\tstage\trun_prefix\trun_id\textra_args\ts1_job\ts2_job\ts1_checkpoint\ts2_checkpoint\n" >"${manifest}"
    for seed in ${SEEDS}; do
        local run_id="${BUNDLE_ID}_${arm}_seed${seed}_static_mlp_gru"
        local s1_ckpt="${CKPT_DIR}/${run_id}_S1_best_score.pt"
        local s2_ckpt="${CKPT_DIR}/${run_id}_S2_PhaseB_best_score.pt"
        local extra_args="--seed ${seed} --threshold-mode argmax --aerosol-hard-weight 0"
        if [[ "${arm}" == "no_pm" ]]; then
            extra_args="${extra_args} --no-pm"
        fi
        local s1_job
        if [[ -n "${s1_data_job}" ]]; then
            s1_job="$(
                LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
                    sbatch --parsable --dependency="afterok:${s1_data_job}" \
                    --job-name="aero_${arm}_s1_${seed}" \
                    --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_RUN_ID="${run_id}",LOWVIS_RNN_S1_DATA_DIR="${s1_data}",LOWVIS_RNN_S2_DATA_DIR="${s2_data}",LOWVIS_RNN_LOCAL_CACHE_ID="${BUNDLE_ID}_${arm}_seed${seed}" \
                    "${TRAIN_DIR}/sub_static_rnn_lowvis_main.slurm"
            )"
        else
            s1_job="$(
                LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
                    sbatch --parsable \
                    --job-name="aero_${arm}_s1_${seed}" \
                    --export=ALL,LOWVIS_RNN_MODE=s1,LOWVIS_RNN_RUN_ID="${run_id}",LOWVIS_RNN_S1_DATA_DIR="${s1_data}",LOWVIS_RNN_S2_DATA_DIR="${s2_data}",LOWVIS_RNN_LOCAL_CACHE_ID="${BUNDLE_ID}_${arm}_seed${seed}" \
                    "${TRAIN_DIR}/sub_static_rnn_lowvis_main.slurm"
            )"
        fi
        arm_s1_jobs="$(append_value "${arm_s1_jobs}" "${s1_job}")"
        ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${s1_job}")"
        if [[ "${arm}" == "full" ]]; then
            FULL_S1_JOBS="${arm_s1_jobs}"
        else
            NO_PM_S1_JOBS="${arm_s1_jobs}"
        fi
        write_state

        local s2_dependency="afterok:${s1_job}"
        if [[ -n "${s2_data_job}" ]]; then
            s2_dependency="${s2_dependency}:${s2_data_job}"
        fi
        local s2_job
        s2_job="$(
            LOWVIS_RNN_EXTRA_ARGS="${extra_args}" \
                sbatch --parsable --dependency="${s2_dependency}" \
                --job-name="aero_${arm}_s2_${seed}" \
                --export=ALL,LOWVIS_RNN_MODE=s2,LOWVIS_RNN_RUN_ID="${run_id}",LOWVIS_RNN_S1_DATA_DIR="${s1_data}",LOWVIS_RNN_S2_DATA_DIR="${s2_data}",LOWVIS_RNN_PRETRAINED_CKPT="${s1_ckpt}",LOWVIS_RNN_LOCAL_CACHE_ID="${BUNDLE_ID}_${arm}_seed${seed}" \
                "${TRAIN_DIR}/sub_static_rnn_lowvis_main.slurm"
        )"
        arm_s2_jobs="$(append_value "${arm_s2_jobs}" "${s2_job}")"
        ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${s2_job}")"
        if [[ "${arm}" == "full" ]]; then
            FULL_S2_JOBS="${arm_s2_jobs}"
        else
            NO_PM_S2_JOBS="${arm_s2_jobs}"
        fi
        printf "%s\t%s\tcandidate_only\tfalse\t%s\t%s\tfull\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${arm}" "${arm}_static_mlp_gru" "${variant_id}" "${seed}" "${BUNDLE_ID}_${arm}_seed${seed}" \
            "${run_id}" "${extra_args}" "${s1_job}" "${s2_job}" "${s1_ckpt}" "${s2_ckpt}" >>"${manifest}"
        write_state
        echo "submitted_${arm}_seed${seed}=${s1_job}->${s2_job}"
    done
}

submit_member_eval() {
    local arm="$1"
    local manifest="$2"
    local split="$3"
    local data_dir="$4"
    local out_dir="$5"
    local dependency="$6"
    MANIFEST="${manifest}" SPLIT="${split}" DATA_DIR="${data_dir}" OUT_DIR="${out_dir}" \
        RUN_EVENT_EVAL=0 EXTRA_ARGS="--config_json none" \
        sbatch --parsable --dependency="afterok:${dependency}" \
        --job-name="aero_${arm}_${split}_eval" --export=ALL \
        "${EVAL_DIR}/sub_static_rnn_precision_candidate_eval.slurm"
}

submit_seed_mean() {
    local manifest="$1"
    local candidate_id="$2"
    local eval_split="$3"
    local member_dir="$4"
    local out_dir="$5"
    local dependency="$6"
    MANIFEST="${manifest}" CANDIDATE_ID="${candidate_id}" EXPECTED_SEEDS="${SEEDS_CSV}" \
        EVAL_SPLIT="${eval_split}" MAIN_EVAL_DIR="${member_dir}" OUT_DIR="${out_dir}" \
        sbatch --parsable --dependency="afterok:${dependency}" \
        --job-name="aero_${candidate_id}_${eval_split}_mean" --export=ALL \
        "${EVAL_DIR}/sub_prepare_static_rnn_seed_mean_for_eval.slurm"
}

STATUS="preflight"
write_state
require_file "${TRAIN_DIR}/sub_s1_data_aerosol_airport25.slurm"
require_file "${TRAIN_DIR}/sub_s2_data_aerosol_airport25.slurm"
require_file "${TRAIN_DIR}/sub_static_rnn_lowvis_main.slurm"
require_file "${EVAL_DIR}/sub_static_rnn_precision_candidate_eval.slurm"
require_file "${EVAL_DIR}/sub_prepare_static_rnn_seed_mean_for_eval.slurm"
require_file "${EVAL_DIR}/sub_analyze_aerosol_ablation.slurm"
require_file "${EVAL_DIR}/analyze_aerosol_ablation.py"

if [[ "$(dataset_state "${FULL_S1_DATA_DIR}" s1)" != "ready" ]]; then
    echo "ERROR: Full S1 dataset is not complete: ${FULL_S1_DATA_DIR}" >&2
    exit 2
fi
if [[ "$(dataset_state "${FULL_S2_DATA_DIR}" s2)" != "ready" ]]; then
    echo "ERROR: Full S2 dataset is not complete: ${FULL_S2_DATA_DIR}" >&2
    exit 2
fi
echo "full_s1_dataset=READY path=${FULL_S1_DATA_DIR}"
echo "full_s2_dataset=READY path=${FULL_S2_DATA_DIR}"
echo "train_commit=$(git -C "${TRAIN_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "eval_commit=$(git -C "${EVAL_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

STATUS="submitting"
write_state
submit_data_if_needed
submit_train_arm full 0 "${FULL_S1_DATA_DIR}" "${FULL_S2_DATA_DIR}" "" "" "${FULL_MANIFEST}"
submit_train_arm no_pm 3 "${NO_PM_S1_DATA_DIR}" "${NO_PM_S2_DATA_DIR}" "${NO_PM_S1_DATA_JOB}" "${NO_PM_S2_DATA_JOB}" "${NO_PM_MANIFEST}"

FULL_VAL_EVAL_JOB="$(submit_member_eval full "${FULL_MANIFEST}" val "${FULL_S2_DATA_DIR}" "${FULL_VAL_EVAL_DIR}" "${FULL_S2_JOBS}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${FULL_VAL_EVAL_JOB}")"
write_state
FULL_TEST_EVAL_JOB="$(submit_member_eval full "${FULL_MANIFEST}" test "${FULL_S2_DATA_DIR}" "${FULL_TEST_EVAL_DIR}" "${FULL_S2_JOBS}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${FULL_TEST_EVAL_JOB}")"
write_state
NO_PM_VAL_EVAL_JOB="$(submit_member_eval nopm "${NO_PM_MANIFEST}" val "${NO_PM_S2_DATA_DIR}" "${NO_PM_VAL_EVAL_DIR}" "${NO_PM_S2_JOBS}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${NO_PM_VAL_EVAL_JOB}")"
write_state
NO_PM_TEST_EVAL_JOB="$(submit_member_eval nopm "${NO_PM_MANIFEST}" test "${NO_PM_S2_DATA_DIR}" "${NO_PM_TEST_EVAL_DIR}" "${NO_PM_S2_JOBS}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${NO_PM_TEST_EVAL_JOB}")"
write_state

FULL_VAL_ENSEMBLE_JOB="$(submit_seed_mean "${FULL_MANIFEST}" full val "${FULL_VAL_EVAL_DIR}" "${FULL_VAL_ENSEMBLE_DIR}" "${FULL_VAL_EVAL_JOB}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${FULL_VAL_ENSEMBLE_JOB}")"
write_state
FULL_TEST_ENSEMBLE_JOB="$(submit_seed_mean "${FULL_MANIFEST}" full test "${FULL_TEST_EVAL_DIR}" "${FULL_TEST_ENSEMBLE_DIR}" "${FULL_TEST_EVAL_JOB}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${FULL_TEST_ENSEMBLE_JOB}")"
write_state
NO_PM_VAL_ENSEMBLE_JOB="$(submit_seed_mean "${NO_PM_MANIFEST}" no_pm val "${NO_PM_VAL_EVAL_DIR}" "${NO_PM_VAL_ENSEMBLE_DIR}" "${NO_PM_VAL_EVAL_JOB}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${NO_PM_VAL_ENSEMBLE_JOB}")"
write_state
NO_PM_TEST_ENSEMBLE_JOB="$(submit_seed_mean "${NO_PM_MANIFEST}" no_pm test "${NO_PM_TEST_EVAL_DIR}" "${NO_PM_TEST_ENSEMBLE_DIR}" "${NO_PM_TEST_EVAL_JOB}")"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${NO_PM_TEST_ENSEMBLE_JOB}")"
write_state

analysis_dependency="afterok:${FULL_VAL_ENSEMBLE_JOB}:${FULL_TEST_ENSEMBLE_JOB}:${NO_PM_VAL_ENSEMBLE_JOB}:${NO_PM_TEST_ENSEMBLE_JOB}"
ANALYSIS_JOB="$(
    FULL_VAL_DIR="${FULL_VAL_ENSEMBLE_DIR}" NO_PM_VAL_DIR="${NO_PM_VAL_ENSEMBLE_DIR}" \
    FULL_TEST_DIR="${FULL_TEST_ENSEMBLE_DIR}" NO_PM_TEST_DIR="${NO_PM_TEST_ENSEMBLE_DIR}" \
    FULL_DATA_DIR="${FULL_S2_DATA_DIR}" OUT_DIR="${ANALYSIS_OUT_DIR}" \
    TARGET_FPR="${TARGET_FPR}" BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS}" EVENT_TIMES="${EVENT_TIMES}" \
        sbatch --parsable --dependency="${analysis_dependency}" --export=ALL \
        --job-name=aero_final_analysis \
        "${EVAL_DIR}/sub_analyze_aerosol_ablation.slurm"
)"
ALL_JOB_IDS="$(append_value "${ALL_JOB_IDS}" "${ANALYSIS_JOB}")"
STATUS="submitted"
write_state

echo "aerosol_ablation_submission=COMPLETE"
echo "bundle_id=${BUNDLE_ID}"
echo "analysis_job=${ANALYSIS_JOB}"
echo "all_job_ids=${ALL_JOB_IDS}"
echo "state_file=${STATE_FILE}"
echo "analysis_out_dir=${ANALYSIS_OUT_DIR}"
