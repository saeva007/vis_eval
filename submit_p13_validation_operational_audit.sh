#!/bin/bash
# Build the formal P13 three-seed validation ensemble and use it to select
# operating points that are then frozen and evaluated on the completed test
# paper-evaluation output. The public entrypoint self-detaches so SSH loss is
# safe after this script prints its handle and state paths.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
PAPER_EVAL="${PAPER_EVAL:-${BASE}/paper_eval}"
TEST_EVAL_DIR="${TEST_EVAL_DIR:-${BASE}/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal}"
TEST_REUSE_DIR="${TEST_REUSE_DIR:-${BASE}/static_rnn_precision_candidate_eval/p13_seed_mean_timefix_20260719_130856_mean_argmax}"

if [[ "${P13_VALIDATION_AUDIT_WORKER:-0}" != "1" ]]; then
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    BUNDLE_ID="${BUNDLE_ID:-p13_validation_operational_audit_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${PAPER_EVAL}/logs"
    LAUNCHER_LOG="${LAUNCHER_LOG:-${PAPER_EVAL}/logs/${BUNDLE_ID}_launcher.log}"
    STATE_FILE="${STATE_FILE:-${PAPER_EVAL}/logs/${BUNDLE_ID}_submission.env}"
    HANDLE_FILE="${PAPER_EVAL}/logs/${BUNDLE_ID}_launcher.env"

    nohup env \
        P13_VALIDATION_AUDIT_WORKER=1 \
        BASE="${BASE}" \
        PAPER_EVAL="${PAPER_EVAL}" \
        TEST_EVAL_DIR="${TEST_EVAL_DIR}" \
        TEST_REUSE_DIR="${TEST_REUSE_DIR}" \
        BUNDLE_ID="${BUNDLE_ID}" \
        LAUNCHER_LOG="${LAUNCHER_LOG}" \
        STATE_FILE="${STATE_FILE}" \
        SOURCE_MANIFEST="${SOURCE_MANIFEST:-}" \
        DATA_DIR="${DATA_DIR:-}" \
        bash "${SCRIPT_PATH}" </dev/null >"${LAUNCHER_LOG}" 2>&1 &
    WORKER_PID=$!

    HANDLE_TMP="${HANDLE_FILE}.tmp.$$"
    {
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'WORKER_PID=%q\n' "${WORKER_PID}"
        printf 'LAUNCHER_LOG=%q\n' "${LAUNCHER_LOG}"
        printf 'STATE_FILE=%q\n' "${STATE_FILE}"
    } > "${HANDLE_TMP}"
    mv -f "${HANDLE_TMP}" "${HANDLE_FILE}"

    echo "Detached P13 validation-to-test operational audit submission; SSH disconnect is safe."
    echo "BUNDLE_ID=${BUNDLE_ID}"
    echo "WORKER_PID=${WORKER_PID}"
    echo "LAUNCHER_LOG=${LAUNCHER_LOG}"
    echo "STATE_FILE=${STATE_FILE}"
    echo "HANDLE_FILE=${HANDLE_FILE}"
    echo "Monitor launcher: tail -f ${LAUNCHER_LOG}"
    exit 0
fi

cd "${PAPER_EVAL}"
mkdir -p logs

BUNDLE_ID="${BUNDLE_ID:?worker requires BUNDLE_ID}"
STATE_FILE="${STATE_FILE:?worker requires STATE_FILE}"
STATUS="starting"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
SOURCE_DATA_DIR="${DATA_DIR:-}"
VAL_MEMBER_DIR="${BASE}/static_rnn_precision_candidate_eval/${BUNDLE_ID}_val_members"
VAL_MEAN_DIR="${BASE}/static_rnn_precision_candidate_eval/${BUNDLE_ID}_val_mean_argmax"
AUDIT_DIR="${TEST_EVAL_DIR}/operational_tradeoff_audit_${BUNDLE_ID}"
VAL_MEMBER_JOB=""
VAL_MEAN_JOB=""
AUDIT_JOB=""

write_state() {
    local tmp="${STATE_FILE}.tmp.$$"
    {
        printf 'STATUS=%q\n' "${STATUS}"
        printf 'BUNDLE_ID=%q\n' "${BUNDLE_ID}"
        printf 'TEST_EVAL_DIR=%q\n' "${TEST_EVAL_DIR}"
        printf 'TEST_REUSE_DIR=%q\n' "${TEST_REUSE_DIR}"
        printf 'SOURCE_MANIFEST=%q\n' "${SOURCE_MANIFEST}"
        printf 'SOURCE_DATA_DIR=%q\n' "${SOURCE_DATA_DIR}"
        printf 'VAL_MEMBER_DIR=%q\n' "${VAL_MEMBER_DIR}"
        printf 'VAL_MEAN_DIR=%q\n' "${VAL_MEAN_DIR}"
        printf 'AUDIT_DIR=%q\n' "${AUDIT_DIR}"
        printf 'VAL_MEMBER_JOB=%q\n' "${VAL_MEMBER_JOB}"
        printf 'VAL_MEAN_JOB=%q\n' "${VAL_MEAN_JOB}"
        printf 'AUDIT_JOB=%q\n' "${AUDIT_JOB}"
    } > "${tmp}"
    mv -f "${tmp}" "${STATE_FILE}"
}

on_exit() {
    local rc=$?
    trap - EXIT
    if [[ "${rc}" -ne 0 ]]; then
        STATUS="failed_rc_${rc}"
    fi
    write_state || true
    exit "${rc}"
}
trap on_exit EXIT

required_files=(
    "${PAPER_EVAL}/run_static_rnn_precision_candidate_eval.py"
    "${PAPER_EVAL}/prepare_static_rnn_seed_mean_for_eval.py"
    "${PAPER_EVAL}/analyze_lowvis_operational_tradeoffs.py"
    "${PAPER_EVAL}/sub_static_rnn_precision_candidate_eval.slurm"
    "${PAPER_EVAL}/sub_prepare_static_rnn_seed_mean_for_eval.slurm"
    "${PAPER_EVAL}/sub_lowvis_operational_tradeoff_audit.slurm"
    "${BASE}/ifs_baseline/activate_eval_torch_runtime.sh"
    "${TEST_EVAL_DIR}/per_sample_eval.csv"
    "${TEST_EVAL_DIR}/run_config.json"
    "${TEST_REUSE_DIR}/probs.npy"
    "${TEST_REUSE_DIR}/run_config.json"
)
for path in "${required_files[@]}"; do
    test -s "${path}" || {
        echo "ERROR: required artifact is missing or empty: ${path}" >&2
        exit 2
    }
    echo "[preflight] file=OK ${path}"
done

JSON_PYTHON="${JSON_PYTHON:-}"
if [[ -z "${JSON_PYTHON}" ]]; then
    JSON_PYTHON="$(command -v python3 || command -v python || true)"
fi
test -n "${JSON_PYTHON}" || {
    echo "ERROR: neither python3 nor python is available for provenance preflight" >&2
    exit 2
}

SOURCE_INFO="$("${JSON_PYTHON}" - "${TEST_REUSE_DIR}/run_config.json" "${TEST_EVAL_DIR}/run_config.json" "${BASE}" "${TEST_REUSE_DIR}" "${SOURCE_MANIFEST}" "${SOURCE_DATA_DIR}" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

reuse_config_path, paper_config_path, base_raw, expected_reuse_raw, manifest_override, data_override = sys.argv[1:]
base = Path(base_raw).expanduser().resolve()
expected_reuse = Path(expected_reuse_raw).expanduser().resolve()
reuse = json.loads(Path(reuse_config_path).read_text(encoding="utf-8"))
paper = json.loads(Path(paper_config_path).read_text(encoding="utf-8"))

def resolved_under_base(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()

configured_reuse = resolved_under_base(str(paper.get("reuse_inference_dir", "")))
if configured_reuse != expected_reuse:
    raise SystemExit(
        f"paper-eval reuse mismatch: config={configured_reuse}, expected={expected_reuse}"
    )

main_config = reuse.get("main", {}).get("eval_run_config", {})
if str(main_config.get("eval_split", "")) != "test":
    raise SystemExit("source ensemble was not built from the frozen test split")

manifest_value = manifest_override or str(reuse.get("manifest", ""))
if not manifest_value:
    raise SystemExit("source ensemble run_config has no manifest")
manifest = resolved_under_base(manifest_value)
if not manifest.is_file():
    raise SystemExit(f"source manifest is missing: {manifest}")

expected_manifest_hash = str(reuse.get("manifest_sha256", ""))
actual_manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
    raise SystemExit("source manifest hash differs from the completed test ensemble")

data_value = data_override or str(main_config.get("data_dir", ""))
if not data_value:
    raise SystemExit("source test member evaluation has no data_dir")
data_dir = resolved_under_base(data_value)
paper_data_dir = resolved_under_base(str(paper.get("data_dir", "")))
if paper_data_dir != data_dir:
    raise SystemExit(f"paper/reuse dataset mismatch: paper={paper_data_dir}, reuse={data_dir}")
for name in ("X_val.npy", "y_val.npy", "meta_val.csv"):
    path = data_dir / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"validation dataset artifact is missing or empty: {path}")

with manifest.open("r", encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("candidate_id") == "p13"]
actual_seeds = [str(row.get("seed", "")) for row in rows]
if sorted(actual_seeds) != ["2718", "314", "42"] or len(rows) != 3:
    raise SystemExit(f"manifest must contain exactly P13 seeds 42,314,2718; got {actual_seeds}")
if {str(row.get("stage", "")) for row in rows} != {"full"}:
    raise SystemExit("P13 validation ensemble requires the three formal full-stage rows")
for row in rows:
    checkpoint = resolved_under_base(str(row.get("s2_checkpoint", "")))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SystemExit(f"seed {row.get('seed')} checkpoint is missing or empty: {checkpoint}")

print(f"{manifest}\t{data_dir}")
PY
)"
IFS=$'\t' read -r SOURCE_MANIFEST SOURCE_DATA_DIR <<< "${SOURCE_INFO}"
test -s "${SOURCE_MANIFEST}" || {
    echo "ERROR: resolved source manifest is invalid: ${SOURCE_MANIFEST}" >&2
    exit 2
}
test -d "${SOURCE_DATA_DIR}" || {
    echo "ERROR: resolved source dataset is invalid: ${SOURCE_DATA_DIR}" >&2
    exit 2
}
echo "[preflight] source_manifest=OK ${SOURCE_MANIFEST}"
echo "[preflight] source_data_dir=OK ${SOURCE_DATA_DIR}"
echo "[preflight] validation_policy=three_seed_mean_post_softmax_then_validation_selection_then_frozen_test"

for output in "${VAL_MEMBER_DIR}" "${VAL_MEAN_DIR}" "${AUDIT_DIR}"; do
    test ! -e "${output}" || {
        echo "ERROR: refusing to reuse existing formal output: ${output}" >&2
        exit 2
    }
    echo "[preflight] output=new ${output}"
done

STATUS="submitting_validation_members"
write_state
VAL_MEMBER_SUBMISSION="$(
    sbatch --parsable \
        --export=ALL,MANIFEST="${SOURCE_MANIFEST}",SPLIT=val,RUN_EVENT_EVAL=0,OUT_DIR="${VAL_MEMBER_DIR}",DEVICE=cpu,DATA_DIR="${SOURCE_DATA_DIR}" \
        sub_static_rnn_precision_candidate_eval.slurm
)"
VAL_MEMBER_JOB="${VAL_MEMBER_SUBMISSION%%;*}"
STATUS="validation_members_submitted"
write_state

STATUS="submitting_validation_mean"
write_state
VAL_MEAN_SUBMISSION="$(
    sbatch --parsable \
        --dependency=afterok:"${VAL_MEMBER_JOB}" \
        --export=ALL,MANIFEST="${SOURCE_MANIFEST}",MAIN_EVAL_DIR="${VAL_MEMBER_DIR}",OUT_DIR="${VAL_MEAN_DIR}",CANDIDATE_ID=p13,EXPECTED_SEEDS=42:314:2718,EVAL_SPLIT=val \
        sub_prepare_static_rnn_seed_mean_for_eval.slurm
)"
VAL_MEAN_JOB="${VAL_MEAN_SUBMISSION%%;*}"
STATUS="validation_mean_submitted"
write_state

STATUS="submitting_frozen_test_audit"
write_state
AUDIT_SUBMISSION="$(
    sbatch --parsable \
        --dependency=afterok:"${VAL_MEAN_JOB}" \
        --export=ALL,EVAL_DIR="${TEST_EVAL_DIR}",SELECTION_DIR="${VAL_MEAN_DIR}",OUT_DIR="${AUDIT_DIR}" \
        sub_lowvis_operational_tradeoff_audit.slurm
)"
AUDIT_JOB="${AUDIT_SUBMISSION%%;*}"
STATUS="submitted"
write_state

echo "[submitted] BUNDLE_ID=${BUNDLE_ID}"
echo "[submitted] VAL_MEMBER_JOB=${VAL_MEMBER_JOB}"
echo "[submitted] VAL_MEAN_JOB=${VAL_MEAN_JOB} dependency=afterok:${VAL_MEMBER_JOB}"
echo "[submitted] AUDIT_JOB=${AUDIT_JOB} dependency=afterok:${VAL_MEAN_JOB}"
echo "[submitted] VAL_MEMBER_DIR=${VAL_MEMBER_DIR}"
echo "[submitted] VAL_MEAN_DIR=${VAL_MEAN_DIR}"
echo "[submitted] AUDIT_DIR=${AUDIT_DIR}"
echo "[submitted] STATE_FILE=${STATE_FILE}"
echo "Monitor jobs: squeue -j ${VAL_MEMBER_JOB},${VAL_MEAN_JOB},${AUDIT_JOB}"
