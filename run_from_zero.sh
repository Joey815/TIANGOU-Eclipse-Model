#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT="${1:?Usage: run_from_zero.sh YYYY-MM-DD [HH:MM:SS] [HH:MM:SS]}"
START_CLOCK="${2:-15:00:00}"
END_CLOCK="${3:-21:00:00}"
DATA_ROOT="${TIANGOU_DATA_ROOT:-${ROOT}/data}"
VENV_DIR="${TIANGOU_VENV_DIR:-${ROOT}/.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"
fi

PREPARE_ARGS=()
if [[ "${SKIP_DEM_DOWNLOAD:-0}" == "1" ]]; then
    PREPARE_ARGS+=(--skip-dem)
fi

"${VENV_DIR}/bin/python" "${ROOT}/scripts/prepare_event.py" \
    --event "${EVENT}" \
    --start "${START_CLOCK}" \
    --end "${END_CLOCK}" \
    --data-root "${DATA_ROOT}" \
    "${PREPARE_ARGS[@]}"

"${VENV_DIR}/bin/python" "${ROOT}/scripts/preflight.py" \
    --event "${EVENT}" \
    --data-root "${DATA_ROOT}" \
    --json-output "${DATA_ROOT}/preflight_${EVENT//-/}.json"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    exit 0
fi

"${VENV_DIR}/bin/python" "${ROOT}/scripts/run_event.py" \
    --event "${EVENT}" \
    --start "${START_CLOCK}" \
    --end "${END_CLOCK}" \
    --data-root "${DATA_ROOT}" \
    --workers "${TIANGOU_WORKERS:-1}" \
    --skip-preflight
