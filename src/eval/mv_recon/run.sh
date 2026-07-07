#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Train/LYJ/miniconda3/envs/OVGGT/bin/python}"
MODEL_NAME="${MODEL_NAME:-OVGGT}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${REPO_ROOT}/ckpt/checkpoints.pth}"
DATA_ROOT="${DATA_ROOT:-/path/to/mount/lyj/OpenDataLab___7-Scenes/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/mv_recon/${MODEL_NAME}_checkpoints}"
MAX_FRAMES="${MAX_FRAMES:-300}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29602}"
OVGGT_MODE="${OVGGT_MODE:-legacy}"

echo "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${SCRIPT_DIR}/launch.py" \
    --weights "${MODEL_WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --ovggt_mode "${OVGGT_MODE}" \
    --data_root "${DATA_ROOT}" \
    --max_frames "${MAX_FRAMES}" \
    "$@"
