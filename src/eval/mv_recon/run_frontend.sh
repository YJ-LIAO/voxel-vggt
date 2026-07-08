#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Train/LYJ/miniconda3/envs/OVGGT/bin/python}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${REPO_ROOT}/ckpt/checkpoints.pth}"
DATA_ROOT="${DATA_ROOT:-/path/to/mount/lyj/OpenDataLab___7-Scenes/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/mv_recon/OVGGT_checkpoints_frontend}"
MAX_FRAMES="${MAX_FRAMES:-200}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29603}"
FRONTEND_ANCHOR_INTERVAL="${FRONTEND_ANCHOR_INTERVAL:-9}"
FRONTEND_DEDUP_BUDGET_TRIGGER_RATIO="${FRONTEND_DEDUP_BUDGET_TRIGGER_RATIO:-1.00}"

echo "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${SCRIPT_DIR}/launch.py" \
    --weights "${MODEL_WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "OVGGT" \
    --ovggt_mode "frontend_eval" \
    --data_root "${DATA_ROOT}" \
    --frontend_anchor_interval "${FRONTEND_ANCHOR_INTERVAL}" \
    --frontend_dedup_budget_trigger_ratio "${FRONTEND_DEDUP_BUDGET_TRIGGER_RATIO}" \
    --max_frames "${MAX_FRAMES}" \
    "$@"
