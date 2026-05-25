#!/bin/bash
# Run 7-Scenes evaluation across 4 GPUs (4-7), 4 test sequences in parallel.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/eval_7scenes_compare.py"
CHECKPOINT="/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth"
NUM_FRAMES=200
BUDGET=200000

echo "Starting 7-Scenes evaluation on GPU 4-7..."
echo "Checkpoint: $CHECKPOINT"
echo "Frames: $NUM_FRAMES | Budget: $BUDGET"
echo "================================================"

# Clean up old results
rm -f "$SCRIPT_DIR/eval_results"/*.json 2>/dev/null || true

# Launch 4 parallel jobs
CUDA_VISIBLE_DEVICES=4 python "$SCRIPT" --scene_id chess/seq-03 --gpu_id 0 --checkpoint "$CHECKPOINT" --num_frames "$NUM_FRAMES" --budget "$BUDGET" &
PID1=$!

CUDA_VISIBLE_DEVICES=5 python "$SCRIPT" --scene_id fire/seq-03 --gpu_id 0 --checkpoint "$CHECKPOINT" --num_frames "$NUM_FRAMES" --budget "$BUDGET" &
PID2=$!

CUDA_VISIBLE_DEVICES=6 python "$SCRIPT" --scene_id office/seq-05 --gpu_id 0 --checkpoint "$CHECKPOINT" --num_frames "$NUM_FRAMES" --budget "$BUDGET" &
PID3=$!

CUDA_VISIBLE_DEVICES=7 python "$SCRIPT" --scene_id redkitchen/seq-03 --gpu_id 0 --checkpoint "$CHECKPOINT" --num_frames "$NUM_FRAMES" --budget "$BUDGET" &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for all jobs to complete..."

wait $PID1 && echo "chess/seq-03 DONE" || echo "chess/seq-03 FAILED"
wait $PID2 && echo "fire/seq-03 DONE" || echo "fire/seq-03 FAILED"
wait $PID3 && echo "office/seq-05 DONE" || echo "office/seq-05 FAILED"
wait $PID4 && echo "redkitchen/seq-03 DONE" || echo "redkitchen/seq-03 FAILED"

echo "================================================"
echo "Summarizing results..."

python "$SCRIPT" --summarize
