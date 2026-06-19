#!/usr/bin/env bash
# Batch runner for the P1 ablation. Distributes (config, scene, seed) jobs across GPUs 6-7.
#
# Smoke test (7 configs x 1 scene x 1 seed, 200 frames) — verifies all configs load + sane ATE:
#   bash tools/batch_p1_ablation.sh smoke
#
# Full Stage-1 matrix (7 configs x 4 scenes x 5 seeds x {200,500,1000} frames) = 420 runs:
#   bash tools/batch_p1_ablation.sh full
# Resumable: re-running skips any (config,scene,nf,seed) whose JSON already exists, so a
# killed/interrupted nohup can be restarted by just re-running the same command.
#
# Determinism already verified bit-identical (no --deterministic flag needed; paired design valid).
set -u

REPO=/path/to/mount/lyj/voxel-vggt
source /mnt/lyj/miniconda3/etc/profile.d/conda.sh && conda activate OVGGT
cd "$REPO"

GPUS=(6 7)
CONFIGS=("P1-none" "P1-current" "P1-v1cap" "P1-mechC-r0.1" "P1-mechC-r0.2" "P1-mechC-r0.3" "P1-mechC-r0.5")
P2_CONFIGS=("P2-w0.3" "P2-w0.5" "P2-w0.7")
SCENES=("chess/seq-03" "fire/seq-03" "office/seq-03" "redkitchen/seq-03")
SEEDS=(0 1 2 3 4)
MODE=${1:-smoke}

RES_DIR="tools/p1_ablation_results_lyj"

run_one () {
    local gpu=$1 cfg=$2 scene=$3 nf=$4 seed=$5
    local tag="$MODE"
    # Resumability: skip if a valid JSON output already exists for this (cfg,scene,nf,seed,tag).
    local out="${RES_DIR}/${cfg}_${scene//\//_}_f${nf}_s${seed}_${tag}.json"
    if [ -s "$out" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then
        echo "  [skip] $cfg $scene f${nf} s${seed} (exists)"; return 0
    fi
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src python tools/run_p1_ablation.py \
        --config "$cfg" --scene "$scene" --num_frames "$nf" --seed "$seed" --tag "$tag" \
        > "${RES_DIR}/log_${cfg}_${scene//\//_}_f${nf}_s${seed}_${tag}.txt" 2>&1
}

if [ "$MODE" = "smoke" ]; then
    # 7 configs x chess/seq-03 x seed0 x 200 frames, 2 GPUs -> 4 rounds
    SMOKE_NF=200; SMOKE_SCENE="chess/seq-03"; SMOKE_SEED=0
    echo "[smoke] ${#CONFIGS[@]} configs x $SMOKE_SCENE x seed$SMOKE_SEED x ${SMOKE_NF}f on ${GPUS[*]}"
    i=0
    for cfg in "${CONFIGS[@]}"; do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        echo "  -> GPU$gpu $cfg"
        run_one "$gpu" "$cfg" "$SMOKE_SCENE" "$SMOKE_NF" "$SMOKE_SEED" &
        i=$((i+1))
        [ $((i % ${#GPUS[@]})) -eq 0 ] && wait
    done
    wait
    echo "[smoke] done. Results in tools/p1_ablation_results_lyj/"

elif [ "$MODE" = "full" ]; then
    # Full Stage-1 matrix: 7 configs x 4 scenes x 5 seeds x {200,500,1000} frames = 420 runs.
    # Outer loop is frame-count so each layer completes fully before the next (incremental analysis,
    # and 1000f — the decisive long-sequence regime — runs last after early tiers are inspectable).
    # 2 GPUs run in parallel; resumable (skips runs whose JSON already exists).
    FRAME_COUNTS=(200 500 1000)
    total=$((${#CONFIGS[@]} * ${#SCENES[@]} * ${#SEEDS[@]} * ${#FRAME_COUNTS[@]}))
    echo "[full] $total runs target on ${GPUS[*]} (resumable: skips completed)"
    for nf in "${FRAME_COUNTS[@]}"; do
      echo "=== frame tier: ${nf}f ==="
      i=0
      for cfg in "${CONFIGS[@]}"; do
        for scene in "${SCENES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            gpu=${GPUS[$((i % ${#GPUS[@]}))]}
            echo "  [$(date +%H:%M:%S)] GPU$gpu $cfg $scene f${nf} s${seed}"
            run_one "$gpu" "$cfg" "$scene" "$nf" "$seed" &
            i=$((i+1))
            [ $((i % ${#GPUS[@]})) -eq 0 ] && wait
          done
        done
      done
      wait
      n_have=$(ls "${RES_DIR}"/*_f${nf}_s*_full.json 2>/dev/null | wc -l)
      echo "=== ${nf}f tier complete: ${n_have} JSONs present ==="
    done
    echo "[full] done. Results in ${RES_DIR}/"

elif [ "$MODE" = "p2" ]; then
    # P2 Phase-1: eviction weight sweep on the production baseline (ring0.2 +
    # budget8334 + intra ON + uniform). 3 weights x 4 scenes x 5 seeds x {500,1000}f
    # = 120 runs. 200f skipped (non-discriminative for scoring). GPUs 4-7 default;
    # override with: bash tools/batch_p1_ablation.sh p2 4 5 6 7
    shift || true
    PGPU=("$@"); [ ${#PGPU[@]} -eq 0 ] && PGPU=(4 5 6 7)
    P2_FRAME_COUNTS=(500 1000)
    total=$((${#P2_CONFIGS[@]} * ${#SCENES[@]} * ${#SEEDS[@]} * ${#P2_FRAME_COUNTS[@]}))
    echo "[p2] $total runs target on ${PGPU[*]} (weights: ${P2_CONFIGS[*]}; resumable)"
    for nf in "${P2_FRAME_COUNTS[@]}"; do
      echo "=== P2 frame tier: ${nf}f ==="
      i=0
      for cfg in "${P2_CONFIGS[@]}"; do
        for scene in "${SCENES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            gpu=${PGPU[$((i % ${#PGPU[@]}))]}
            echo "  [$(date +%H:%M:%S)] GPU$gpu $cfg $scene f${nf} s${seed}"
            run_one "$gpu" "$cfg" "$scene" "$nf" "$seed" &
            i=$((i+1))
            [ $((i % ${#PGPU[@]})) -eq 0 ] && wait
          done
        done
      done
      wait
      n_have=$(ls "${RES_DIR}"/P2-*_f${nf}_s*_p2.json 2>/dev/null | wc -l)
      echo "=== P2 ${nf}f tier complete: ${n_have} JSONs present ==="
    done
    echo "[p2] done. Results in ${RES_DIR}/"
else
    echo "Usage: bash tools/batch_p1_ablation.sh {smoke|full|p2 [gpu...]}"
    exit 1
fi
