#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_V1=0
# script directory (where this .sh lives)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# If you want tasks_config.yaml relative to where you launched the script:
# dir="$(pwd)"
# Better: tasks_config.yaml lives next to this script (common pattern)
dir="$SCRIPT_DIR"

# Go to repo root (adjust if needed)
cd "$SCRIPT_DIR/../../../"
REPO_ROOT="$(pwd)"

export SimpleHYBRID_Disable_Single_Time_Decay=1
policies=( "LRU" "LFU" "WA" "FIFO" "SimpleHYBRID"   )
block_counts=( 50000 80000 10000 ) #13532 18945  24358 

TTFT_1B_A100="1.125167e-06 * (batch_size ** 1.124) * (max_len ** 1.333)"
TPOT=0.005
MODEL=models/Llama-3.2-1B-Instruct
# Ensure output directory exists
OUT_DIR="$dir/1B_A100_TPOT$TPOT/"
mkdir -p "$OUT_DIR"
for number in "${block_counts[@]}"; do
  for policy in "${policies[@]}"; do
    echo "Running simulation: Policy=$policy, Blocks=$number"

    PYTHONHASHSEED=0 python run_simulation.py \
      --model "$MODEL" \
      --max-num-seqs 32 \
      --max-model-len 32768 \
      --block-size 16 \
      --enable-prefix-caching \
      --simulator-enable-virtual-time \
      --device "cpu" \
      --simulator-num-gpu-blocks "$number" \
      --simulator-num-cpu-blocks 0 \
      --eviction-policy "$policy" \
      --simulator-task-config "$dir/tasks_config.yaml" \
      --simulator-ttft "$TTFT_1B_A100" \
      --simulator-tpot $TPOT \
      --log-level "WARNING" \
      --simulator-output-dir "$OUT_DIR" 
    #   --opt-block-use-map-path "$REPO_ROOT/experiment/evaluation/HitRatio/Config1-2/Llama-3.1-8B-Instruct/8B_A100_TPOT0.0145/24358_NONE_2026-01-04_01-14-42/use_map.json"
  done
done
