#!/bin/bash
set -e  # 出错立即停止
cd ../../../
export VLLM_USE_V1=0

LOG_DIR="TaskData/Hybrid/test1/logs"
mkdir -p "$LOG_DIR"
for block in 50000 60000 70000 80000 90000 100000 ; do
  for policy in    OPT-Online    ;  do
    for ttft in "3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)"; do
      echo "=============================================="
      echo "Running policy=$policy | TTFT=$ttft"
      echo "=============================================="

      timestamp=$(date +"%Y%m%d_%H%M%S")
      LOG_FILE="${LOG_DIR}/policy_${policy}_block_${block}_${timestamp}.log"
      echo "$LOG_FILE"

      PYTHONHASHSEED=0  python run_simulation.py \
        --model /root/nfs/download/Llama-3.2-1B-Instruct \
        --max-num-seqs 32 \
        --max-model-len 131072 \
        --block-size 8 \
        --enable-prefix-caching \
        --simulator-enable-virtual-time \
        --device "cpu" \
        --simulator-num-gpu-blocks $block \
        --simulator-num-cpu-blocks 100000 \
        --eviction-policy "$policy" \
        --simulator-output-dir "result/Hybrid/test1" \
        --simulator-ttft "$ttft" \
        --simulator-tpot 0.01 \
        --log-level "WARNING" \
        --opt-block-use-map-path  result/Hybrid/test1-new/NONE2025-12-16_23-21-56/use_map.json \
        --simulator-task-config "TaskData/Hybrid/test1/tasks_config.yaml" \
        --enable-hybrid-opt \
        2>&1 | tee "$LOG_FILE"
      echo "✅ Finished policy=$policy | TTFT=$ttft"
      echo "📄 Log saved to: $LOG_FILE"
      echo "=============================================="
    done
  done
done