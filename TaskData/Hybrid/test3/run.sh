#!/bin/bash
set -e  # 出错立即停止
cd ../../../

LOG_DIR="TaskData/Hybrid/test3/logs"
mkdir -p "$LOG_DIR"

for block in 50000   60000 70000 80000 90000 100000  ; do
  for policy in  SimpleHYBRID  ; do
    echo "=============================================="
    echo "Running policy=$policy | TTFT=$ttft"
    echo "=============================================="
    timestamp=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/policy_${policy}block_${block}_${timestamp}.log"
    python run_simulation.py \
      --model /root/nfs/download/Llama-3.2-1B-Instruct \
      --max-num-seqs 32 \
      --max-model-len 131072 \
      --block-size 8 \
      --enable-prefix-caching  \
      --simulator-enable-virtual-time \
      --simulator-num-gpu-blocks $block \
      --simulator-num-cpu-blocks 100000 \
      --eviction-policy "$policy" \
      --simulator-output-dir  "result/Hybrid/test3" \
      --simulator-ttft  "3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)" \
      --simulator-tpot 0.01 \
      --simulator-task-config "TaskData/Hybrid/test3/tasks_config.yaml" \
      --tag "V9: based on V8| fix access time| add multiple task" \
      >"$LOG_FILE" 2>&1
    echo "✅ Finished policy=$policy | TTFT=$ttft"
    echo "📄 Log saved to: $LOG_FILE"
    echo "=============================================="
  done  
done
      # --enable-hybrid-opt \