#!/bin/bash
set -e 
cd ../../../
export VLLM_USE_V1=0
export COLLECT_TRACE=1
block=100000
policy="NONE"
ttft="3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)"

PYTHONHASHSEED=0 python run_simulation.py \
    --model /root/nfs/download/Llama-3.2-1B-Instruct \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --block-size 8 \
    --simulator-enable-virtual-time \
    --device "cpu" \
    --simulator-num-gpu-blocks $block \
    --simulator-num-cpu-blocks 100000 \
    --eviction-policy "$policy" \
    --simulator-output-dir "result/Hybrid/test1-new" \
    --simulator-ttft "$ttft" \
    --simulator-tpot 0.01 \
    --log-level "WARNING" \
    --simulator-task-config "TaskData/Hybrid/test1/tasks_config.yaml" \
    2>&1 | tee "$LOG_FILE"

echo "✅ Finished policy=$policy | TTFT=$ttft"
echo "📄 Log saved to: $LOG_FILE"
echo "=============================================="
