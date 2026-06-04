cd ../..
export VLLM_USE_V1=0
export COLLECT_TRACE=0
for policy in  OPT-Online    ; do
    PYTHONHASHSEED=0 python run_simulation.py \
        --device "cpu" \
        --model "/root/nfs/download/Llama-3.2-1B-Instruct" \
        --max-num-seqs 32 \
        --max-model-len 131072   \
        --block-size 32 \
        --enable-prefix-caching \
        --simulator-enable-virtual-time \
        --simulator-num-gpu-blocks 10000  \
        --simulator-num-cpu-blocks 50000  \
        --eviction-policy "$policy" \
        --simulator-ttft "3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)" \
        --simulator-tpot 0.01 \
        --log-level "warning" \
        --simulator-output-dir  result/ToolBench-lr5000/ \
        --simulator-task-config TaskData/ToolBench/tasks_config.yaml \
        --opt-block-use-map-path  result/ToolBench-lr5000/NONE2025-12-03_20-30-21/use_map.json
done