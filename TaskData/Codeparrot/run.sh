cd ../..
for policy in LRU LFU FIFO; do
    python run_simulation.py \
        --model "/root/nfs/download/Llama-3.2-1B-Instruct" \
        --max-num-seqs 32 \
        --max-model-len 131072   \
        --block-size 8 \
        --enable-prefix-caching \
        --simulator-enable-virtual-time \
        --simulator-num-gpu-blocks 50000  \
        --simulator-num-cpu-blocks 50000  \
        --eviction-policy "$policy" \
        --simulator-ttft "3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)" \
        --simulator-tpot 0.01 \
        --simulator-output-dir  result/Codeparrot/ \
        --simulator-task-config TaskData/Codeparrot/tasks_config.yaml
done