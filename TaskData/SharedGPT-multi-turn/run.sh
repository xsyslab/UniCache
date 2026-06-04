cd ../..

python run_simulation.py \
    --model "/root/nfs/download/Llama-3.2-1B-Instruct" \
    --max-num-seqs 32 \
    --max-model-len 131072   \
    --block-size 8 \
    --enable-prefix-caching \
    --simulator-enable-virtual-time \
    --simulator-num-gpu-blocks 50000  \
    --simulator-num-cpu-blocks 50000  \
    --eviction-policy "FIFO" \
    --simulator-output-dir  result/SharedGPT-multi-turn/ \
    --simulator-task-config TaskData/SharedGPT-multi-turn/tasks_config.yaml