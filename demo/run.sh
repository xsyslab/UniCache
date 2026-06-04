cd ../
export VLLM_USE_V1=0
export VLLM_PLATFORM=cpu

python run_simulation.py \
    --model "./models/Llama-3.1-8B-Instruct" \
    --max-num-seqs 32 \
    --max-model-len 131072   \
    --block-size 8 \
    --enable-prefix-caching \
    --simulator-enable-virtual-time \
    --simulator-num-gpu-blocks 10000  \
    --simulator-num-cpu-blocks 50000  \
    --eviction-policy "SimpleHYBRID" \
    --simulator-output-dir  demo/result/ \
    --simulator-task-config demo/tasks_config.yaml \
    --device "cpu" \