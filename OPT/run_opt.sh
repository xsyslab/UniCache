cd ../
export VLLM_USE_V1=0
PYTHONHASHSEED=0 python run_simulation.py \
    --model "/root/nfs/download/Llama-3.2-1B-Instruct" \
    --max-num-seqs 32 \
    --max-model-len 131072   \
    --block-size 8 \
    --enable-prefix-caching \
    --simulator-enable-virtual-time \
    --simulator-num-gpu-blocks 2000  \
    --simulator-num-cpu-blocks 50000  \
    --eviction-policy "OPT" \
    --log-level "WARNING" \
    --simulator-output-dir  OPT/result_lr5000/ \
    --simulator-task-config OPT/tasks_config.yaml \
    --device "cpu" \
    --opt-block-use-map-path  ./OPT/result_lr5000/NONE2025-12-03_20-56-26/use_map.json \
