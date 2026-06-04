export CUDA_VISIBLE_DEVICES=2
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
LM_CACHE_DISABLE_STORAGE_CONNECTORS=1 
cd ../
python run_gpu.py \
    --model "/root/nfs/download/Llama-3.2-1B-Instruct" \
    --max-num-seqs 32 \
    --max-model-len 131072   \
    --block-size 32 \
    --enable-prefix-caching \
    --simulator-num-gpu-blocks 10000  \
    --simulator-num-cpu-blocks 0  \
    --eviction-policy "FIFO" \
    --simulator-output-dir  ./demo/result/ \
    --simulator-task-config ./demo/tasks_config.yaml \
    --device "cuda" \
    --disable-simulation \
    --enable-lmcache \
    --lmcache-chunk-size 32 \
    --lmcache-local-cpu  \
    --lmcache-max-local-cpu-size 0.5 \
    # --simulator-enable-virtual-time \