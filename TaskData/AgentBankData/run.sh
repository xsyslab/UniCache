cd ../..

mkdir -p TaskData/AgentBankData/AgentBank_apps

python run_simulation.py \
    --model "/root/nfs/download/Llama-3.2-1B-Instruct" \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --block-size 8 \
    --enable-prefix-caching \
    --simulator-enable-virtual-time \
    --simulator-num-gpu-blocks 10000 \
    --simulator-num-cpu-blocks 10000 \
    --eviction-policy "LFU" \
    --simulator-ttft "3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)" \
    --tag "V9: based on V8| fix access time" \
    --simulator-tpot 0.01 \
    --simulator-output-dir result/AgentBank_apps/ \
    --simulator-task-config TaskData/AgentBankData/tasks_config.yaml \
    > TaskData/AgentBankData/AgentBank_apps/simulation.log 2>&1
