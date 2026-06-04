#!/usr/bin/env bash
set -eu
(set -o pipefail) 2>/dev/null || true

model_dir="models/Llama-3.1-8B-Instruct" # Llama-3.1-8B-Instruct
model_base="${model_dir##*/}"

cur_dir="$(pwd)"                 
repo_root="$(cd "$cur_dir/../.." && pwd)"  

out_dir="${cur_dir}/${model_base}"
mkdir -p "$out_dir"
json_list=(AgentBank_alfred_scale_5000.json AgentBank_apps.json AgentBank_alfred.json  )
for file_name in "${json_list[@]}"; do
    json_file="$cur_dir/$file_name"
    [ -f "$json_file" ] || { echo "跳过: $file_name 不存在"; continue; }
    base_name="$(basename "$json_file" .json)"
    new_file="${out_dir}/${base_name}_session_data.jsonl"   

    echo "Processing $json_file..."
    python "${repo_root}/prepare_session_data.py" \
        --raw_data_path "$json_file" \
        --save_data_path "$new_file" \
        --tokenizer_path "${repo_root}/${model_dir}"

    echo "Processed $json_file -> $new_file"
done
