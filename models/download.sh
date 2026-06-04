cd ../
modelscope download \
  --model unsloth/Llama-3.2-3B \
  --exclude "*.safetensors" \
  --local_dir  ./models/Llama-3.2-3B

modelscope download \
  --model unsloth/Llama-3.2-1B-Instruct \
  --exclude "*.safetensors" \
  --local_dir  ./models/Llama-3.2-1B-Instruct

modelscope download \
  --model unsloth/Llama-3.1-8B-Instruct \
  --exclude "*.safetensors" \
  --local_dir  ./models/Llama-3.1-8B-Instruct

modelscope download \
  --model Qwen/Qwen2.5-32B \
  --exclude "*.safetensors" \
  --local_dir ./models/Qwen2.5-32B

modelscope download \
  --model Qwen/Qwen2.5-14B  \
  --exclude "*.safetensors" \
  --local_dir ./models/Qwen2.5-14B 
modelscope download \
  --model LLM-Research/Meta-Llama-3-70B \
  --exclude "*.safetensors" \
  --local_dir  ./models/Meta-Llama-3-70B