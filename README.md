# UniCache
## CPU-based Prefix Caching Simulator

### Overview

UniCache provides a CPU-based simulator for evaluating **prefix caching** strategies in large language model (LLM) serving systems. This project builds a trace-driven prefix caching simulator based on **vLLM**, a popular open-source LLM inference engine. At a high level, the simulator preserves vLLM’s CPU-side control plane, including request admission, scheduling, and KV cache bookkeeping, while replacing GPU-side model execution with an output simulator that emits predefined tokens from traces.
This design allows us to efficiently replay real-world query traces and evaluate different prefix cache eviction policies without requiring expensive GPU execution.


### Features

- Trace-driven simulation using real-world LLM query patterns
- CPU-based execution without GPU model inference
- Integration with vLLM’s scheduler and KV cache management logic
- Block-level prefix cache hit-rate analysis
- KV block reuse and frequency statistics
- Support for evaluating different cache configurations
- Support for comparing multiple eviction policies, such as:
  - LRU
  - LFU
  - FIFO
  - OPT
  - LeCAR
  - ARC
  - WA
 

### Getting Started

To set up and run the simulator, follow these steps:

- Clone the repository
- Install dependencies:

``` bash
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="simulator"
PY_VER="3.10"
eval "$(/home/shadeform/anaconda3/bin/conda shell.bash hook)"
conda --version

echo "==> Checking conda..."
command -v conda >/dev/null 2>&1 || { echo "❌ conda not found in PATH"; exit 1; }

echo "==> Creating conda env: ${ENV_NAME} (python=${PY_VER})"
 
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Env already exists, skip create."
else
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi

echo "==> Activating env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Upgrading pip"
python -m pip install -U pip

echo "==> Installing CPU PyTorch 2.6.0 stack"
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing project requirements"
python -m pip install -r requirements/build.txt -r requirements/common.txt -r requirements/cpu.txt

echo "==> Installing extra packages"
python -m pip install matplotlib ijson seaborn    openpyxl
python -m pip install jupyter notebook ipykernel modelscope
pip uninstall -y transformers tokenizers
pip install "transformers==4.53.2"  "tokenizers==0.21.2" 
echo "✅ Done. To use:"
echo "   conda activate ${ENV_NAME}"
```

### Quick Start
This demo runs on the sample dataset `ShareGPT_cleaned_valid_session_data_head20.jsonl`, which contains the first 20 sessions from the ShareGPT dataset.

``` bash 
cd demo
bash run.sh
```

**Argument for simulator**



These arguments are inherited directly from vLLM: ([vLLM Engine Arguments](https://docs.vllm.ai/en/v0.6.5/usage/engine_args.html))

| Argument                  | Type / Default                         | Description |
| ------------------------- | -------------------------------------- | ----------- |
| `--model`                 | `str`, default: `Llama-3.2-1B-Instruct` | Path to the local model directory or Hugging Face model name to load. |
| `--max-model-len`         | `int`, default: `4096`                 | Maximum sequence length supported by the model. |
| `--max-num-seqs`          | `int`, default: `32`                   | Maximum number of concurrent sequences. |
| `--block-size`            | `int`, default: `8`                    | Size, in tokens, of a contiguous KV cache block. This determines the cache allocation granularity. |
| `--enable-prefix-caching` | `flag`                                 | Enable prefix caching for repeated prompts. |
| `--eviction-policy`       | `str`, default: `FIFO`                 | KV cache eviction policy. Choices: `FIFO`, `LRU`, `LFU`, `WA`, `SimpleHYBRID`, `SimpleHYBRID-OPT`, `OPT`, `OPT-Online`, `OFFSET-TAIL`, `OFFSET-HEAD`, `LeCaR`, `ARC`, `NONE`. `SimpleHYBRID` is our method|

**Argument for Simulation**

| Argument                          | Type / Default                                              | Description                                                                                                                                                                                                                                    |
| --------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--eviction-policy`       | `str`, default: `FIFO`                 | KV cache eviction policy. Choices: `FIFO`, `LRU`, `LFU`, `WA`, `SimpleHYBRID`, `SimpleHYBRID-OPT`, `OPT`, `OPT-Online`, `OFFSET-TAIL`, `OFFSET-HEAD`, `LeCaR`, `ARC`, `NONE`. `SimpleHYBRID` is our method|
| `--simulator-num-gpu-blocks`        | `int`, default: `100000` | Number of available GPU cache blocks to simulate. |
| `--simulator-output-dir`            | `str`, default: `tmp/` | Directory where simulator outputs are stored. |
| `--simulator-enable-virtual-time`   | `flag` | Enable virtual-time simulation mode. In this mode, the simulator does not wait in real time; instead, it advances time using simulated TTFT and TPOT. |
| `--simulator-task-config`           | `str`, default: `None` | Path to the YAML task configuration file. This file defines per-task simulation parameters, such as dataset path, arrival process, workload parameters, and random seed. |
| `--simulator-ttft`                  | `str`, default: `3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)` | Simulated Time-To-First-Token (TTFT). This can be either a fixed value or a fitting expression based on batch size and input length. |
| `--simulator-tpot`                  | `float`, default: `0.01` | Simulated Time-Per-Output-Token (TPOT), in seconds. |
| `--simulator_wa_fit_interval`       | `int`, default: `1200` | Time interval, in seconds, for periodically fitting the workload-aware model. This option is only effective when `--eviction-policy` is set to `WA`. |
| `--opt-block-use-map-path`          | `str`, default: `None` | Path to the JSON file containing the block future-use map used by the `OPT` eviction policy. If not set, OPT-related eviction is disabled. |
| `--log-level`                       | `str`, default: `INFO` | Logging level for all modules. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--disable-simulation`              | `flag` | Disable the KV-cache simulator and run actual model inference. |
| `--device`                          | `str`, default: `cpu` | Device used for execution, such as `cpu` or `cuda`. |
| `--tensor_parallel_size`            | `int`, default: `1` | Tensor parallel size used for model execution. |
| `--enable-block-hit-track`          | `flag` | Enable block-level hit tracking for the KV cache. |
  

**Arguments for Task Configuration**

Example of a task configuration file (`.yaml`):
``` yaml
tasks:
  - name: ShareGPT-Multi-Turn
    task_id: 0
    data_path: ./demo/ShareGPT_cleaned_valid_session_data_head20.jsonl
    lambda_rate: 1
    mu: 4.15
    sigma: 0.971
    max_trace_duration: 1800
    seed: 42
    max_record:  20
```
| Field                | Type / Default          | Description                                                                                                             |
| -------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `name`               | `str`                   | Unique name of the task. Used to identify different datasets or workloads within the simulation.                        |
| `task_id`            | `int`                   | Numerical ID of the task. Used to generate deterministic seeds and manage task-level randomness.                        |
| `data_path`          | `str`                   | Path to the input dataset file (e.g., `.jsonl` or `.pkl`) containing session and conversation data for simulation.      |
| `lambda_rate`        | `float`, default: `0.5` | Session arrival rate (λ) in sessions per second. Used to generate session start times following a Poisson distribution. |
| `mu`                 | `float`, default: `0.0` | Mean (μ) parameter of the **lognormal distribution** controlling the inter-turn waiting time within each session.       |
| `sigma`              | `float`, default: `0.0` | Standard deviation (σ) of the lognormal distribution controlling inter-turn intervals.                                  |
| `max_trace_duration` | `int`, default: `-1`    | Maximum allowed duration (in seconds) of a single session trace. Use `-1` for no limit.                                 |
| `seed`               | `int`, default: `42`    | Random seed used for deterministic sampling and reproducibility.                                                        |
| `max_record`         | `int`, default: `-1`    | Maximum number of sessions to load from the dataset. Use `-1` to include all available records.                         |

The config file also support hybrid task:

 ``` yaml
tasks:
  - name: AgentBank_apps
    task_id: 0
    data_path: TaskData/AgentBankData/AgentBank_apps_session_data.jsonl
    lambda_rate: 1
    mu: 0
    sigma: 0
    max_trace_duration: -1
    seed: 42
    max_record: 1800

  - name: Codeparrot
    task_id: 1
    data_path: TaskData/Codeparrot/Code_session_data.jsonl
    lambda_rate: 1
    mu: 0
    sigma: 0
    max_trace_duration: -1
    seed: 42
    max_record: 1800

 ```

### How the Simulator Works
The simulator models multi-session concurrent inference with stochastic arrival times and inter-turn intervals.
It approximates the temporal dynamics of a real-world serving environment by maintaining a global virtual clock and processing requests in simulated time order.

- **Session arrival time** is sampled from a Poisson distribution:

  $Arrival\ Time∼Poisson(λ)$

- **Inter-turn interval** within a session follows a Lognormal distribution:

  $Interval∼Lognormal(μ,σ)$

- Multiple sessions run concurrently, each following its own session trace.

- The simulation does not involve any real model computation. Instead, it replays pre-generated outputs to emulate inference behavior without invoking the actual model. In fact, no model is initialized during the simulation.

**Simulation Engine Loop**
``` text
Initialize:
    global_virtual_clock ← 0
    future_request_heap ← ∅  # min-heap ordered by request.timestamp
   

Initialization Phase:
    for each session in all_sessions do
        first_turn ← session.first_turn
        push (first_turn.timestamp, first_turn) into future_request_heap
    end for

Main Loop:
    while (future_request_heap is not empty) 
          or (Engine has unfinished requests) do
    
        # Step 1: Release all requests whose time has arrived
        while (future_request_heap not empty) and
              (future_request_heap.top.timestamp ≤ global_virtual_clock) do
            R_next ← pop from future_request_heap
            Engine.add_request(R_next)
            

        
        if Engine.has_unfinished_requests() then
            request_outputs ← Engine.step()
      
            Δt ← ComputeStepDelta(request_outputs)
            global_virtual_clock ← global_virtual_clock + Δt
            
            for each output in request_outputs do
                if output.is_finished then
                    if output.has_next_turn then
                        next_turn ← output.next_turn
                        push (next_turn.timestamp, next_turn)
                             into future_request_heap
                            
            
         else:
         # If idle but future events exist, jump to next event time
         else if future_request_heap not empty then
            next_time ← future_request_heap.top.timestamp
            global_virtual_clock ← next_time
            
         # If nothing remains, terminate simulation
         else
            break
```
+ The global virtual clock is advanced either by model execution (Δt) or by jumping to the next scheduled event.

+ The future request heap ensures strict chronological order of execution across all sessions.

+ Each request’s next turn is scheduled immediately upon completion of the current turn, using the sampled lognormal inter-turn delay.



### Simulation Data Preparation
The directory `TaskData/` contains subdirectories for preparing data for different simulation tasks.

Each subdirectory includes its own **README** file with detailed instructions on how to preprocess or generate the required datasets.

Data preparation may take some time, but it only needs to be done **once** before running the simulations.

Each session record contains:
+ A unique session ID
+ The number of turns in the session
+ Each turn’s input text and output text
+ The input and output tokens for each turn

## GPU Inference
For GPU-based inference, please set up the environment following the official vLLM installation guide: https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html

In addition, please install LMCache: `pip install lmcache==0.2.1`

Then run the GPU inference demo:
``` bash 
cd demo
bash run_gpu.sh
```


