# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import time
from datetime import datetime

from vllm import LLM
from vllm.logger import init_logger
from vllm.simulator.config import setup_simulator_config
from vllm.simulator.simulator import Simulator

logger = init_logger("vllm.run_simulation")


def parser_args():
    parser = argparse.ArgumentParser(description="Run simulation")
    # === Core options ===
    parser.add_argument("--model",
                        type=str,
                        default="Llama-3.2-1B-Instruct",
                        help="Path or Hugging Face name of the model")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=32,
                        help="Maximum number of concurrent sequences")
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Size of a contiguous cache block in number of tokens")
    parser.add_argument("--enable-prefix-caching",
                        action="store_true",
                        help="Enable prefix caching for repeated prompts")
    parser.add_argument("--eviction-policy",
                        default="FIFO",
                        help="KV cache eviction policy",
                        choices=[
                            "FIFO", "LRU", "LFU", "WA", "SimpleHYBRID",
                            "SimpleHYBRID-OPT", "OPT", "OPT-Online",
                            "OFFSET-TAIL", "OFFSET-HEAD", "LeCaR", "ARC",
                            "NONE"
                        ])
    # === Simulation ===
    parser.add_argument("--simulator-num-gpu-blocks",
                        type=int,
                        default=100000,
                        help="GPU block limit for simulator")
    parser.add_argument("--simulator-num-cpu-blocks",
                        type=int,
                        default=100000,
                        help="CPU block limit for simulator")
    parser.add_argument("--simulator-output-dir",
                        help="Directory for simulator outputs",
                        default="tmp/")
    parser.add_argument(
        "--simulator-enable-virtual-time",
        action="store_true",
        help="Enable virtual-time mode for simulation (no real-time waiting).")
    parser.add_argument("--simulator-task-config",
                        type=str,
                        help="Path to the task configuration YAML file.")
    parser.add_argument(
        "--simulator-ttft",
        type=str,
        default="3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)",
        help=
        ("Simulated TTFT value. "
         "Can be either a fixed float (e.g., 0.1) or a string "
         "to use empirical fitting TTFT = 3.770845e-05 * batch^0.998 * input_len^1.018."
         ),
    )

    parser.add_argument(
        "--simulator-tpot",
        type=float,
        default=0.01,
        help="Simulated Time-To-Output-Token (TOPT) in seconds.")
    parser.add_argument(
        "--simulator_wa_fit_interval",
        type=int,
        default=1200,
        help=
        ("Time interval (in seconds) for periodically fitting the workload-aware model. "
         "This option is only effective when the eviction policy is set to 'WA'."
         ),
    )

    parser.add_argument(
        "--opt-block-use-map-path",
        type=str,
        default=None,
        help=(
            "Path to the JSON file containing the block future-use map used "
            "by the OPT eviction policy. If not set, OPT eviction is disabled."
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for all modules (case-insensitive).",
    )
    parser.add_argument(
        "--disable-simulation",
        action="store_true",
        help="Disable the KV-cache simulator and run model inference.",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        default="cpu",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--enable-block-hit-track",
        action="store_true",
        help="Enable block-level hit tracking for KV-cache.",
    )

    # === LLM Cache options ===
    parser.add_argument("--enable-lmcache", action="store_true")
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument("--lmcache-local-cpu", action="store_true")
    parser.add_argument("--lmcache-max-local-cpu-size",
                        type=float,
                        default=0.1)
    args = parser.parse_args()
    return args


def run_simulation(llm, args):
    print("[Simulator] Simulation started.")
    print("Initialized Engine successfully...")
    start_time = time.perf_counter()
    simulator = Simulator(llm.llm_engine, args)
    simulator.engine_loop()
    end = time.perf_counter()
    duration = end - start_time

    print("=" * 60)
    print("🎯 [Simulator] Simulation completed successfully.")
    print(f"⏱️  Total runtime        : {duration:.2f} seconds")
    print(
        f"🕒  Final global time    : {simulator.get_time()-simulator.start_time:.2f}"
    )
    print("=" * 60)
    simulator.save_result(duration)


def main():
    args = parser_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("vllm").setLevel(log_level)
    logging.getLogger("vllm.run_simulation").setLevel(log_level)
    folder_name = f"{args.simulator_num_gpu_blocks}_{args.eviction_policy}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    args.simulator_output_dir = os.path.join(args.simulator_output_dir,
                                             folder_name)
    print(args)
    setup_simulator_config(args)
    if args.enable_prefix_caching:
        assert args.eviction_policy != "NONE", "Prefix caching is not supported with NONE eviction policy"
    if args.eviction_policy == "OPT":
        assert args.opt_block_use_map_path is not None, "OPT eviction policy requires opt_block_use_map_path"
    llm = LLM(
        model=args.model,
        device=args.device,
        task="generate",
        gpu_memory_utilization=0.8,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        cpu_offload_gb=0,
        tensor_parallel_size=args.tensor_parallel_size,
        swap_space=0,
        block_size=args.block_size,
        enable_prefix_caching=args.enable_prefix_caching,
        load_format="dummy",
        enable_chunked_prefill=False,
    )
    run_simulation(llm, args)


if __name__ == "__main__":
    main()
