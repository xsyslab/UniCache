# SPDX-License-Identifier: Apache-2.0
import hashlib
import heapq
import itertools
import json
import math
import os
import time
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.simulator.config import GLOBAL_SIMULATOR_CONFIG
from vllm.simulator.data_parser import read_simulator_data
from vllm.simulator.global_output_simulator import get_global_output_simulator
from vllm.simulator.opt_utils import create_block_future_use_map
from vllm.simulator.timer_utils import GLOBAL_TIMER
from vllm.simulator.utils import (generate_poisson_requests,
                                  plot_block_hit_cdf, plot_hit_rates,
                                  plot_interval_distribution)
from vllm.simulator.wa_helper import GLOBAL_WA_HELPER

logger = init_logger(__name__)


@dataclass
class SimulatorRequestInput:
    """A single simulation request unit for the engine."""
    request_id: str
    conv_key: str
    prompt: str
    params: SamplingParams


def get_data_from_config(task_configs):
    '''
    Load all session data for each task configuration.
    task_configs: A list of task dictionaries loaded from YAML.
    combined_data: A merged dictionary containing all sessions across tasks.
    separate_data: A nested dictionary {task_name: {session_id: session_data}}.
    '''
    combined_data = {}
    separate_data = {}
    for task in task_configs:
        data_path = task["data_path"]
        max_record = task.get("max_record", -1)

        data = read_simulator_data(data_path, max_record)
        logger.info(
            f"[Simulator] Loading session data for task={task['name']} from {data_path} (max={max_record})"
        )
        logger.info(
            f"[Simulator] Loaded {len(data)} sessions for task={task['name']}")

        combined_data.update(data)
        separate_data[task["name"]] = data
    logger.info(
        f"[Simulator] Combined {len(combined_data)} sessions from {len(task_configs)} tasks."
    )
    return combined_data, separate_data


def build_simulator_request(data: dict, turn_id: int):
    '''
    Build a SimulatorRequestInput object for one conversation turn.
    data: The session data dictionary.
    turn_id: The index of the turn to build.
    '''
    session_id = data["id"]
    total_turns = data["num_turns"]
    logger.debug(
        f"[Simulator] Building request for session id: {session_id }, turn {turn_id}/{total_turns }"
    )
    if turn_id > total_turns:
        logger.info(
            f"Turn {turn_id} exceeds number of turns {total_turns }. No more turns to process."
        )
        return None
    conv_key = f"id_{data['id']}_turn_{turn_id}"
    turn_data = data[f"turn_{turn_id}"]
    simulator_input = SimulatorRequestInput(
        request_id=conv_key,  #deterministic_request_id(session_id, turn_id),
        prompt=turn_data["input_text"],
        conv_key=conv_key,
        params=SamplingParams(temperature=0,
                              max_tokens=len(turn_data["response_ids"])),
    )
    return simulator_input


def get_all_task_params(task_configs):
    '''
    Collect per-task simulation parameters.
    Returns:
        A dictionary {task_name: {lambda_rate, mu, sigma, seed, task_id, ...}}
    '''
    result = {}
    for task in task_configs:
        result[task["name"]] = task
    logger.info(
        f"[Simulator] Parsed parameters for {len(result)} tasks: {list(result.keys())}"
    )
    return result


def create_session_id_2_task_name_mapping(separate_data):
    """
    Build a mapping from session_id to task_name.
    This enables the simulator to retrieve task-level parameters during generating waiting time.
    """
    mapping = {}
    for name, data in separate_data.items():
        session_ids = data.keys()
        for session_id in session_ids:
            mapping[session_id] = name
    logger.debug(
        f"[Simulator] Built session→task mapping for {len(mapping )} sessions."
    )
    return mapping


def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """Convert a string to a stable 32-bit seed."""
    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


def deterministic_request_id(session_id: str, turn_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}::turn_{turn_id}"))


class Simulator:

    def __init__(self, engine, args):
        self.args = args
        self.engine = engine
        self.output_simulator = get_global_output_simulator()
        # Min-heap storing (timestamp, counter, request)
        self.simulation_future_requests = []
        heapq.heapify(self.simulation_future_requests)
        self._heap_counter = itertools.count()
        # Simulation parameters
        with open(args.simulator_task_config) as f:
            config = yaml.safe_load(f)
            self.task_configs = config["tasks"]
            print(self.task_configs)
            logger.info(
                f"[Simulator] Initialized with {len(self.task_configs)} tasks."
            )
        ## === Load all session data ===
        self.session_data, self.separate_data = get_data_from_config(
            self.task_configs)
        self.output_simulator.set_session_data(self.session_data)
        # === Load task parameters and session→task mapping ===
        self.task_paras = get_all_task_params(self.task_configs)
        self.session_id_2_task_name_mapping = create_session_id_2_task_name_mapping(
            self.separate_data)
        # === Initialize virtual time and request heap ===
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            self._validate_timings()
            GLOBAL_TIMER.set_time(0.0)
            self.start_time = 0
        else:
            self.start_time = time.time()
        self._init_session_rngs()
        self.init_heap()
        if GLOBAL_SIMULATOR_CONFIG.disable_simulation:
            self.collected_qttft = {}
            self.collected_ttft = {}

    def _init_session_rngs(self):
        self.session_rngs = {}
        for session_id, task_name in self.session_id_2_task_name_mapping.items(
        ):
            para = self.task_paras[task_name]
            seed = stable_seed_from_str(session_id,
                                        base_seed=para["seed"] +
                                        para["task_id"] * 1000)
            self.session_rngs[session_id] = np.random.default_rng(seed)

    def _validate_timings(self):
        """
            Validate and normalize simulator_ttft and simulator_tpot.
            '3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)'.
        """
        ttft = self.args.simulator_ttft
        if isinstance(ttft, str):
            ttft = ttft.strip()
            if any(token in ttft
                   for token in ["batch_size", "max_len", "**", "*"]):
                self.args.simulator_ttft = ttft
            else:
                # convert to float
                try:
                    self.args.simulator_ttft = float(ttft)
                except ValueError:
                    raise ValueError(f"Invalid simulator_ttft: {ttft}")
        else:
            raise ValueError(
                "simulator_ttft must be a float or a valid formula string.")
        if self.args.simulator_tpot < 0:
            raise ValueError("simulator_tpot must be a non-negative number.")

    def init_heap(self):
        """
        Initialize the heap of 1st turn requests.
        For each task:
            - Generate Poisson-distributed session arrival times.
            - Shuffle session order for randomization.
            - Push the first turn (turn=1) of each session into the heap.
        """
        logger.info(
            f"[Simulator] Initializing future request heap for {len(self.task_paras)} tasks..."
        )
        for name, para in self.task_paras.items():
            task_data = list(self.separate_data[name].values())
            rng = np.random.default_rng(para["seed"] + para["task_id"])
            rng.shuffle(task_data)
            arrival_times = generate_poisson_requests(
                lambda_rate=para["lambda_rate"],
                num_sessions=len(task_data),
                seed=para["seed"] + para["task_id"],
            )
            assert len(arrival_times) == len(task_data)
            # Push each first-turn request into the heap
            for i, arrival_time in enumerate(arrival_times):
                data = task_data[i]
                simulator_input = build_simulator_request(data, turn_id=1)
                assert simulator_input is not None
                count = next(self._heap_counter)
                time_stamp = arrival_time + self.start_time
                heapq.heappush(self.simulation_future_requests,
                               (time_stamp, count, simulator_input))
                logger.debug(
                    f"[Simulator] {simulator_input.request_id} scheduled 1st turn request at time {time_stamp} | delta={arrival_time} "
                )
            logger.debug(
                f"[Simulator] Task={name} scheduled {len(task_data)} initial requests."
            )
        logger.info(
            f"[Simulator] Total initial requests in heap: {len(self.simulation_future_requests)}"
        )
        del self.separate_data

    def engine_loop(self):
        """
        Virtual-time main loop:
        - Release ready requests (ts ≤ global)
        - Step engine if there are unfinished requests
        - Advance virtual time (Δt)
        - Schedule next turns on finished outputs
        - If idle but future events exist, jump clock to next event
        """
        while self.engine.has_unfinished_requests(
        ) or self.simulation_future_requests:
            # Step 1: release all requests that are due
            self._check_future_requests()
            # Step 2: if there are unfinished requests, step the engine
            if self.engine.has_unfinished_requests():
                request_outputs = self.engine.step()
                # Advance virtual clock by Δt for this step
                if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
                    if len(request_outputs) != 0 and isinstance(
                            request_outputs[0], RequestOutput):
                        batch_size = len(request_outputs)

                        is_prefill = len(
                            request_outputs[0].outputs[0].token_ids) == 1
                        if is_prefill:
                            phase = "prefill"
                            ttft_expr = self.args.simulator_ttft
                            if isinstance(ttft_expr, (int, float)):
                                delta_t = ttft_expr
                            elif isinstance(ttft_expr, str):
                                max_len = max(
                                    len(output.prompt_token_ids)
                                    for output in request_outputs)
                                delta_t = eval(
                                    ttft_expr, {
                                        "batch_size": batch_size,
                                        "max_len": max_len,
                                        "math": math
                                    })
                                logger.debug(
                                    f"[Simulator] Prefill batch size: {batch_size}, max len: {max_len}, delta_t: {delta_t}"
                                )
                            else:
                                raise ValueError(
                                    f"Unexpected simulator_ttft type: {type(ttft_expr)}"
                                )
                        else:
                            delta_t = self.args.simulator_tpot
                            phase = "decode"
                            logger.debug(
                                f"[Simulator] Decode batch size: {batch_size}, delta_t: {delta_t}"
                            )
                        GLOBAL_TIMER.advance(delta_t)
                        logger.debug(
                            f"[Simulator] Advanced time by {delta_t:.2f}s for {phase} step. New time: {self.get_time():.2f}s"
                        )
                # For each output, if finished, schedule next turn (if any)
                for output in request_outputs:
                    if isinstance(output, RequestOutput):
                        if output.finished:
                            if GLOBAL_SIMULATOR_CONFIG.disable_simulation:
                                self._collect_stats(output)
                            if self._maybe_schedule_next_turn(output):
                                # After scheduling next turn, immediately check future requests
                                # Avoid jumping over newly scheduled requests
                                self._check_future_requests()
            # Step 3: if no unfinished requests, but future events exist, jump to next event
            if not self.engine.has_unfinished_requests(
            ) and self.simulation_future_requests:
                # Jump to next event time (CPU simulation)
                next_event_time = self.simulation_future_requests[0][0]
                if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
                    if next_event_time > self.get_time():
                        logger.debug(
                            f"[Simulator] Engine idle. Jumping time from {self.get_time():.2f}s to next event at {next_event_time:.2f}s"
                        )
                        GLOBAL_TIMER.set_time(next_event_time)
                else:
                    # Wait until next event time (GPU inference)
                    wait_time = next_event_time - self.get_time()
                    logger.info(
                        f"[Simulator] Engine idle. Waiting {wait_time:.2f}s for next event at {next_event_time:.2f}s"
                    )
                    if wait_time > 0:
                        time.sleep(wait_time)

    def _check_future_requests(self):
        released = 0
        while self.simulation_future_requests and self.simulation_future_requests[
                0][0] <= self.get_time():
            ts, _, req = heapq.heappop(self.simulation_future_requests)
            # Add to engine
            self.engine.add_request(
                request_id=req.request_id,
                prompt=req.prompt,
                params=req.params,
                conv_key=req.conv_key,
            )
            logger.debug(f"[Simulator] Engine Add request {req.request_id} "
                         f"(ts={ts:.2f}, global_time={self.get_time():.2f}) "
                         f"Decode len: {req.params.max_tokens}")
            released += 1
        if released:
            logger.debug(
                f"[Simulator] Released {released} request(s) at t={self.get_time():.2f}"
            )

    def _collect_stats(self, output: RequestOutput):
        # collect stats from RequestOutput
        # logger.info(f"[Simulator] Collecting stats for request {output.request_id}")
        metrics = output.metrics
        if metrics is not None:
            if metrics.arrival_time is not None and metrics.first_token_time is not None:
                qttft = metrics.first_token_time - metrics.arrival_time
                ttft = metrics.first_token_time - metrics.first_scheduled_time
                self.collected_qttft[output.request_id] = qttft
                self.collected_ttft[output.request_id] = ttft

    def _maybe_schedule_next_turn(self, output: RequestOutput):

        request_id = output.request_id
        logger.debug(
            f"[Simulator] request {output.request_id} finish at t={self.get_time():.2f}s"
        )
        conv_key = self.output_simulator.request_id_2_conv_key_map.get(
            request_id, None)
        assert conv_key is not None, f"conv_key not found for request_id {request_id}"
        # Parse session id and turn_id from conv_key
        session_id, turn_id = self.output_simulator.parser_conv_key(conv_key)
        next_turn_id = turn_id + 1
        sample_input = build_simulator_request(self.session_data[session_id],
                                               next_turn_id)
        if sample_input is None:
            logger.info(
                f"[Simulator] Session {session_id} completed all turns.")
            return False

        task_name = self.session_id_2_task_name_mapping[session_id]
        task_para = self.task_paras[task_name]  # generate wait time

        # Deterministic RNG per session
        rng = self.session_rngs[session_id]
        if task_para["mu"] == 0.0 and task_para["sigma"] == 0.0:
            wait_time = 0.0
        else:
            wait_time = rng.lognormal(mean=task_para["mu"],
                                      sigma=task_para["sigma"])
        finished_ts = self.get_time()
        new_ts = finished_ts + wait_time
        # if task_para["max_trace_duration"] != -1 and new_ts > task_para["max_trace_duration"]:
        #     logger.debug(f"[Simulator] Session {session_id} exceeded max trace duration. ")
        #     return False
        count = next(self._heap_counter)
        heapq.heappush(self.simulation_future_requests,
                       (new_ts, count, sample_input))

        logger.debug(
            f"[Simulator] Scheduled next turn {next_turn_id} for session {session_id} after wait {wait_time:.2f}s at t={finished_ts + wait_time:.2f}s"
        )
        return True

    def get_time(self):
        # CPU simulation: use virtual time
        # GPU inference: use real time
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            return GLOBAL_TIMER.get_current_time()
        else:
            return time.time()

    def save_result(self, simulation_time):
        output_dir = self.args.simulator_output_dir
        os.makedirs(output_dir, exist_ok=True)
        # save args
        args_path = os.path.join(output_dir, "args.json")
        if self.args.eviction_policy == "SimpleHYBRID":
            if os.getenv("SimpleHYBRID_Disable_Single_Time_Decay",
                         "0").strip() == "1":
                self.args.eviction_policy = "SimpleHYBRID_Disable_Single_Time_Decay"
            if os.getenv("SimpleHYBRID_Disable_HIT_EFFICIENCY",
                         "0").strip() == "1":
                self.args.eviction_policy = "SimpleHYBRID_Disable_HIT_EFFICIENCY"

        if self.args.eviction_policy == "SimpleHYBRID-OPT":
            if os.getenv("SimpleHYBRID_Disable_Single_Time_Decay",
                         "0").strip() == "1":
                self.args.eviction_policy = "SimpleHYBRID_Disable_Single_Time_Decay-OPT"
            if os.getenv("SimpleHYBRID_Disable_HIT_EFFICIENCY",
                         "0").strip() == "1":
                self.args.eviction_policy = "SimpleHYBRID_Disable_HIT_EFFICIENCY-OPT"

        with open(args_path, "w") as f:
            json.dump(vars(self.args), f, indent=2)

        self.output_simulator.save_metadata()
        summary = {}
        summary["global_time"] = self.get_time() - self.start_time
        summary["simulation_time"] = simulation_time
        summary["simulator_data"] = {
            "max_records":
            len(self.session_data),
            "len_request_id_2_conv_key_map":
            len(self.output_simulator.request_id_2_conv_key_map),
            "len_request_id_2_hit_rate_map":
            len(self.output_simulator.request_id_2_hit_rate_map),
        }
        print(f"[Simulator] Block size: {self.args.block_size} | "
              f"Num GPU blocks: {self.args.simulator_num_gpu_blocks} | "
              f"Eviction policy: {self.args.eviction_policy}")

        hit_rates = list(
            self.output_simulator.request_id_2_hit_rate_map.values())
        if hit_rates:
            summary["hit_rate_summary"] = {
                "total_requests":
                len(hit_rates),
                "average_hit_rate":
                float(np.mean(hit_rates)),
                "min_hit_rate":
                float(min(hit_rates)),
                "max_hit_rate":
                float(max(hit_rates)),
                "std_dev_hit_rate":
                float(np.std(hit_rates)),
                "global_hit_tokens_num":
                self.output_simulator.global_hit_tokens_num,
                "global_tokens":
                self.output_simulator.global_tokens,
                "global_hit_rate":
                self.output_simulator.global_hit_tokens_num /
                self.output_simulator.global_tokens
            }
            if GLOBAL_SIMULATOR_CONFIG.enable_lmcache:
                summary["hit_rate_summary"][
                    "lmcache_hit_token"] = self.output_simulator.lmcache_hit_token
                summary["hit_rate_summary"][
                    "lmcache_hit_rate"] = self.output_simulator.lmcache_hit_token / self.output_simulator.global_tokens
            print(
                f"[HitRateSummary] Total: {summary['hit_rate_summary']['total_requests']} "
                f"| Avg: {summary['hit_rate_summary']['average_hit_rate']:.4f} "
                f"| Min: {summary['hit_rate_summary']['min_hit_rate']:.4f} "
                f"| Max: {summary['hit_rate_summary']['max_hit_rate']:.4f} "
                f"| Std: {summary['hit_rate_summary']['std_dev_hit_rate']:.4f}"
            )
            print(
                f"[GlobalHitStats] HitTokens: {summary['hit_rate_summary']['global_hit_tokens_num']} "
                f"| TotalTokens: {summary['hit_rate_summary']['global_tokens']} "
                f"| GlobalHitRate: {summary['hit_rate_summary']['global_hit_rate']:.4f}"
            )
            plot_hit_rates(hit_rates, output_dir=output_dir)

        if GLOBAL_SIMULATOR_CONFIG.enable_block_hit_track:
            block_id_2_count = self.output_simulator.block_id_2_count
            block_id_2_token_ids = self.output_simulator.block_id_2_token_ids
            if block_id_2_count:
                max_hit_count = max(block_id_2_count.values())
                max_hit_block_ids = [
                    bid for bid, count in block_id_2_count.items()
                    if count == max_hit_count
                ]
            else:
                max_hit_count = 0
                max_hit_block_ids = []
            summary["block_hit_analysis"] = {
                "max_block_hit_count": max_hit_count,
                "block_ids_with_max_hits": max_hit_block_ids,
                "num_blocks_with_max_hits": len(max_hit_block_ids),
            }
            plot_block_hit_cdf(block_id_2_count, output_dir=output_dir)
        if GLOBAL_SIMULATOR_CONFIG.disable_simulation:
            ids = np.array(list(self.collected_qttft.keys()))
            qttfts = np.array(list(self.collected_qttft.values()),
                              dtype=np.float32)
            qttft_values = np.array(list(self.collected_qttft.values()),
                                    dtype=np.float64)
            summary["avg_qttft"] = float(np.mean(qttft_values))

            ids = np.array(list(self.collected_ttft.keys()))
            ttfts = np.array(list(self.collected_ttft.values()),
                             dtype=np.float32)
            ttft_values = np.array(list(self.collected_ttft.values()),
                                   dtype=np.float64)
            summary["avg_ttft"] = float(np.mean(ttft_values))

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        task_config_path = os.path.join(output_dir, "config_out.yaml")
        with open(task_config_path, "w") as f:
            yaml.safe_dump({"tasks": self.task_configs},
                           f,
                           sort_keys=False,
                           allow_unicode=True,
                           default_flow_style=False)
        if self.args.eviction_policy == "WA":
            # save all collected intervals
            all_data = GLOBAL_WA_HELPER.all_data
            for task_name, data in all_data.items():
                if not data:
                    continue
                df = pd.DataFrame({"interval": data})
                csv_path = os.path.join(output_dir,
                                        f"{task_name}_intervals.csv")
                df.to_csv(csv_path, index=False)
                logger.info(
                    f"[WA] Saved interval data for {task_name} ({len(data)} samples) → {csv_path}"
                )
                try:
                    plot_interval_distribution(task_name,
                                               data,
                                               save_dir=output_dir)
                    logger.info(
                        f"[WA] Saved distribution plots for {task_name} → {output_dir}/{task_name}_distribution.png"
                    )
                except Exception as e:
                    logger.warning(f"[WA] Failed to plot {task_name}: {e}")
        # save collected QTTFT
        if GLOBAL_SIMULATOR_CONFIG.disable_simulation:
            ids = np.array(list(self.collected_qttft.keys()))
            qttfts = np.array(list(self.collected_qttft.values()),
                              dtype=np.float32)
            np.savez_compressed(os.path.join(output_dir, "qttft_results.npz"),
                                request_id=ids,
                                qttft=qttfts)
            print(
                f"[Simulator] Saved QTTFT binary results → {output_dir}/qttft_results.npz"
            )
            qttft_values = np.array(list(self.collected_qttft.values()),
                                    dtype=np.float64)
            summary = {
                "count": len(qttft_values),
                "avg": float(np.mean(qttft_values)),
                "p80": float(np.percentile(qttft_values, 80)),
                "p90": float(np.percentile(qttft_values, 90)),
                "min": float(np.min(qttft_values)),
                "max": float(np.max(qttft_values)),
                "std": float(np.std(qttft_values)),
            }
            print(f"[QTTFT Summary] count={summary['count']} | "
                  f"avg={summary['avg']:.4f}s | p80={summary['p80']:.4f}s | "
                  f"p90={summary['p90']:.4f}s | min={summary['min']:.4f}s | "
                  f"max={summary['max']:.4f}s | std={summary['std']:.4f}s")
            ids = np.array(list(self.collected_ttft.keys()))
            ttfts = np.array(list(self.collected_ttft.values()),
                             dtype=np.float32)

            np.savez_compressed(os.path.join(output_dir, "ttft_results.npz"),
                                request_id=ids,
                                ttft=ttfts)
            print(
                f"[Simulator] Saved TTFT binary results → {output_dir}/ttft_results.npz"
            )

            ttft_values = np.array(list(self.collected_ttft.values()),
                                   dtype=np.float64)
            summary = {
                "count": len(ttft_values),
                "avg": float(np.mean(ttft_values)),
                "p80": float(np.percentile(ttft_values, 80)),
                "p90": float(np.percentile(ttft_values, 90)),
                "min": float(np.min(ttft_values)),
                "max": float(np.max(ttft_values)),
                "std": float(np.std(ttft_values)),
            }
            print(f"[TTFT Summary] count={summary['count']} | "
                  f"avg={summary['avg']:.4f}s | p80={summary['p80']:.4f}s | "
                  f"p90={summary['p90']:.4f}s | min={summary['min']:.4f}s | "
                  f"max={summary['max']:.4f}s | std={summary['std']:.4f}s")
        if os.getenv("COLLECT_TRACE", "0") == "1":
            trace_path = os.path.join(output_dir, "trace_data.json")
            print(f"[Simulator] Saving trace data to {trace_path}")
            with open(trace_path, "w") as f:
                json.dump(self.engine.trace_data, f, indent=2)
            # create block reuse time data
            # key: content hash
            # value: reuse iteration index
            use_map_path = os.path.join(output_dir, "use_map.json")
            create_block_future_use_map(trace_path, self.session_data,
                                        self.args.block_size, use_map_path)
            print(f"[Simulator] Saving block use map to {use_map_path}")
