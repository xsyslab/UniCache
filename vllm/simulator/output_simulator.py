# SPDX-License-Identifier: Apache-2.0
import os
import pickle
import time
from collections import deque
from typing import List

from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceGroupMetadata, SequenceOutput)
from vllm.simulator.config import GLOBAL_SIMULATOR_CONFIG, SimulatorConfig

logger = init_logger(__name__)


class OutputSimulator:

    def __init__(self, simulator_config: SimulatorConfig, tokenizer,
                 eviction_policy, block_size):
        logger.info(
            f"[Simulator] Init OutputSimulator with configuration: {simulator_config}"
        )
        self.simulator_config = simulator_config
        self.eos_token_id = tokenizer.eos_token_id
        self.eviction_policy = eviction_policy
        self.block_size = block_size
        self.output_dir = simulator_config.simulator_output_dir
        self.session_data = None
        # Mapping from vLLM-assigned request ID to conversation key
        # e.g., request_id='cmpl-17c838e152484c72bb25782aee00ece0-0'
        #       → conv_key='id_QWJhYvA_0_turn_1'
        # conversation key is used to fetch data from self.session_data
        self.request_id_2_conv_key_map: dict[str, str] = {}
        # Tracks hit rate statistics per request
        self.request_id_2_hit_rate_map: dict[str, float] = {}
        self.request_id_2_hit_count_map: dict[str, tuple] = {}
        # Track hit tokens
        self.global_hit_tokens_num = 0
        self.global_tokens = 0
        # Mapping from block ID to count
        if GLOBAL_SIMULATOR_CONFIG.enable_block_hit_track:
            self.processed_requests_for_block: set[str] = set()
            self.block_id_2_count: dict[int, int] = {}
            self.block_id_2_token_ids: dict[int, list] = {}
            # self.content_2_new_ids: dict[tuple[int, ...], int] = {}
            self.archive_id = simulator_config.simulator_num_gpu_blocks
            # track reuse time for multi-turn
            self.reuse_time: dict[str, List[float]] = {
            }  # conversation key to reuse time

        # record_cache_hit for task
        self.DECAY_QUEUES = [
            "single", "text", "file", "image", "search", "agentic"
        ]
        self.tracked_hit_stats = {
            q: deque(maxlen=500)
            for q in self.DECAY_QUEUES
        }
        self.step_track = 0
        # track hit token when LM Cache is used, called in model_runner
        if GLOBAL_SIMULATOR_CONFIG.enable_lmcache:
            self.lmcache_hit_token = 0

    def bind_request_id(self, request_id: str, conv_key: str) -> None:
        """
        Bind a vLLM request ID to a conversation key.
            - request_id: the unique identifier used by vLLM.
            - conv_key: a unique string derived from the dataset, 
              typically formatted as 'id_<session_id>_turn_<t>'.
        
        Notes:
        - One conversation (conv_key) can be sent to the server multiple times,
        resulting in multiple request_ids pointing to the same conv_key.
        - But each request_id must only be bound once (no reassignment allowed).
        """
        if request_id in self.request_id_2_conv_key_map:
            raise ValueError(
                f"[Simulator] request_id [{request_id}] is already bound to "
                f"[{self.request_id_2_conv_key_map[request_id]}], "
                f"cannot rebind to [{conv_key}].")
        logger.debug("[Simulator] Bind request_id=[%s], conv_key=[%s]",
                     request_id, conv_key)

        self.request_id_2_conv_key_map[request_id] = conv_key

    def update_hit_token_for_lmcache(self, computed_token):
        # update hit token when LM Cache is used, called in vllm/worker/model_runner.py
        # When CPU offload is used, the hit token is updated in model_runner.py
        # This count may be larger than the hit tokens recorded inside
        #   `update_hit_rates()`, because LMCache can reuse KV blocks from CPU.
        self.lmcache_hit_token += computed_token

    def update_hit_rates(self, request_id: str, computed_token: int,
                         prompt_token: int) -> None:
        """
            Update the prefix cache hit rate for a specific request.

            This function records the cache performance of a new request during its prefilling phase.
            Each request is processed only once for hit rate tracking to avoid double counting.
            Called in vllm/core/scheduler.py
            Args:
                request_id (str): Unique identifier assigned by vLLM for the request.
                computed_token (int): Number of tokens that hit the KV cache (cache hits).
                prompt_token (int): Total number of tokens in the prompt sequence.

        """

        if request_id not in self.request_id_2_hit_rate_map:
            # Compute the hit rate for the current request
            hit_rate = computed_token / prompt_token
            logger.debug(
                "[Simulator] Request ID: {}, Hit Tokens: {}, Total Tokens: {}, Hit Rate: {:.2f}%"
                .format(request_id, computed_token, prompt_token,
                        hit_rate * 100))
            # Update global counters for total hits and total tokens
            # Used to compute global hit rate = global_hit_tokens_num / global_tokens
            self.global_hit_tokens_num += computed_token
            self.global_tokens += prompt_token

            # Track the number of requests processed so far
            self.step_track += 1
            # Record mapping between request and its (hit, total) token pair
            self.request_id_2_hit_count_map[request_id] = (computed_token,
                                                           prompt_token)

            # Identify the workload type (e.g., ShareGPT, alfred, others)
            # for workload-aware statistics
            conv_key = self.request_id_2_conv_key_map[request_id]
            key_id = self.parser_conv_key(conv_key)[0]
            if key_id.startswith(
                ("ShareGPT-multi-turn", "text_ShareGPT-multi-turn")):
                active_queue = "text"
            elif key_id.startswith("image_ShareGPT-multi-turn"):
                active_queue = "image"
            elif key_id.startswith("search_ShareGPT-multi-turn"):
                active_queue = "search"
            elif key_id.startswith("file_ShareGPT-multi-turn"):
                active_queue = "file"
            elif key_id.startswith("alfred_"):
                active_queue = "agentic"
            else:
                active_queue = "single"
            # Update the hit rate for the active queue
            logger.debug(f"key_id {key_id} | active_queue {active_queue}")
            self.tracked_hit_stats[active_queue].append(computed_token)
            decay_factor = 0.95
            for q in self.tracked_hit_stats.keys():
                if q != active_queue and len(self.tracked_hit_stats[q]) > 0:
                    decayed = [
                        x * decay_factor for x in self.tracked_hit_stats[q]
                    ]
                    self.tracked_hit_stats[q].clear()
                    self.tracked_hit_stats[q].extend(decayed)

            # Store hit rate in per-request map
            self.request_id_2_hit_rate_map[request_id] = hit_rate
            session_id, turn = self.parser_conv_key(conv_key)
            logger.debug(
                "[Simulator] Updated hit rate: request_id=%s, session=%s, turn=%s, hit_rate=%.2f%%",
                request_id,
                session_id,
                turn,
                hit_rate * 100,
            )

    def update_evicted_block(self, block_id):
        # Evicted blocks can be reassigned new token data,
        # so reusing block_ids may result in incorrect token tracking.

        # Evicted block may be decoded token, will not be stored in block_id_2_count
        if block_id not in self.block_id_2_count:
            return
        self.archive_id += 1
        archive_key = self.archive_id
        self.block_id_2_count[archive_key] = self.block_id_2_count[block_id]
        self.block_id_2_token_ids[archive_key] = self.block_id_2_token_ids.get(
            block_id, [])
        del self.block_id_2_count[block_id]
        del self.block_id_2_token_ids[block_id]

    def update_hit_blocks(self, request_id,
                          common_computed_block_nums: list[int],
                          block_size: int, token_ids: list[int]):
        # This method should only be invoked when
        # the request is first processed. (prefill only)
        # Therefore, block for decoded tokens will not be tracked.
        if request_id in self.processed_requests_for_block:
            return
        # only track when the request is first processed (prefill only)

        self.processed_requests_for_block.add(request_id)

        for i, block_id in enumerate(common_computed_block_nums):
            token_ids_for_block = token_ids[i * block_size:(i + 1) *
                                            block_size]
            self.block_id_2_count[block_id] = self.block_id_2_count.get(
                block_id, 0) + 1
            if block_id in self.block_id_2_token_ids:
                old_tokens = self.block_id_2_token_ids[block_id]
                if old_tokens != token_ids_for_block:
                    raise ValueError(
                        f"Token IDs for block_id={block_id} changed! "
                        f"old={old_tokens[:10]}... new={token_ids_for_block[:10]}..."
                    )
            else:

                self.block_id_2_token_ids[block_id] = token_ids_for_block

    def parser_conv_key(self, conv_key: str) -> tuple[str, int]:
        # Expected format: id_<some_id>_turn_<turn>
        # Safe even if id contains "_"
        if "_turn_" not in conv_key:
            raise ValueError(f"Invalid conversation key format: {conv_key}")
        id_part, turn_str = conv_key.rsplit("_turn_", 1)
        id_str = id_part.removeprefix("id_")
        return id_str, int(turn_str)

    def simulate_outputs(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> List[SamplerOutput]:
        '''
        Use the predefined output token ids to simulate the outputs.
        '''
        completion_sequence_group_outputs = []
        for data in seq_group_metadata_list:
            request_id = data.request_id
            assert len(data.seq_data) == 1
            seq = next(iter(data.seq_data.values()))
            # already generated tokens
            output_token_ids = seq.output_token_ids
            # id_QWJhYvA_0_turn_0
            conv_key = self.request_id_2_conv_key_map.get(request_id)
            if conv_key is None:
                raise ValueError(
                    f"Could not find conversation key for request ID: "
                    f"{request_id}")
            id, turn = self.parser_conv_key(conv_key)

            stored_output_token_ids = self.session_data[id][
                "turn_" + str(turn)]["response_ids"]

            if len(output_token_ids) < len(stored_output_token_ids):
                new_token = stored_output_token_ids[len(output_token_ids)]
            elif len(output_token_ids) == len(stored_output_token_ids):
                new_token = self.eos_token_id
            else:
                new_token = self.eos_token_id
                raise IndexError(
                    f"Simulated token overflow for key={conv_key}, "
                    f"request_id={request_id}")
            output = CompletionSequenceGroupOutput(samples=[
                SequenceOutput(parent_seq_id=request_id,
                               output_token=new_token,
                               logprobs={
                                   new_token:
                                   Logprob(logprob=float('inf'),
                                           rank=None,
                                           decoded_token=None)
                               })
            ],
                                                   prompt_logprobs=None)
            completion_sequence_group_outputs.append(output)
        outputs = [
            SamplerOutput(outputs=completion_sequence_group_outputs,
                          sampled_token_probs=None,
                          sampled_token_ids=None,
                          spec_decode_worker_metrics=None)
        ]
        return outputs

    def save_metadata(self):
        start = time.perf_counter()
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        data = {
            "request_id_2_conv_key_map": self.request_id_2_conv_key_map,
            "request_id_2_hit_rate_map": self.request_id_2_hit_rate_map,
            "request_id_2_hit_count_map": self.request_id_2_hit_count_map,
        }
        if GLOBAL_SIMULATOR_CONFIG.enable_block_hit_track:
            data["block_id_2_count"] = self.block_id_2_count
            data["block_id_2_token_ids"] = self.block_id_2_token_ids
            data["reuse_time"] = self.reuse_time

        save_path = os.path.join(output_dir, "simulator_data.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        end = time.perf_counter()  # 结束计时
        print(
            f"[Simulator] Saved metadata to {save_path}, time taken = {end - start:.4f}s"
        )
        if GLOBAL_SIMULATOR_CONFIG.enable_lmcache:
            print(f"Simulator: lmcache hit token = {self.lmcache_hit_token }")
            print(
                f"Simulator: lmcache hit rate = {self.lmcache_hit_token / self.global_tokens}"
            )
        return

    def set_session_data(self, data):
        self.session_data = data
