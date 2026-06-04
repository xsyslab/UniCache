# SPDX-License-Identifier: Apache-2.0
import enum
import heapq
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from functools import lru_cache
from typing import Dict, List, Tuple, Union

import numpy as np

from vllm.logger import init_logger
from vllm.simulator.config import GLOBAL_SIMULATOR_CONFIG
from vllm.simulator.global_output_simulator import get_global_output_simulator
from vllm.simulator.hybrid_utils import BlockType, TaskType
from vllm.simulator.opt_utils import (GLOBAL_OFFLINE_OPT_HELPER,
                                      GLOBAL_ONLINE_OPT_HELPER)
from vllm.simulator.timer_utils import GLOBAL_TIMER
from vllm.simulator.wa_helper import GLOBAL_WA_HELPER

logger = init_logger(__name__)


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    FIFO = enum.auto()
    LFU = enum.auto()
    WA = enum.auto()
    SimpleHYBRID = enum.auto()
    OPT = enum.auto()
    OFFSET = enum.auto()
    LeCaR = enum.auto()
    ARC = enum.auto()
    NONE = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class OPTEvictor(Evictor):

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        if GLOBAL_SIMULATOR_CONFIG.eviction_policy == "OPT":
            self.OPTHelper = GLOBAL_OFFLINE_OPT_HELPER
        elif GLOBAL_SIMULATOR_CONFIG.eviction_policy == "OPT-Online":
            self.OPTHelper = GLOBAL_ONLINE_OPT_HELPER
        else:
            raise ValueError("Invalid eviction policy")

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _get_next_use(self, content_hash):
        return self.OPTHelper._get_next_use(content_hash)

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):

        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        nu = self._get_next_use(content_hash)
        # Key order: (1) -next-use, (2) last_accessed, (3) -num_hashed_tokens ...
        heapq.heappush(
            self.priority_queue,
            (-nu, last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def evict(self) -> Tuple[int, int]:

        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            neg_next_use, last_accessed, neg_token, block_id, content_hash = heapq.heappop(
                self.priority_queue)

            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash
        raise ValueError("No usable cache memory left")

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[int, float, int, int, int]] = []
        for block_id, block in self.free_table.items():
            nu = self._get_next_use(block.content_hash)
            new_priority_queue.append(
                (-nu, block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class OffsetEvictor(Evictor):
    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.tail_first = GLOBAL_SIMULATOR_CONFIG.tail_first
        self.head_first = GLOBAL_SIMULATOR_CONFIG.head_first
        assert (self.tail_first ^ self.head_first
                ), "Exactly one of tail_first/head_first must be True"

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            offset, last_accessed, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        if self.tail_first:
            heapq.heappush(
                self.priority_queue,
                (-num_hashed_tokens, last_accessed, block_id, content_hash))
        else:
            heapq.heappush(
                self.priority_queue,
                (num_hashed_tokens, last_accessed, block_id, content_hash))

        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > OffsetEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            if self.tail_first:
                new_priority_queue.append(
                    (-block.num_hashed_tokens, block.last_accessed, block_id,
                     block.content_hash))
            else:
                new_priority_queue.append(
                    (block.num_hashed_tokens, block.last_accessed, block_id,
                     block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """
    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)

        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):

        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class WABlockMetaData:

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, task_type: TaskType):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.task_type = task_type

    def _print(self):
        print(self.__repr__())

    def __repr__(self):
        return (f"[WABlockMetaData] "
                f"hash={self.content_hash}, "
                f"tokens={self.num_hashed_tokens}, "
                f"task_type={self.task_type.name}, "
                f"last_accessed={self.last_accessed:.2f}")

    def calculate_priority(self):
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            cur_time = GLOBAL_TIMER.get_current_time()
        else:
            cur_time = time.time()
        elapsed_time = cur_time - self.last_accessed
        priority = GLOBAL_WA_HELPER.get_reuse_probability(
            self.task_type, elapsed_time)
        return priority


class WAEvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int,
                              WABlockMetaData] = {}  # id -> WABlockMetaData
        self.priority_queue: Dict[str, Union[deque, List[Tuple]]] = {
            "single": [],  # Single-turn: PREFILL_SHARED
            "text": [],  # Multi-turn TEXT
            "file": [],  # Multi-turn FILE
            "image": [],  # Multi-turn Chat
            "search": [],  # Multi-turn SEARCH
            "agentic": [],  # Multi-turn Agentic
        }
        self.DECAY_QUEUES = [
            "single", "text", "file", "image", "search", "agentic"
        ]

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _select_queue_name(self, block: WABlockMetaData) -> str:

        if block.task_type == TaskType.MULTI_TURN_TEXT:
            return "text"

        if block.task_type == TaskType.MULTI_TURN_FILE:
            return "file"

        if block.task_type == TaskType.MULTI_TURN_IMAGE:
            return "image"

        if block.task_type == TaskType.MULTI_TURN_SEARCH:
            return "search"

        if block.task_type == TaskType.MULTI_TURN_AGENTIC:
            return "agentic"

        if block.task_type == TaskType.SINGLE_TURN:
            return "single"

        raise ValueError(
            f"Unknown task type: {block.task_type}, block type: {block.block_type}"
        )

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        # get candidate from queues
        queue_names = list(self.priority_queue.keys())
        candidates: Dict[str, int] = {}
        # get candidates
        for name in queue_names:
            q = self.priority_queue[name]
            while q:
                last_accessed, _, block_id, content_hash = heapq.heappop(q)
                block = self.free_table.get(block_id)
                if block and (block.last_accessed == last_accessed
                              and block.content_hash == content_hash):
                    candidates[name] = block_id
                    break

        if len(candidates) > 0:
            best_name, evicted_id, evicted_score, evicted_offset = None, None, float(
                "inf"), 0
            for name, block_id in candidates.items():
                block = self.free_table[block_id]
                score = block.calculate_priority()
                if score < evicted_score:
                    best_name, evicted_id, evicted_score, evicted_offset = name, block_id, score, block.num_hashed_tokens
                elif score == evicted_score:
                    # offset
                    if block.num_hashed_tokens > evicted_offset:
                        best_name, evicted_id, evicted_score, evicted_offset = name, block_id, score, block.num_hashed_tokens
            logger.info(
                f"[WA] evict name {best_name} evicted_score {evicted_score}")
            # Evict the lowest-priority candidate
            evicted_block = self.free_table.pop(evicted_id)
            # Put the rest back into their queues
            for name, block_id in candidates.items():
                if block_id == evicted_id:
                    continue
                block = self.free_table[block_id]
                heapq.heappush(self.priority_queue[name], (
                    block.last_accessed,
                    -block.num_hashed_tokens,
                    block_id,
                    block.content_hash,
                ))
            return evicted_id, evicted_block.content_hash
        raise ValueError("[CacheManager] No usable cache memory left!\n"
                         f"Free blocks: {len(self.free_table)}\n"
                         "")

        # comapre candidates
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, task_type: TaskType):
        self.free_table[block_id] = WABlockMetaData(content_hash,
                                                    num_hashed_tokens,
                                                    last_accessed, task_type)
        block = self.free_table[block_id]
        queue_name = self._select_queue_name(block)
        q = self.priority_queue[queue_name]
        heapq.heappush(
            q, (last_accessed, -num_hashed_tokens, block_id, content_hash))

    def _cleanup_if_necessary(self):
        total_len = sum(len(q) for q in self.priority_queue.values())
        if total_len > WAEvictor.CLEANUP_THRESHOLD * len(self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_queues = {
            "single": [],  # Single-turn: PREFILL_SHARED
            "text": [],  # Multi-turn TEXT
            "file": [],  # Multi-turn FILE
            "image": [],  # Multi-turn Chat
            "search": [],  # Multi-turn SEARCH
            "agentic": [],  # Multi-turn Agentic
        }

        for block_id, block in self.free_table.items():
            queue_name = self._select_queue_name(block)
            new_queues[queue_name].append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))

        for k, v in new_queues.items():
            heapq.heapify(v)
            self.priority_queue[k] = v

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def update(self):
        raise NotImplementedError


class LFUBlockMetaData:

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 access_count: int):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.access_count = access_count


class LFUEvictor(Evictor):
    """Evicts blocks with the lowest access count. Ties are broken using
    largest block size (num_hashed_tokens)."""

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, LFUBlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            access_count, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table
                    and self.free_table[block_id].access_count == access_count
                    and self.free_table[block_id].content_hash
                    == content_hash):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            access_count: int):
        self.free_table[block_id] = LFUBlockMetaData(content_hash,
                                                     num_hashed_tokens,
                                                     access_count)
        heapq.heappush(
            self.priority_queue,
            (access_count, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, access_count: int):
        self.free_table[block_id].access_count = access_count

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > self.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[int, int, int, int]] = []
        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.access_count, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class FIFOBlockMetaData:

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 first_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.first_accessed = first_accessed


class FIFOEvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, FIFOBlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:

        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        while self.priority_queue:
            first_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].first_accessed == first_accessed
                    and self.free_table[block_id].content_hash
                    == content_hash):
                self.free_table.pop(block_id)
                return block_id, content_hash
        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            first_accessed: float):
        self.free_table[block_id] = FIFOBlockMetaData(content_hash,
                                                      num_hashed_tokens,
                                                      first_accessed)
        heapq.heappush(
            self.priority_queue,
            (first_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > FIFOEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.first_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    def update(self, block_id: int, _: float):
        # FIFO 不需要更新 first_accessed，但为了接口一致性提供空实现
        pass

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class ARCBlockMetaData:
    """Metadata for a real cached block or a ghost entry.

    For ARC:
    - New idle block enters T1.
    - update() on a block in T1 promotes it to T2.
    - update() on a block in T2 refreshes it in T2.
    - B1/B2 store ghost entries (history only, not real cached data).
    """

    def __init__(self, content_hash: int, last_accessed: float,
                 access_count: int):

        self.content_hash = content_hash
        self.last_accessed = last_accessed
        self.access_count = access_count


class ARCHistoryMetaData:

    def __init__(self, content_hash: int):
        self.content_hash = content_hash


class ARCEvictor(Evictor):
    """Adaptive Replacement Cache (ARC) evictor.
     ARC maintains four LRU-ordered lists:
      - T1: recent entries in cache (seen once recently)
      - T2: frequent entries in cache (seen at least twice)
      - B1: ghost history of entries evicted from T1
      - B2: ghost history of entries evicted from T2
    Notes:
    - T1/T2 contain real evictable blocks.
    - B1/B2 are ghost entries only.
    """
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.cache_size = GLOBAL_SIMULATOR_CONFIG.simulator_num_gpu_blocks  # stale block
        self.p = 0  # target size for T1
        self.free_table: Dict[int, ARCBlockMetaData] = {}

        self.T1 = []  # hit only one time
        self.T2 = []  # hit more than one time
        self.B1: OrderedDict[int, ARCHistoryMetaData] = OrderedDict()
        self.B2: OrderedDict[int, ARCHistoryMetaData] = OrderedDict()
        self.miss_in_B2 = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def miss(self, content_hash: int):
        if content_hash in self.B1:
            self.p = min(self.p + max(1,
                                      len(self.B2) // len(self.B1)),
                         self.cache_size)
            self.B1.pop(content_hash)
            self.miss_in_B2 = False
        elif content_hash in self.B2:
            self.p = max(self.p - max(1, len(self.B1) // len(self.B2)), 0)
            self.B2.pop(content_hash)
            self.miss_in_B2 = True
        else:
            self.miss_in_B2 = False

    def evict(self):
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        len_L1 = len(self.T1) + len(self.B1)
        len_L2 = len(self.T2) + len(self.B2)
        T1_id = self._peek_lru_candidate(self.T1)
        T2_id = self._peek_lru_candidate(self.T2)
        evicted = None
        policy = None

        if T1_id is None and T2_id is None:
            raise ValueError("No usable cache memory left")

        if len_L1 >= self.cache_size:
            if len(self.T1) < self.cache_size:
                if self.B1:
                    self.B1.popitem(last=False)
                evicted, policy = self.replace()
            else:
                evicted = T1_id
                policy = "T1"
        elif len_L1 < self.cache_size and len_L1 + len_L2 >= self.cache_size:
            if len_L1 + len_L2 == 2 * self.cache_size:
                if self.B2:
                    self.B2.popitem(last=False)
            evicted, policy = self.replace()

        if evicted is None:
            if policy == "T1":
                evicted = T2_id
                policy = "T2"
            else:
                evicted = T1_id
                policy = "T1"

        victim = self.free_table.pop(evicted)
        if policy == "T1":
            heapq.heappop(self.T1)
            self.B1[victim.content_hash] = ARCHistoryMetaData(
                victim.content_hash)
        else:
            heapq.heappop(self.T2)
            self.B2[victim.content_hash] = ARCHistoryMetaData(
                victim.content_hash)

        return evicted, victim.content_hash

    def replace(self):
        x_in_B2 = self.miss_in_B2
        T1_id = self._peek_lru_candidate(self.T1)
        T2_id = self._peek_lru_candidate(self.T2)
        if T1_id is None and T2_id is None:
            raise ValueError("No usable cache memory left")
        if T1_id is None:
            return T2_id, "T2"
        if T2_id is None:
            return T1_id, "T1"

        if len(self.T1) > 0 and ((x_in_B2 and len(self.T1) == self.p)
                                 or len(self.T1) > self.p):
            evicted = T1_id
            return evicted, "T1"
        else:
            evicted = T2_id
            return evicted, "T2"

    def _peek_lru_candidate(self, queue):
        while queue:
            last_accessed, block_id, content_hash, access_count = queue[0]
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed
                    and self.free_table[block_id].access_count
                    == access_count):
                return block_id
            heapq.heappop(queue)

        return None

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, access_count: int):

        self.free_table[block_id] = ARCBlockMetaData(content_hash,
                                                     last_accessed,
                                                     access_count)
        if access_count == 1:
            # put T1
            heapq.heappush(self.T1,
                           (last_accessed, block_id, content_hash,
                            access_count))  #TODO: 修改 不用 num_hashed_tokens
        else:
            # put T2
            heapq.heappush(
                self.T2, (last_accessed, block_id, content_hash, access_count))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].access_count += 1

    def _cleanup_if_necessary(self):
        if len(self.T1) + len(
                self.T2) > ARCEvictor.CLEANUP_THRESHOLD * len(self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_T1: List[Tuple[float, int, int, int]] = []
        new_T2: List[Tuple[float, int, int, int]] = []
        for block_id, block in self.free_table.items():
            if block.access_count == 1:
                new_T1.append((block.last_accessed, block_id,
                               block.content_hash, block.access_count))
            else:
                new_T2.append((block.last_accessed, block_id,
                               block.content_hash, block.access_count))

        heapq.heapify(new_T1)
        heapq.heapify(new_T2)
        self.T1 = new_T1
        self.T2 = new_T2

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)


class HYBRIDBlockMetaData:

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 access_count: int, first_accessed: float,
                 last_accessed: float, block_type: BlockType,
                 task_type: TaskType):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.access_count = access_count
        self.first_accessed = first_accessed
        self.last_accessed = last_accessed
        self.block_type = block_type
        self.task_type = task_type

    def _print(self):
        print(f"[BlockMetaData] hash={self.content_hash}, "
              f"tokens={self.num_hashed_tokens}, "
              f"access_count={self.access_count}, "
              f"first_accessed={self.first_accessed:.4f}, "
              f"last_accessed={self.last_accessed:.4f}, "
              f"block_type={self.block_type}, "
              f"task_type={self.task_type}, ")

    def calculate_priority(self, max_offset=None, alpha_balance=None):
        # [simulator] use virtual time
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            cur_time = GLOBAL_TIMER.get_current_time()
        else:
            cur_time = time.time()
        if alpha_balance is None or not isinstance(alpha_balance, dict):
            raise ValueError("alpha_balance must be provided as a dict")
        if max_offset is None or max_offset == 0:
            max_offset = 1
        elapsed = cur_time - self.last_accessed
        # system
        if self.block_type == BlockType.PREFILL_SYSTEM:
            return 1
        if self.block_type in (BlockType.DECODE, BlockType.PREFILL_NOT_SHARED):
            return -1.0

        if self.task_type == TaskType.MULTI_TURN_TEXT:
            mu, sigma = 4.77, 1.279  #  4.15, 0.971 #
            alpha = alpha_balance.get("text", 1.0)
            reuse_prob = (1.0 - _cached_lognorm_cdf(int(elapsed), mu, sigma))
            return reuse_prob * alpha

        if self.task_type == TaskType.MULTI_TURN_IMAGE:
            mu, sigma = 4.4449, 1.09066
            alpha = alpha_balance.get("image", 1.0)
            reuse_prob = (1.0 - _cached_lognorm_cdf(int(elapsed), mu, sigma))
            return reuse_prob * alpha

        if self.task_type == TaskType.MULTI_TURN_SEARCH:
            mu, sigma = 4.961663, 1.13877
            alpha = alpha_balance.get("search", 1.0)
            reuse_prob = (1.0 - _cached_lognorm_cdf(int(elapsed), mu, sigma))
            return reuse_prob * alpha

        if self.task_type == TaskType.MULTI_TURN_FILE:
            mu, sigma = 5.07658, 1.06383
            alpha = alpha_balance.get("file", 1.0)
            reuse_prob = (1.0 - _cached_lognorm_cdf(int(elapsed), mu, sigma))
            return reuse_prob * alpha

        if self.task_type == TaskType.MULTI_TURN_AGENTIC:
            mu, sigma = 1.81, 1.092
            alpha = alpha_balance.get("agentic", 1.0)
            reuse_prob = 1.0 - _cached_lognorm_cdf(int(elapsed), mu, sigma)
            return reuse_prob * alpha

        if self.block_type == BlockType.PREFILL_SHARED:
            alpha = alpha_balance.get("single", 1.0)
            offset_factor = 1.0 - self.num_hashed_tokens / max_offset
            offset_factor = max(0.0, min(1.0, offset_factor))
            decay = math.exp(-elapsed * _INV_HALF_LIFE)
            return offset_factor * alpha * decay
        raise ValueError(f"Unknown block type: {self.block_type}")


class LeCaRBlockMetaData:

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, access_count: int):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.access_count = access_count


class LeCaRHistoryMetaData:

    def __init__(self, content_hash: int, evicted_time: float):
        self.content_hash = content_hash
        self.evicted_time = evicted_time


class LeCaREvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        np.random.seed(123)  # TODO:set seed
        self.cache_size = GLOBAL_SIMULATOR_CONFIG.simulator_num_gpu_blocks
        self.free_table: Dict[int, LeCaRBlockMetaData] = {}
        self.LRU = []
        self.LFU = []

        self.history_size = self.cache_size
        self.LRU_HIST: OrderedDict[int, LeCaRHistoryMetaData] = OrderedDict()
        self.LFU_HIST: OrderedDict[int, LeCaRHistoryMetaData] = OrderedDict()

        self.initial_weight = 0.5
        self.learning_rate = 0.45
        self.discount_rate = 0.005**(1 / self.cache_size)
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def getChoice(self):
        return 0 if np.random.rand() < self.W[0] else 1

    def miss(self, content_hash: int):
        # use content_hash as key for block
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            cur_time = GLOBAL_TIMER.get_current_time()
        else:
            cur_time = time.time()
        if content_hash in self.LRU_HIST:
            entry = self.LRU_HIST.pop(content_hash)
            reward_lru = -(self.discount_rate**(cur_time - entry.evicted_time))
            self.adjustWeights(reward_lru, 0)
        elif content_hash in self.LFU_HIST:
            entry = self.LFU_HIST.pop(content_hash)
            reward_lfu = -(self.discount_rate**(cur_time - entry.evicted_time))
            self.adjustWeights(0, reward_lfu)

    def adjustWeights(self, rewardLRU, rewardLFU):
        reward = np.array([rewardLRU, rewardLFU], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)
        if self.W[0] >= 0.99:
            self.W = np.array([0.99, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.99], dtype=np.float32)

    def addToHistory(self, victim: LeCaRBlockMetaData, evicted_time: float,
                     policy: int):
        # Add Entry to history dictated by policy
        # policy: 0, Add Entry to LRU History
        #         1, Add Entry to LFU History
        policy_history = None
        if policy == 0:
            policy_history = self.LRU_HIST
        elif policy == 1:
            policy_history = self.LFU_HIST
        elif policy == -1:
            return
        else:
            raise ValueError(f"Invalid policy value: {policy}")

        content_hash = victim.content_hash
        if content_hash in policy_history:
            del policy_history[content_hash]
        if len(policy_history) >= self.history_size:
            policy_history.popitem(last=False)
        policy_history[content_hash] = LeCaRHistoryMetaData(
            content_hash, evicted_time)

        # Evict from history is it is full
    def _peek_lru_candidate(self):
        while self.LRU:
            last_accessed, _, block_id, content_hash = self.LRU[0]
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                return block_id
            heapq.heappop(self.LRU)
        return None

    def _peek_lfu_candidate(self):
        while self.LFU:
            access_count, _, block_id, content_hash = self.LFU[0]
            if (block_id in self.free_table and
                    self.free_table[block_id].access_count == access_count):
                return block_id
            heapq.heappop(self.LFU)
        return None

    def evict(self) -> Tuple[int, int]:
        lru_id = self._peek_lru_candidate()
        lfu_id = self._peek_lfu_candidate()
        if lru_id is None and lfu_id is None:
            raise ValueError("No usable cache memory left")
        if lru_id is None:
            victim_id = lfu_id
            policy = -1
            heapq.heappop(self.LFU)
        elif lfu_id is None:
            victim_id = lru_id
            policy = -1
            heapq.heappop(self.LRU)
        elif lru_id == lfu_id:
            victim_id = lru_id
            policy = -1
            heapq.heappop(self.LRU)
            heapq.heappop(self.LFU)
        else:
            policy = self.getChoice()
            if policy == 0:
                victim_id = lru_id
                heapq.heappop(self.LRU)
            else:
                victim_id = lfu_id
                heapq.heappop(self.LFU)

        victim = self.free_table.pop(victim_id)
        cur_time = GLOBAL_TIMER.get_current_time(
        ) if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time else time.time(
        )
        self.addToHistory(victim, cur_time, policy)
        return victim_id, victim.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, access_count: int):
        self.free_table[block_id] = LeCaRBlockMetaData(content_hash,
                                                       num_hashed_tokens,
                                                       last_accessed,
                                                       access_count)
        heapq.heappush(self.LRU, (
            last_accessed,
            -num_hashed_tokens,
            block_id,
            content_hash,
        ))
        heapq.heappush(self.LFU, (
            access_count,
            -num_hashed_tokens,
            block_id,
            content_hash,
        ))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].access_count += 1

    def _cleanup_if_necessary(self):
        max_len = max(len(self.LRU), len(self.LFU))
        if max_len > LeCaREvictor.CLEANUP_THRESHOLD * len(self.free_table):
            self._cleanup()

    def _cleanup(self):
        lru_queue: List[Tuple[float, int, int, int]] = []
        lfu_queue: List[Tuple[int, int, int, int]] = []
        for block_id, block in self.free_table.items():
            heapq.heappush(lru_queue,
                           (block.last_accessed, -block.num_hashed_tokens,
                            block_id, block.content_hash))
            heapq.heappush(lfu_queue,
                           (block.access_count, -block.num_hashed_tokens,
                            block_id, block.content_hash))
        heapq.heapify(lru_queue)
        heapq.heapify(lfu_queue)
        self.LRU = lru_queue
        self.LFU = lfu_queue

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)


@lru_cache(maxsize=8)
def _cached_lognorm_params(mu: float, sigma: float):
    """Cache lognormal parameters to avoid recomputation."""
    return mu, sigma, sigma * math.sqrt(2.0)


@lru_cache(maxsize=8)
def _cached_lognorm_params_v1(mu: float, sigma: float):
    """Cache lognormal parameters to avoid recomputation."""
    inv_sigma = 1.0 / sigma
    denom = sigma * math.sqrt(2.0)
    norm_const = inv_sigma / math.sqrt(2.0 * math.pi)
    return mu, sigma, inv_sigma, denom, norm_const


def _cached_lognorm_prob(x: float, mu: float, sigma: float) -> float:
    """Compute CDF of lognormal distribution."""
    if x <= 0:
        return 0.0
    mu, sigma, inv_sigma, denom, _ = _cached_lognorm_params_v1(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


def _cached_lognorm_cdf(x: float, mu: float, sigma: float) -> float:
    """Compute CDF of lognormal distribution using fast erf-based approximation."""
    if x <= 0:
        return 0.0
    mu, sigma, denom = _cached_lognorm_params(mu, sigma)
    z = (math.log(x) - mu) / denom
    #  CDF
    return 0.5 * (1.0 + math.erf(z))


TASK_CONFIG = {
    TaskType.MULTI_TURN_TEXT: (4.77, 1.279, "text"),
    TaskType.MULTI_TURN_IMAGE: (4.4449, 1.09066, "image"),
    TaskType.MULTI_TURN_SEARCH: (4.961663, 1.13877, "search"),
    TaskType.MULTI_TURN_FILE: (5.07658, 1.06383, "file"),
    TaskType.MULTI_TURN_AGENTIC: (1.81, 1.092, "agentic"),
}
# === Cached helper functions ===
_INV_HALF_LIFE = 1.0 / 600
#  decay
_PRECOMPUTED_DECAY = [math.exp(-t * _INV_HALF_LIFE) for t in range(20000)]


class SimpleHYBRIDlockMetaData:

    def __init__(self,
                 content_hash: int,
                 num_hashed_tokens: int,
                 last_accessed: float,
                 access_count: int,
                 task_type: TaskType,
                 is_decode: bool,
                 next_use: int = None):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.task_type = task_type
        self.is_decode = is_decode
        self.access_count = access_count
        self.next_use = next_use

    def _print(self):
        print(self.__repr__())

    def __repr__(self):
        return f"SimpleHYBRIDlockMetaData(content_hash={self.content_hash}, num_hashed_tokens={self.num_hashed_tokens}, last_accessed={self.last_accessed}, task_type={self.task_type}, is_decode={self.is_decode})"

    def calculate_priority(self, max_offset=None, alpha_balance=None):
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            cur_time = GLOBAL_TIMER.get_current_time()
        else:
            cur_time = time.time()
        if alpha_balance is None or not isinstance(alpha_balance, dict):
            raise ValueError("alpha_balance must be provided as a dict")
        if max_offset is None or max_offset == 0:
            max_offset = 1
        elapsed = cur_time - self.last_accessed
        e = int(elapsed)
        get_cdf = _cached_lognorm_cdf
        task_cfg = TASK_CONFIG.get(self.task_type)

        if task_cfg is not None:
            mu, sigma, alpha_key = task_cfg
            alpha = alpha_balance.get(alpha_key, 1.0)
            reuse_prob = 1.0 - get_cdf(e, mu, sigma)
            if os.getenv("SimpleHYBRID_Disable_HIT_EFFICIENCY", "0") == "0":
                return alpha * reuse_prob
            else:
                return reuse_prob
        if self.task_type == TaskType.SINGLE_TURN:
            alpha = alpha_balance.get("single", 1.0)
            offset_factor = 1.0 - self.num_hashed_tokens / (max_offset or 1)
            offset_factor = max(0.0, min(1.0, offset_factor))
            if os.getenv("SimpleHYBRID_Disable_Single_Time_Decay", "0") == "0":
                d = _PRECOMPUTED_DECAY[e] if e < len(
                    _PRECOMPUTED_DECAY) else math.exp(-elapsed *
                                                      _INV_HALF_LIFE)
                return offset_factor * alpha * d
            else:
                if os.getenv("SimpleHYBRID_Disable_HIT_EFFICIENCY",
                             "0") == "0":
                    return offset_factor * alpha
                else:
                    return offset_factor
        return None


class SimpleHYBRIDEvictor(Evictor):
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, SimpleHYBRIDlockMetaData] = {}
        self.priority_queue: Dict[str, Union[deque, List[Tuple]]] = {
            "evict_first": [],
            "single": [],
            "text": [],
            "file": [],
            "image": [],
            "search": [],
            "agentic": [],
        }
        self.DECAY_QUEUES = [
            "single", "text", "file", "image", "search", "agentic"
        ]
        self.MULTI_TURN_QUEUES = ["text", "file", "image", "search", "agentic"]
        self.SINGLE_TURN_QUEUES = ["single"]
        self.alpha_single = 1.0
        self.alpha_text = 1.0
        self.alpha_file = 1.0
        self.alpha_image = 1.0
        self.alpha_search = 1.0
        self.alpha_agentic = 1.0
        self.alpha_balance = {
            "single": self.alpha_single,
            "text": self.alpha_text,
            "file": self.alpha_file,
            "image": self.alpha_image,
            "search": self.alpha_search,
            "agentic": self.alpha_agentic,
        }
        self.max_num_first_offset = 2048
        self.output_simulator = get_global_output_simulator()
        self._last_alpha_update_step = -1
        if GLOBAL_SIMULATOR_CONFIG.eviction_policy == "SimpleHYBRID-OPT":
            self.OPTHelper = GLOBAL_ONLINE_OPT_HELPER
            self.enable_hybrid_opt = True
        else:
            self.enable_hybrid_opt = False

    def _get_next_use(self, content_hash):
        return self.OPTHelper._get_next_use(content_hash)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _select_queue_name(self, block: SimpleHYBRIDlockMetaData) -> str:
        if block.task_type == TaskType.MULTI_TURN_TEXT:
            return "text"

        if block.task_type == TaskType.MULTI_TURN_FILE:
            return "file"

        if block.task_type == TaskType.MULTI_TURN_IMAGE:
            return "image"

        if block.task_type == TaskType.MULTI_TURN_SEARCH:
            return "search"

        if block.task_type == TaskType.MULTI_TURN_AGENTIC:
            return "agentic"

        if block.task_type == TaskType.SINGLE_TURN:
            if block.is_decode:
                return "evict_first"
            else:
                return "single"

        if block.task_type == TaskType.SINGLE_TURN_RANDOM:
            return "evict_first"
        raise ValueError(f"Unknown task type: {block.task_type}")

    def evict(self) -> Tuple[int, int]:
        if self.enable_hybrid_opt:
            return self._opt_evict()
        else:
            return self._naive_evict()

    def _opt_evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        # === Step 1. evict from evict_first ===
        q = self.priority_queue["evict_first"]
        while q:
            neg_next_use, neg_tokens, last_accessed, block_id, content_hash = heapq.heappop(
                q)
            if block_id not in self.free_table:
                continue
            block = self.free_table.get(block_id)
            if (block.content_hash == content_hash
                    and block.last_accessed == last_accessed):
                self.free_table.pop(block_id)
                logger.debug("Evicted block %d from evict_first", block_id)
                return block_id, content_hash
        # === Step 2. evict from decay queues ===
        candidates: Dict[str, int] = {}
        for name in self.DECAY_QUEUES:
            q = self.priority_queue[name]
            # find one candidate for each queue
            while q:
                if name in self.MULTI_TURN_QUEUES:
                    neg_next_use, last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(
                        q)
                elif name in self.SINGLE_TURN_QUEUES:
                    neg_next_use, neg_tokens, access_count, last_accessed, block_id, content_hash = heapq.heappop(
                        q)
                else:
                    raise ValueError(f"Unknown queue name: {name}")
                block = self.free_table.get(block_id)
                if block and (block.last_accessed == last_accessed
                              and block.content_hash == content_hash):
                    candidates[name] = block_id
                    break

        if len(candidates) > 0:
            step = self.output_simulator.step_track
            if step % 10 == 0 and step != self._last_alpha_update_step:
                self.update_alpha_balance()
                self._last_alpha_update_step = step
            # first compare neg_next_use
            best_name, evicted_id, next_use, evicted_score, evicted_offset = None, None, -float(
                "inf"), float("inf"), 0
            if len(candidates) == 1:
                evicted_id = next(iter(candidates.values()))
                evicted_block = self.free_table.pop(evicted_id)
                return evicted_id, evicted_block.content_hash
            else:
                for name, block_id in candidates.items():
                    block = self.free_table[block_id]
                    score = block.calculate_priority(self.max_num_first_offset,
                                                     self.alpha_balance)
                    # 1. Prefer block with larger next_use
                    if block.next_use > next_use:
                        best_name, evicted_id, next_use, evicted_score, evicted_offset = name, block_id, block.next_use, score, block.num_hashed_tokens
                    elif block.next_use == next_use:
                        if score < evicted_score:
                            best_name, evicted_id, next_use, evicted_score, evicted_offset = name, block_id, block.next_use, score, block.num_hashed_tokens
                        elif score == evicted_score:
                            if block.num_hashed_tokens > evicted_offset:
                                best_name, evicted_id, next_use, evicted_score, evicted_offset = name, block_id, block.next_use, score, block.num_hashed_tokens
                # Evict the lowest-priority candidate
                evicted_block = self.free_table.pop(evicted_id)
                # Put the rest back into their queues
                for name, block_id in candidates.items():
                    if block_id == evicted_id:
                        continue
                    block = self.free_table[block_id]
                    if name in self.MULTI_TURN_QUEUES:
                        heapq.heappush(self.priority_queue[name], (
                            -block.next_use,
                            block.last_accessed,
                            -block.num_hashed_tokens,
                            block_id,
                            block.content_hash,
                        ))
                    elif name in self.SINGLE_TURN_QUEUES:
                        heapq.heappush(
                            self.priority_queue[name],
                            (
                                -block.next_use,
                                -block.num_hashed_tokens,
                                block.access_count,
                                block.last_accessed,
                                block_id,
                                block.content_hash,
                            ),
                        )
                    else:
                        raise ValueError(f"Unknown queue name: {name}")
            logger.debug(f"Evict block {evicted_id} from queue {best_name}")
            return evicted_id, evicted_block.content_hash

    def _naive_evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        # === Step 1. evict from evict_first ===
        q = self.priority_queue["evict_first"]
        while q:
            neg_tokens, last_accessed, block_id, content_hash = heapq.heappop(
                q)
            if block_id not in self.free_table:
                continue
            block = self.free_table.get(block_id)
            if (block.content_hash == content_hash
                    and block.last_accessed == last_accessed):
                self.free_table.pop(block_id)
                logger.debug("Evicted block %d from evict_first", block_id)
                return block_id, content_hash

        # === Step 2. evict from decay queues ===
        candidates: Dict[str, int] = {}
        for name in self.DECAY_QUEUES:
            q = self.priority_queue[name]
            # find one candidate for each queue
            while q:
                if name in self.MULTI_TURN_QUEUES:
                    last_accessed, neg_tokens, block_id, content_hash = heapq.heappop(
                        q)
                elif name in self.SINGLE_TURN_QUEUES:
                    neg_tokens, access_count, last_accessed, block_id, content_hash = heapq.heappop(
                        q)
                else:
                    raise ValueError(f"Unknown queue name: {name}")
                block = self.free_table.get(block_id)
                if block and (block.last_accessed == last_accessed
                              and block.content_hash == content_hash):
                    candidates[name] = block_id
                    break
        if len(candidates) > 0:
            step = self.output_simulator.step_track
            if step % 10 == 0 and step != self._last_alpha_update_step:
                self.update_alpha_balance()
                self._last_alpha_update_step = step

            best_name, evicted_id, evicted_score, evicted_offset = None, None, float(
                "inf"), 0
            if len(candidates) == 1:
                evicted_id = next(iter(candidates.values()))
                evicted_block = self.free_table.pop(evicted_id)
                return evicted_id, evicted_block.content_hash
            else:
                for name, block_id in candidates.items():
                    block = self.free_table[block_id]
                    score = block.calculate_priority(self.max_num_first_offset,
                                                     self.alpha_balance)
                    if score < evicted_score:
                        best_name, evicted_id, evicted_score, evicted_offset = name, block_id, score, block.num_hashed_tokens
                    elif abs(score - evicted_score) <= 1e-9:
                        if block.num_hashed_tokens > evicted_offset:
                            best_name, evicted_id, evicted_score, evicted_offset = name, block_id, score, block.num_hashed_tokens
                # Evict the lowest-priority candidate
                evicted_block = self.free_table.pop(evicted_id)
                # Put the rest back into their queues
                for name, block_id in candidates.items():
                    if block_id == evicted_id:
                        continue

                    block = self.free_table[block_id]
                    if name in self.MULTI_TURN_QUEUES:
                        heapq.heappush(self.priority_queue[name], (
                            block.last_accessed,
                            -block.num_hashed_tokens,
                            block_id,
                            block.content_hash,
                        ))
                    elif name in self.SINGLE_TURN_QUEUES:
                        heapq.heappush(
                            self.priority_queue[name],
                            (
                                -block.num_hashed_tokens,
                                block.access_count,
                                block.last_accessed,
                                block_id,
                                block.content_hash,
                            ),
                        )
                    else:
                        raise ValueError(f"Unknown queue name: {name}")
            logger.debug(f"Evict block {evicted_id} from queue {best_name}")
            return evicted_id, evicted_block.content_hash
        raise ValueError(
            "[CacheManager] No usable cache memory left!\n"
            f"Free blocks: {len(self.free_table)}\n"
            f"Evict-first queue: {len(self.priority_queue['evict_first'])}\n"
            f"Single-turn queue: {len(self.priority_queue['single'])}\n"
            f"Text queue: {len(self.priority_queue['text'])}\n"
            f"File queue: {len(self.priority_queue['file'])}\n"
            f"Image queue: {len(self.priority_queue['image'])}\n"
            f"Search queue: {len(self.priority_queue['search'])}\n"
            f"Agentic queue: {len(self.priority_queue['agentic'])}\n")

    def _naive_add(self, block_id: int, content_hash: int,
                   num_hashed_tokens: int, last_accessed: float,
                   access_count: int, task_type: TaskType, is_decode: bool):

        self.free_table[block_id] = SimpleHYBRIDlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, access_count,
            task_type, is_decode)
        block = self.free_table[block_id]
        queue_name = self._select_queue_name(block)
        logger.debug(
            f"Evictor add block_id { block_id},  task_type {task_type},  queue_name {queue_name}"
        )
        q = self.priority_queue[queue_name]
        if queue_name == "evict_first":
            heapq.heappush(
                q, (-num_hashed_tokens, last_accessed, block_id, content_hash))
        elif queue_name in self.MULTI_TURN_QUEUES:
            heapq.heappush(q, (
                last_accessed,
                -num_hashed_tokens,
                block_id,
                content_hash,
            ))
        elif queue_name in self.SINGLE_TURN_QUEUES:
            heapq.heappush(q, (
                -num_hashed_tokens,
                access_count,
                last_accessed,
                block_id,
                content_hash,
            ))
        else:
            raise NotImplementedError(f"Unknown queue: {queue_name}")

    def _opt_add(self, block_id: int, content_hash: int,
                 num_hashed_tokens: int, last_accessed: float,
                 access_count: int, task_type: TaskType, is_decode: bool):
        next_use = self._get_next_use(content_hash)
        self.free_table[block_id] = SimpleHYBRIDlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, access_count,
            task_type, is_decode, next_use)
        block = self.free_table[block_id]
        queue_name = self._select_queue_name(block)
        logger.debug(
            f"Evictor add block_id { block_id},  task_type {task_type},  queue_name {queue_name}"
        )
        q = self.priority_queue[queue_name]
        if queue_name == "evict_first":
            heapq.heappush(q, (-next_use, -num_hashed_tokens, last_accessed,
                               block_id, content_hash))
        elif queue_name in self.MULTI_TURN_QUEUES:
            heapq.heappush(q, (
                -next_use,
                last_accessed,
                -num_hashed_tokens,
                block_id,
                content_hash,
            ))
        elif queue_name in self.SINGLE_TURN_QUEUES:
            heapq.heappush(q, (
                -next_use,
                -num_hashed_tokens,
                access_count,
                last_accessed,
                block_id,
                content_hash,
            ))
        else:
            raise NotImplementedError(f"Unknown queue: {queue_name}")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, access_count: int, task_type: TaskType,
            is_decode: bool):

        if self.enable_hybrid_opt:
            self._opt_add(block_id, content_hash, num_hashed_tokens,
                          last_accessed, access_count, task_type, is_decode)
        else:
            self._naive_add(block_id, content_hash, num_hashed_tokens,
                            last_accessed, access_count, task_type, is_decode)

    def update(self, block_id: int, last_accessed: float, task_type: TaskType,
               is_decode: bool):
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].access_count += 1
        self.free_table[block_id].task_type = task_type
        self.free_table[block_id].is_decode = is_decode

    def _cleanup_if_necessary(self):
        total_len = sum(len(q) for q in self.priority_queue.values())
        if total_len > SimpleHYBRIDEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_queues = {
            "evict_first":
            [],  # Single-turn: DECODE / PREFILL_NOT_SHARED sorted by offset
            "single": [],  # Single-turn: PREFILL_SHARED
            "text": [],  # Multi-turn TEXT
            "file": [],  # Multi-turn FILE
            "image": [],  # Multi-turn Chat
            "search": [],  # Multi-turn SEARCH
            "agentic": [],  # Multi-turn Agentic
        }
        if self.enable_hybrid_opt:
            for block_id, block in self.free_table.items():
                queue_name = self._select_queue_name(block)
                if queue_name == "evict_first":
                    new_queues[queue_name].append(
                        (-block.next_use, -block.num_hashed_tokens,
                         block.last_accessed, block_id, block.content_hash))

                elif queue_name in self.MULTI_TURN_QUEUES:
                    # LRU
                    new_queues[queue_name].append((
                        -block.next_use,
                        block.last_accessed,
                        -block.num_hashed_tokens,
                        block_id,
                        block.content_hash,
                    ))
                elif queue_name in self.SINGLE_TURN_QUEUES:
                    new_queues[queue_name].append((
                        -block.next_use,
                        -block.num_hashed_tokens,
                        block.access_count,
                        block.last_accessed,
                        block_id,
                        block.content_hash,
                    ))
                else:
                    raise NotImplementedError(f"Unknown queue: {queue_name}")
        else:
            for block_id, block in self.free_table.items():
                queue_name = self._select_queue_name(block)
                if queue_name == "evict_first":
                    new_queues[queue_name].append(
                        (-block.num_hashed_tokens, block.last_accessed,
                         block_id, block.content_hash))

                elif queue_name in self.MULTI_TURN_QUEUES:
                    # LRU
                    new_queues[queue_name].append((
                        block.last_accessed,
                        -block.num_hashed_tokens,
                        block_id,
                        block.content_hash,
                    ))
                elif queue_name in self.SINGLE_TURN_QUEUES:
                    new_queues[queue_name].append((
                        -block.num_hashed_tokens,
                        block.access_count,
                        block.last_accessed,
                        block_id,
                        block.content_hash,
                    ))
                else:
                    raise NotImplementedError(f"Unknown queue: {queue_name}")

        self.priority_queue.clear()
        self.priority_queue.update(new_queues)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def update_alpha_balance(self, smooth=0.1, eps=1e-3, T=0.2):
        """
        Dynamically update the priority weights (α) for different workload types
        based on online hit-token statistics.
        Args:
            smooth (float): Smoothing factor for exponential moving average. 
                            Closer to 1 → slower updates, more stability.
            eps (float): Small constant to avoid division by zero.
            T (float): Temperature factor controlling the sensitivity of balance 
                    adjustment. Smaller T → more sensitive changes.
        """
        # === 1️⃣ Check simulator attributes ===
        if not hasattr(self,
                       "output_simulator") or self.output_simulator is None:
            raise ValueError(
                "HYBRIDEvictor: output_simulator not set, cannot update α.")
        if not hasattr(self.output_simulator, "tracked_hit_stats"):
            raise ValueError("output_simulator missing tracked_hit_stats.")
        # === 2️⃣ Collect hit-token statistics ===
        # hit_stats: {queue_name: [hit_tokens, ...]}
        hit_stats = self.output_simulator.tracked_hit_stats
        total_hit = {q: float(np.sum(hits)) for q, hits in hit_stats.items()}
        total_all = sum(total_hit.values()) + eps
        # === 3️⃣ Compute cache occupancy ratio ===
        total_cache = len(self.free_table) + eps
        capacity_ratio = {
            q: len(self.priority_queue.get(q, [])) / total_cache
            for q in total_hit
        }
        hit_efficiency = {}
        valid_efficiency = {}  # ✅ 仅保留参与归一化的有效项
        for q in total_hit:
            hits = total_hit[q]
            cap = capacity_ratio.get(q, 0.0)
            if hits > 0 and cap > 0:
                eff = (hits / total_all) / (cap + eps)
                hit_efficiency[q] = eff
                valid_efficiency[q] = eff
            else:
                # 对无效队列保留当前 α，不纳入统计
                hit_efficiency[q] = 0.0

        # === 5️⃣ Normalize and temperature-scale ===
        if len(valid_efficiency) == 0:
            logger.warning(
                "[AlphaBalance] All queues have zero efficiency; skip update.")
            return

        eff_values = np.array(list(valid_efficiency.values()))
        mean_eff = np.mean(eff_values) + eps
        scaled = (eff_values / mean_eff)**(1 / T)

        # === 6️⃣ Compute adaptive clipping bounds ===
        mean_val = np.mean(scaled)
        std_val = np.std(scaled)
        lower = max(0.001, mean_val - 2 * std_val)
        upper = min(10.0, mean_val + 2 * std_val)

        # === 7️⃣ Smoothly update α (EMA-based) ===
        new_alpha = {}
        valid_keys = list(valid_efficiency.keys())
        for i, q in enumerate(valid_keys):
            prev = getattr(self, f"alpha_{q}", 1.0)
            val = scaled[i]
            updated = smooth * prev + (1 - smooth) * val
            updated = float(np.clip(updated, lower, upper))
            setattr(self, f"alpha_{q}", updated)
            new_alpha[q] = updated

        for q in total_hit:
            if q not in valid_efficiency:
                new_alpha[q] = getattr(self, f"alpha_{q}", 0)

        # === 8️⃣ Occasionally log the updates ===
        self.alpha_balance = new_alpha
        if random.random() < 0.1:
            msg = "[AlphaBalance] Updated (hit_token-based): " + ", ".join(
                f"{k}={v:.3f}" for k, v in new_alpha.items())
            msg += " | hit_eff=" + ", ".join(f"{k}={hit_efficiency[k]:.3f}"
                                             for k in hit_efficiency)
            msg += " | hit_token=" + ", ".join(f"{k}={total_hit[k]:.0f}"
                                               for k in total_hit)
            logger.info(msg)

    # def update_alpha_balance(self, smooth=0.9, eps=1e-3, T=0.2):

    #     """
    #     Dynamically update the priority weights (α) for different workload types
    #     based on online hit-token statistics.
    #     Args:
    #         smooth (float): Smoothing factor for exponential moving average.
    #                         Closer to 1 → slower updates, more stability.
    #         eps (float): Small constant to avoid division by zero.
    #         T (float): Temperature factor controlling the sensitivity of balance
    #                 adjustment. Smaller T → more sensitive changes.
    #     """
    #     # === 1️⃣ Check simulator attributes ===
    #     if not hasattr(self, "output_simulator") or self.output_simulator is None:
    #         raise ValueError("HYBRIDEvictor: output_simulator not set, cannot update α.")
    #     if not hasattr(self.output_simulator, "tracked_hit_stats"):
    #         raise ValueError("output_simulator missing tracked_hit_stats.")
    #     # === 2️⃣ Collect hit-token statistics ===
    #     # hit_stats: {queue_name: [hit_tokens, ...]}
    #     hit_stats = self.output_simulator.tracked_hit_stats
    #     free_cnt = len(self.free_table) + eps
    #     pqueue = self.priority_queue
    #     alpha_bal = self.alpha_balance

    #     total_hit = {}
    #     total_all = eps

    #     # --- FIRST LOOP: compute total_hit & capacity ratio ---
    #     capacity_ratio = {}
    #     for q, hits in hit_stats.items():
    #         h = float(sum(hits))
    #         total_hit[q] = h
    #         total_all += h
    #         capacity_ratio[q] = len(pqueue.get(q, ())) / free_cnt
    #     # --- SECOND LOOP: compute hit efficiency ---
    #     valid_keys = []
    #     eff_vals = []   # store only values used in normalization
    #     hit_eff = {}
    #     for q, h in total_hit.items():
    #         cap = capacity_ratio[q]
    #     if h > 0 and cap > 0:
    #         eff = (h / total_all) / (cap + eps)
    #         hit_eff[q] = eff
    #         valid_keys.append(q)
    #         eff_vals.append(eff)
    #     else:
    #         hit_eff[q] = 0.0

    #     if not eff_vals:
    #         return

    #     # --- Normalize + temperature ---
    #     n = len(eff_vals)
    #     mean_eff = sum(eff_vals) / n + eps

    #     scaled = [(v / mean_eff) ** (1 / T) for v in eff_vals]

    #     # --- clipping bounds ---
    #     mean_s = sum(scaled) / n
    #     std_s = (sum((x - mean_s) ** 2 for x in scaled) / n) ** 0.5

    #     lower = max(0.001, mean_s - 2 * std_s)
    #     upper = min(10.0,  mean_s + 2 * std_s)

    #     # --- Update α (EMA) ---
    #     new_alpha = alpha_bal.copy()
    #     for q, val in zip(valid_keys, scaled):
    #         prev = alpha_bal.get(q, 1.0)
    #         x = smooth * prev + (1 - smooth) * val
    #         new_alpha[q] = max(lower, min(upper, x))

    #     # keep others unchanged
    #     self.alpha_balance = new_alpha
    #     if random.random() < 0.1:
    #         logger.info(
    #         "[AlphaBalance] " +
    #         ", ".join(f"{k}={new_alpha[k]:.3f}" for k in new_alpha) +
    #         " | eff=" + ", ".join(f"{k}={hit_eff[k]:.3f}" for k in hit_eff)
    #     )


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    elif eviction_policy == EvictionPolicy.LFU:
        return LFUEvictor()
    elif eviction_policy == EvictionPolicy.WA:
        return WAEvictor()
    elif eviction_policy == EvictionPolicy.SimpleHYBRID:
        return SimpleHYBRIDEvictor()
    elif eviction_policy == EvictionPolicy.OPT:
        return OPTEvictor()
    elif eviction_policy == EvictionPolicy.OFFSET:
        return OffsetEvictor()
    elif eviction_policy == EvictionPolicy.LeCaR:
        return LeCaREvictor()
    elif eviction_policy == EvictionPolicy.ARC:
        return ARCEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
