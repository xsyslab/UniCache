# SPDX-License-Identifier: Apache-2.0
import json

from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, SequenceStatus

logger = init_logger(__name__)

PREFILL_CNT = 0


def parser_conv_key(conv_key: str) -> tuple[str, int]:
    # Expected format: id_<some_id>_turn_<turn>
    # Safe even if id contains "_"
    if "_turn_" not in conv_key:
        raise ValueError(f"Invalid conversation key format: {conv_key}")
    id_part, turn_str = conv_key.rsplit("_turn_", 1)
    id_str = id_part.removeprefix("id_")
    return id_str, int(turn_str)


def get_all_token_hash(token_ids, block_size=8):
    _none_hash: int = hash('None')
    full_block_num = len(token_ids) // block_size
    all_hash = []
    prev_block_hash = None
    for i in range(full_block_num):
        tokens = token_ids[i * block_size:(i + 1) * block_size]
        is_first_block = (i == 0)
        if i == 0 and prev_block_hash is None:
            prev_block_hash = _none_hash
        cur_hash = hash((is_first_block, prev_block_hash, *tokens, None))
        prev_block_hash = cur_hash
        all_hash.append(cur_hash)
    return all_hash


def create_block_future_use_map(trace_path, real_data, block_size, save_path):
    request_content_hash = {}
    reuse_time_map_hash_2_iter = {}
    with open(trace_path) as f:
        trace_data = json.load(f)
    for k, v in real_data.items():
        num_turns = v["num_turns"]
        for t in range(1, num_turns + 1):
            d = v[f"turn_{t}"]
            all_token = d["input_ids"] + (d["response_ids"][:-1])
            all_hash = get_all_token_hash(all_token, block_size)
            request_id = f"id_{k}_turn_{t}"
            request_content_hash[request_id] = {}
            request_content_hash[request_id]["all_block_hash"] = all_hash
            request_content_hash[request_id]["full_block_num"] = len(all_hash)

    for iter_cnt, req in trace_data.items():
        item = req
        seen_hashes_this_iter = set()
        for k, v in item.items():
            id_str, turn_str = parser_conv_key(k)
            input_token = real_data[id_str][f"turn_{turn_str}"]["input_ids"]
            if v["is_prompt"]:
                prefill_block = len(input_token) // block_size
                prefill_block_hash = request_content_hash[k][
                    "all_block_hash"][:prefill_block]
                for block_hash in prefill_block_hash:
                    if block_hash in seen_hashes_this_iter:
                        continue
                    seen_hashes_this_iter.add(block_hash)
                    if block_hash not in reuse_time_map_hash_2_iter:
                        reuse_time_map_hash_2_iter[block_hash] = []
                    reuse_time_map_hash_2_iter[block_hash].append(
                        int(iter_cnt))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(reuse_time_map_hash_2_iter, f, indent=2)


class OfflineOPTHelper:

    def __init__(self):
        self.reuse_track = None

    def set_reuse_track(self, map_path: str):
        with open(map_path) as f:
            self.reuse_track = json.load(f)

    def _get_next_use(self, content_hash):
        times = self.reuse_track.get(str(content_hash), [])
        global PREFILL_CNT
        while times and times[0] <= PREFILL_CNT:
            times.pop(0)
        self.reuse_track[str(content_hash)] = times
        if not times:
            return float("inf")
        return times[0]


GLOBAL_OFFLINE_OPT_HELPER = OfflineOPTHelper()


class OnlineOPTHelper:

    def __init__(self, block_size=8):
        self.req_counter = 0
        # map block hash to use  req-counter-index list
        self.reuse_track = {}
        # used to calculate content hash
        self.block_size = block_size
        self.touch_req_counter = 0

    def _get_next_use(self, content_hash):
        times = self.reuse_track.get(content_hash, [])
        global PREFILL_CNT

        while times and times[0] <= PREFILL_CNT:
            times.pop(0)
        self.reuse_track[str(content_hash)] = times

        if not times:
            return float("inf")

        return times[0]

    def _add_processed_request(self, seq_group: SequenceGroup):
        seq_data: Dict[int, SequenceData] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq_id = seq.seq_id
            seq_data[seq_id] = seq.data
        seq_data_0 = next(iter(seq_data.values()))
        prompt_token_ids = seq_data_0.prompt_token_ids
        all_hash = get_all_token_hash(prompt_token_ids, self.block_size)
        for h in all_hash:
            if h not in self.reuse_track:
                self.reuse_track[h] = []
            self.reuse_track[h].append(self.req_counter)
        self.req_counter += 1


GLOBAL_ONLINE_OPT_HELPER = OnlineOPTHelper()
