# SPDX-License-Identifier: Apache-2.0
# Find the shared prefix of a task
import json
from collections import Counter

from tqdm import tqdm


def common_prefix(seq1, seq2):
    prefix = []
    for a, b in zip(seq1, seq2):
        if a == b:
            prefix.append(a)
        else:
            break
    return tuple(prefix)


def find_sublist(big, small):
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return i, i + len(small)
    return None


def get_middle_part(big_list, prefix_ids, suffix_ids):
    start_pos = find_sublist(big_list, prefix_ids)
    end_pos = find_sublist(big_list, suffix_ids)
    middle = big_list[start_pos[0]:end_pos[1]]
    return middle


def read_simulator_data(jsonl_path, max_records=None):
    print(f"[Simulator] read {max_records} records from {jsonl_path}")
    results = {}
    with open(jsonl_path, encoding='utf-8') as f:
        progress = tqdm(unit="entry", desc="Reading", total=max_records)

        for i, line in enumerate(f):
            entry = json.loads(line)
            results.update(entry)
            progress.update(1)

            if max_records is not None and i + 1 >= max_records:
                break

        progress.close()

    return results


jsonl_path = "/root/nfs/codespace/PrefixCacheRepo/PrefixCacheDev/vllm/simulator/token_ids.json"
with open(jsonl_path, encoding='utf-8') as f:
    config = json.load(f)
session_data = list(
    read_simulator_data("ToolBench_session_data.jsonl", 10000).values())
middle_counter = Counter()
for data in session_data:
    print("====================")
    prompt = data["turn_1"]["input_ids"]
    print("prompt len", len(prompt))
    systemp_promt = config["Llama-3.2-1B-Instruct"]["ToolBench"][
        "system prompt"]["token ids"]
    print("systemp_promt len", len(systemp_promt))
    common_system_prompt = common_prefix(prompt, systemp_promt)

    assert list(common_system_prompt) == systemp_promt
    middle_part = get_middle_part(
        prompt, config["Llama-3.2-1B-Instruct"]["ToolBench"]
        ["prefill shared part"]["prefix"], config["Llama-3.2-1B-Instruct"]
        ["ToolBench"]["prefill shared part"]["suffix"])
    print("middle part", len(middle_part)),

    middle_counter[tuple(middle_part)] += 1
    print("==left==\n",
          len(prompt) - len(common_system_prompt) - len(middle_part))
    left = prompt[len(common_system_prompt) + len(middle_part):]
print("Top 10 most common middle parts:")
for seq, count in middle_counter.most_common(10):
    print(f"Count={count}, Length={len(seq)}")
