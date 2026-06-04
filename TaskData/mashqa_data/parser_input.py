# SPDX-License-Identifier: Apache-2.0
# Find the shared prefix of a task
import json
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer


def find_sublist(big, small):
    """
    通用子序列查找：支持 big 是 tuple，small 是 list 或 tuple。
    返回 (start_idx, end_idx)，找不到则返回 None。
    """
    n, m = len(big), len(small)
    if m == 0 or n == 0 or m > n:
        return None

    # tuple 与 list 混合比较时，统一逐段转 tuple（避免全局 list()）
    small_tuple = tuple(small)
    for i in range(n - m + 1):
        if big[i:i + m] == small_tuple:
            return (i, i + m)
    return None


def get_middle_part(big_list, prefix_ids, suffix_ids):
    """
    从 big_list (tuple) 中截取 prefix_ids 与 suffix_ids 之间的部分。
    自动兼容 tuple + list 混用。
    """
    start_pos = find_sublist(big_list, prefix_ids)
    end_pos = find_sublist(big_list, suffix_ids)

    if start_pos is None or end_pos is None:
        print(
            f"[WARN] prefix/suffix not found: start={start_pos}, end={end_pos}"
        )
        return []

    # tuple 切片本身高效（不会复制所有元素）
    return list(big_list[start_pos[1]:end_pos[0]])


def common_prefix_length(seq1, seq2):
    length = 0
    for a, b in zip(seq1, seq2):
        if a == b:
            length += 1
        else:
            break
    return length


def common_prefix(seq1, seq2):
    prefix = []
    for a, b in zip(seq1, seq2):
        if a == b:
            prefix.append(a)
        else:
            break
    return tuple(prefix)


# def find_sublist(big, small):
#     for i in range(len(big) - len(small) + 1):
#         if big[i:i + len(small)] == small:
#             return i, i + len(small)
#     return None

# def get_middle_part(big_list, prefix_ids, suffix_ids):
#     start_pos = find_sublist(big_list, prefix_ids)
#     print(start_pos)
#     end_pos = find_sublist(big_list, suffix_ids)
#     print(end_pos)
#     middle = big_list[start_pos[0]:end_pos[1]]
#     return middle

tokenizer = AutoTokenizer.from_pretrained(
    "/root/nfs/download/Llama-3.2-1B-Instruct")


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


prompt_token_ids = (
    128000, 128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25,
    6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1032, 17907, 220, 2366, 20,
    271, 128009, 128006, 882, 128007, 271, 5618, 4320, 279, 3488, 3196, 389,
    279, 22755, 3770, 627, 69803, 14546, 18710, 374, 264, 53582, 315, 3682,
    18710, 430, 13980, 994, 264, 15748, 84564, 17563, 5764, 1063,
    1376, 315, 98147, 13, 578, 98147, 1436, 387, 59123, 12922, 320, 21470, 439,
    11011, 264, 7899, 11890, 499, 430, 499, 527, 912, 1695, 477, 66121, 705,
    1624, 36807, 320, 21470, 439, 11, 19428, 16024, 315, 5922, 33028, 11, 8060,
    11, 477, 3515, 11411, 264, 7589, 8, 477, 1063, 1023, 1464, 449, 8903, 13,
    17680, 14546, 18710, 22223, 17715, 832, 704, 315, 1475, 3116, 1274,
    16584, 311, 279, 8952, 369, 18710, 13, 767, 274, 8634, 26, 274, 8634, 28,
    274, 8634, 1393, 53266, 274, 8634, 681, 20824, 11169, 6802, 663, 284, 364,
    1325, 3733, 7112, 422, 40869, 2329, 3405, 8252, 1024, 3566, 2329, 3405,
    8252, 78795, 4957, 2093, 364, 37969, 43555, 3566, 2329, 64981, 17, 43728,
    4220, 9069, 2178, 3566, 2329, 64981, 17, 43728, 43283, 16065, 284, 837, 26,
    5035, 1723, 368, 314, 3566, 2329, 558, 558, 318, 58036, 2178, 9804, 52397,
    42411, 12863, 1861, 4302, 4740, 4389, 4450, 43129, 60820, 12863, 4772,
    1397, 6561, 10191, 17, 14, 16, 33526, 14555, 43467, 42403, 22061, 4181,
    734, 41842, 28911, 8, 314, 2835, 28911, 8435, 2358, 58410, 25, 3566, 2329,
    3405, 8252, 78795, 4957, 2093, 364, 37969, 6, 1024, 758, 758, 82, 66424,
    11, 27076, 3110, 25, 16092, 7203, 11840, 291, 10983, 22283, 28, 10191,
    1483, 6750, 92117, 5787, 11251, 7, 15, 69738, 3064, 493, 7562, 292, 851,
    4670, 19815, 7199, 25, 837, 11, 12773, 14908, 25, 314, 11722, 412, 25, 837,
    2529, 3113, 43283, 25, 837, 11, 3941, 25, 364, 258, 78505, 518, 28502, 25,
    837, 6547, 18605, 18605, 335, 775, 314, 5035, 1723, 12159, 9804, 52397,
    42411, 12863, 1861, 5514, 2178, 18605, 335, 10771, 311, 279, 5165, 10181,
    315, 38895, 6401, 11, 264, 1732, 889, 374, 94241, 374, 704, 315, 5916, 449,
    8903, 13, 9029, 449, 98147, 1253, 6865, 330, 41783, 1210, 2582, 814, 1253,
    617, 15234, 323, 3900, 31356, 6848, 13, 1789, 3187, 11, 814, 1253, 1781,
    430, 3885, 649, 6865, 872, 11555, 477, 527, 4560, 311, 11682, 1124, 13,
    2582, 814, 2643, 1781, 814, 527, 43890, 555, 279, 38736, 477, 527, 4934,
    555, 279, 4379, 369, 3515, 11411, 264, 9977, 430, 814, 2216, 1550, 539,
    5379, 13, 9029, 449, 94241, 18710, 1253, 636, 19021, 369, 912, 10186, 2944,
    13, 2582, 814, 1253, 8493, 264, 2763, 315, 892, 555, 5694, 477, 304, 4950,
    11, 21811, 2391, 279, 1938, 323, 19994, 35447, 520, 3814, 13, 362, 1732,
    449, 94241, 18710, 1253, 39332, 11341, 555, 539, 73509, 477, 10223, 15447,
    13, 2582, 430, 1732, 1253, 387, 2653, 311, 3137, 311, 13, 19292, 568, 477,
    1364, 20025, 13739, 477, 775, 2795, 2574, 430, 1304, 912, 5647, 13, 9029,
    449, 1023, 10723, 49909, 11, 1778, 439, 58533, 11, 1101, 3217, 98147, 13,
    2030, 1884, 449, 94241, 18710, 6118, 617, 1624, 36807, 477, 59123, 12922,
    430, 527, 13263, 449, 22100, 922, 18710, 320, 21470, 439, 5922, 33028, 477,
    8060, 705, 20444, 94241, 13803, 304, 58533, 527, 810, 3629, 33288, 477,
    11866, 43736, 323, 617, 912, 8196, 3717, 311, 264, 20247, 1614, 320, 2000,
    3187, 11, 7422, 40721, 527, 2768, 1124, 369, 912, 2944, 1023, 1109, 311,
    19259, 1124, 570, 9029, 449, 94241, 18710, 1101, 1253, 387, 2854, 54873,
    477, 51975, 315, 279, 11555, 323, 1456, 311, 10477, 1124, 13, 44245, 779,
    3727, 420, 955, 315, 18710, 1633, 5107, 311, 58681, 13, 2030, 23842, 374,
    3062, 13, 11699, 6514, 374, 2204, 1109, 369, 2536, 92109, 14546, 18710, 13,
    7429, 11, 3515, 832, 9417, 315, 94241, 18710, 12992, 279, 6140, 315, 65919,
    19823, 449, 46350, 18243, 315, 94241, 18710, 11, 893, 689, 11, 323, 1524,
    18639, 13, 7874, 13803, 369, 6978, 889, 617, 94241, 18710, 2997, 25, 4701,
    7709, 81469, 25538, 49686, 39515, 47517, 4298, 9925, 50405, 77956, 53317,
    28479, 4998, 677, 1429, 7462, 36807, 477, 59123, 12922, 34067, 11, 6514,
    369, 94241, 18710, 374, 2728, 304, 264, 8952, 6376, 13, 3011, 1648, 11,
    279, 8893, 706, 3345, 16967, 555, 10723, 2890, 15749, 13, 34496, 31010,
    527, 1511, 311, 70236, 279, 1732, 596, 20247, 11, 11383, 2737, 28559, 315,
    65211, 1821, 323, 3276, 3153, 5759, 14546, 31010, 13, 6898, 3153, 5759,
    14546, 11217, 7958, 90351, 83189, 430, 2187, 10758, 1990, 32015, 7917, 304,
    5789, 315, 279, 8271, 430, 37377, 1057, 5845, 311, 45493, 323, 31335, 2038,
    922, 279, 1917, 2212, 603, 13, 2684, 527, 264, 1396, 315, 3276, 3153, 5759,
    14546, 11, 477, 18247, 273, 27330, 11, 31010, 17037, 1511, 3432, 13, 4314,
    2997, 10025, 716, 307, 606, 11, 8492, 12341, 391, 483, 11, 934, 51998, 391,
    483, 11, 264, 4664, 575, 8875, 1286, 11, 1841, 575, 8875, 483, 11, 323,
    439, 268, 391, 483, 13, 9062, 5623, 706, 5016, 3185, 6372, 323, 1253, 1782,
    304, 1202, 14830, 41265, 5643, 13, 34067, 11, 3582, 11, 1521, 11217, 527,
    2731, 66441, 1109, 6931, 3276, 3153, 5759, 51003, 13, 31969, 369, 94241,
    18710, 374, 1633, 7524, 13, 9029, 527, 3025, 311, 11993, 11, 6118, 2949,
    3892, 4038, 13, 2030, 77975, 6593, 1833, 5352, 1253, 387, 5995, 13, 1442,
    279, 31010, 656, 539, 990, 311, 842, 279, 98147, 323, 18710, 11, 7170,
    25396, 12296, 58921, 15419, 320, 469, 1182, 8, 374, 1511, 13, 1102, 596,
    3062, 369, 279, 8893, 311, 990, 449, 279, 10896, 311, 1505, 279, 1455,
    7524, 11217, 449, 279, 3325, 3185, 6372, 13, 9393, 94241, 18710, 374, 5115,
    6129, 11, 279, 5326, 315, 18639, 374, 1101, 2294, 627, 14924, 25, 3639,
    374, 94241, 18710, 5380, 16533, 25, 128009, 128006, 78191, 128007, 271)

jsonl_path = "/root/nfs/codespace/PrefixCacheRepo/PrefixCacheDev/vllm/simulator/token_ids.json"
with open(jsonl_path, encoding='utf-8') as f:
    config = json.load(f)
session_data = list(
    read_simulator_data("mashqa_session_data.jsonl", 10000).values())
middle_counter = Counter()
sys_set = set()
for data in session_data:
    prompt = data["turn_1"]["input_ids"]
    if type(prompt) != tuple:
        prompt = tuple(prompt)
    print("prompt len", len(prompt))
    systemp_promt = config["Llama-3.2-1B-Instruct"]["QA"]["system prompt"][
        "token ids"]
    print("systemp_promt len", len(systemp_promt))
    common_system_prompt = common_prefix(prompt, systemp_promt)
    assert list(common_system_prompt) == systemp_promt
    middle_part = get_middle_part(
        prompt,
        config["Llama-3.2-1B-Instruct"]["QA"]["prefill shared part"]["prefix"],
        config["Llama-3.2-1B-Instruct"]["QA"]["prefill shared part"]["suffix"])
    print("middle part", len(middle_part)),
    middle_counter[tuple(middle_part)] += 1
    # print("left", len(prompt) - len(common_system_prompt) - len(middle_part))
print("Top 10 most common middle parts:")
for seq, count in middle_counter.most_common(10):
    print(f"Count={count}, Length={len(seq)}")
print(min(middle_counter.values()))
