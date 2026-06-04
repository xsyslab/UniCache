# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class TaskType(str, Enum):
    MULTI_TURN_TEXT = "multi_turn_text"
    MULTI_TURN_FILE = "multi_turn_file"
    MULTI_TURN_IMAGE = "multi_turn_image"
    MULTI_TURN_SEARCH = "multi_turn_search"
    MULTI_TURN_AGENTIC = "multi_turn_agentic"

    SINGLE_TURN = "single_turn"
    SINGLE_TURN_RANDOM = "single_turn_random"

    DEFAULT = "default"


class BlockType(str, Enum):
    PREFILL_SYSTEM = "prefill_system"
    PREFILL_SHARED = "prefill_shared"
    PREFILL_NOT_SHARED = "prefill_not_shared"
    DECODE = "decode"
    DEFAULT = "default"


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


def find_sublist(big, small):
    """
    General-purpose subsequence finder.
    Supports `big` as a tuple and `small` as either list or tuple.
    Returns (start_idx, end_idx) if found, otherwise None.
    """
    n, m = len(big), len(small)
    if m == 0 or n == 0 or m > n:
        return None

    # When comparing tuple vs list, convert the smaller one to tuple
    # (avoids full list conversion of `big` for efficiency)
    small_tuple = tuple(small)
    for i in range(n - m + 1):
        if big[i:i + m] == small_tuple:
            return (i, i + m)
    return None


def get_middle_part(big_list, prefix_ids, suffix_ids):
    """
    Extract the subsequence between prefix_ids and suffix_ids from big_list.
    Automatically supports mixed types (tuple + list).
    Returns the middle part as a list.
    """
    start_pos = find_sublist(big_list, prefix_ids)
    end_pos = find_sublist(big_list, suffix_ids)

    if start_pos is None or end_pos is None:
        print(
            f"[WARN] prefix/suffix not found: start={start_pos}, end={end_pos}"
        )
        return []

    # Tuple slicing is efficient (creates a new tuple referencing original memory)
    return list(big_list[start_pos[1]:end_pos[0]])
