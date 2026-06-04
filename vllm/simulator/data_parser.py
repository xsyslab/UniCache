# SPDX-License-Identifier: Apache-2.0
import json

import ijson
from tqdm import tqdm

from vllm.logger import init_logger

logger = init_logger(__name__)


def get_prompt_text(tokenizer, conversations, add_generation_prompt=True):
    chat = []
    for turn in conversations:
        # Normalize each turn to the format expected by the chat template
        chat.append({"role": turn["from"], "content": turn["value"].strip()})
    ends_with_user = chat[-1].get("role", "").lower() == "user"
    # Validation: user should end the conversation before generation
    if not ends_with_user and add_generation_prompt:
        raise ValueError(
            "The last turn must be from 'user' when add_generation_prompt=True."
        )
    if ends_with_user and not add_generation_prompt:
        raise ValueError("The last turn must be from 'assistant' when "
                         "add_generation_prompt=False.")
    # If add_generation_prompt is True, a placeholder will be appended to signal
    # the model that an assistant response is expected next.
    # Note date_string is fixed to "13 Sep 2025", this is used to make the system prompt deterministic
    prompt_text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        date_string="13 Sep 2025")
    return prompt_text


def prepare_session_data_with_system_prompt(entry, tokenizer):
    id = entry["id"]
    conversations = entry["conversations"]
    assert conversations[0]["from"] == "system"
    turn = (len(conversations) - 1) // 2
    result = dict()
    result["id"] = id
    result["num_turns"] = turn
    for i in range(1, turn + 1):
        key = f"turn_{i}"
        result[key] = dict()
        input_text = get_prompt_text(tokenizer,
                                     conversations[:2 * i],
                                     add_generation_prompt=True)
        # input ids have bos token
        input_ids = tokenizer.encode(input_text,
                                     return_tensors=None,
                                     add_special_tokens=True)
        # add eos
        result[key]["input_text"] = input_text
        result[key]["input_ids"] = input_ids
        try:
            response_text = conversations[2 * i]["value"] + tokenizer.decode(
                [tokenizer.eos_token_id])
            # avoid bos token
            response_ids = tokenizer.encode(response_text,
                                            return_tensors=None,
                                            add_special_tokens=False)
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = response_ids
        except Exception:
            # no response
            response_text = tokenizer.decode([tokenizer.eos_token_id])
            response_ids = [tokenizer.eos_token_id]
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = response_ids
    return result


def prepare_session_data(entry, tokenizer):
    id = entry["id"]
    conversations = entry["conversations"]
    if conversations[0]["from"] == "system":
        return prepare_session_data_with_system_prompt(entry, tokenizer)
    turn = len(conversations) // 2
    result = dict()
    result["id"] = id
    result["num_turns"] = turn
    for i in range(1, turn + 1):
        key = f"turn_{i}"
        result[key] = dict()
        input_text = get_prompt_text(tokenizer,
                                     conversations[:2 * i - 1],
                                     add_generation_prompt=True)
        # input ids have bos token
        input_ids = tokenizer.encode(input_text,
                                     return_tensors=None,
                                     add_special_tokens=True)
        # add eos
        result[key]["input_text"] = input_text
        result[key]["input_ids"] = input_ids
        try:
            response_text = conversations[2 * i -
                                          1]["value"] + tokenizer.decode(
                                              [tokenizer.eos_token_id])
            # avoid bos token
            response_ids = tokenizer.encode(response_text,
                                            return_tensors=None,
                                            add_special_tokens=False)
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = response_ids

            # full_text = get_prompt_text(tokenizer,
            #                             conversations[:2 * i],
            #                             add_generation_prompt=False)
            # full_ids = tokenizer.encode(full_text, return_tensors=None)
            # result[key]["full_text"] = full_text
            # result[key]["full_ids"] = full_ids
        except Exception:
            # no response
            response_text = tokenizer.decode([tokenizer.eos_token_id])
            response_ids = [tokenizer.eos_token_id]
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = response_ids
            # all conversation
            # full_text = get_prompt_text(
            #     tokenizer, conversations,
            #     add_generation_prompt=True) + tokenizer.decode(
            #         [tokenizer.eos_token_id])
            # result[key]["full_text"] = full_text
            # full_ids = tokenizer.encode(full_text, return_tensors=None)
            # result[key]["full_ids"] = full_ids
    return result


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


def prepare_simulator_data_streaming_jsonl(raw_data_path, tokenizer,
                                           save_path):
    print("[Simulator] Streaming data preparation -> JSONL format")

    with open(raw_data_path, encoding="utf-8") as f_in, \
        open(save_path, "w", encoding="utf-8") as f_out:
        parser = ijson.items(f_in, "item")
        progress = tqdm(desc="Processing entries", unit="entry")
        for entry in parser:
            result = prepare_session_data(entry, tokenizer)
            json_line = json.dumps({result["id"]: result}, ensure_ascii=False)
            f_out.write(json_line + "\n")  # 一行一个 session
            progress.update(1)
        progress.close()

    print(f"✅ Streaming session data saved to {save_path} (JSONL format)")
