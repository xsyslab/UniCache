# SPDX-License-Identifier: Apache-2.0
import argparse
import json


def parser_args():
    args = argparse.ArgumentParser()
    args.add_argument("--raw_data_path",
                      type=str,
                      default="ShareGPT_V3_unfiltered_cleaned_split.json")
    args.add_argument("--clean_data_path",
                      type=str,
                      default="ShareGPT_cleaned_valid.json")
    return args.parse_args()


def normalize_role(role):
    # Normalize role names to one of: 'user', 'assistant', or 'system'.
    role = role.lower()
    if role in ["human", "user"]:
        return "user"
    elif role in ["gpt", "chatgpt", "bing", "bard"]:
        return "assistant"
    elif role == "system":
        return "system"
    else:
        return None


def filter_role(data):
    """
    Process each conversation by:

    1. Normalize role labels to standard values: 
       'user', 'assistant', or 'system'.

    2. Remove all 'system' messages.
       These messages typically appear at the beginning 
       of conversations and are platform watermarks 
       (e.g., "This chat conversation is shared from TypingMind.com").

    3. Remove leading 'assistant' messages to ensure conversations always start
       with a user input.
    """
    for sample in data:
        conversation = sample.get("conversations", [])
        new_conversation = []
        for msg in conversation:
            role = normalize_role(msg.get("from", ""))
            value = msg.get("value", "").strip()
            if role and value:
                new_conversation.append({"from": role, "value": value})
        if new_conversation and new_conversation[0]["from"] == "system":
            new_conversation.pop(0)
        while new_conversation and new_conversation[0]["from"] in [
                "assistant"
        ]:
            new_conversation.pop(0)
        sample["conversations"] = new_conversation
    return data


def is_valid_conversation(conversations):
    """
    Check if a conversation is valid:
    1. Must contain at least one user-assistant pair (minimum 2 turns).
    2. All roles must be either 'user' or 'assistant'.
    3. No empty messages.
    4. Turns must alternate strictly as user → assistant → user → ...
    """
    if len(conversations) < 2:
        return False
    valid_roles = {"user", "assistant"}
    for i, msg in enumerate(conversations):
        role = msg.get("from")
        content = msg.get("value", "").strip()
        # 1. role must be valid
        if role not in valid_roles:
            return False
        # 2. content must be non-empty
        if not content:
            return False
        # 3. role alternates: user → assistant → user ...
        expected_role = "user" if i % 2 == 0 else "assistant"
        if role != expected_role:
            return False
    return True


# load raw data
args = parser_args()
with open(args.raw_data_path) as f:
    data = json.load(f)
print("Load data from:", args.raw_data_path, "Sample count:", len(data))
filter_role(data)
cleaned_data = []
for sample in data:
    conversations = sample.get("conversations", [])
    if is_valid_conversation(conversations):
        cleaned_data.append(sample)
        sample["id"] = "ShareGPT-multi-turn-" + sample["id"]

with open(args.clean_data_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
print(f"Cleaned data saved to {args.clean_data_path} "
      f"(total samples: {len(cleaned_data)})")
