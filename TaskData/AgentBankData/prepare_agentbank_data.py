# SPDX-License-Identifier: Apache-2.0
import json

from datasets import load_dataset

task_list = [
    'alfworld', 'apps', 'gsm8k', 'hotpotqa', 'humaneval', 'intercode_bash',
    'intercode_sql', 'iqa', 'math', 'mathqa', 'mbpp', 'mbpp_before',
    'mind2web', 'rearrange', 'strategyqa', 'triviaqa', 'webarena', 'webshop',
    "alfred"
]


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
        return ValueError(f"Unknown role: {role}")


def prepare_AgentBank_data(task):
    dataset = load_dataset("Solaris99/AgentBank", task)
    subset = dataset["train"][:]
    sample_list = [
        dict(zip(subset.keys(), values)) for values in zip(*subset.values())
    ]
    for sample in sample_list:
        if "conversations" in sample:
            for msg in sample["conversations"]:
                if "from" in msg:
                    role = normalize_role(msg["from"])
                    msg["from"] = role
    with open(f"AgentBank_{task}.json", "w", encoding="utf-8") as f:
        json.dump(sample_list, f, indent=2, ensure_ascii=False)


for idx, task in enumerate(task_list, 1):
    print(f"\n=== [{idx}/{len(task_list)}] Processing: {task} ===")
    prepare_AgentBank_data(task)
