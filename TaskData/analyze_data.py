# SPDX-License-Identifier: Apache-2.0
import glob
import json
import os

import pandas as pd


def analyze(data):

    result = {}
    result["num_sessions"] = len(data)

    num_turns_list = []
    input_token_list = []
    output_token_list = []
    session_total_tokens = []

    for sample in data:
        item = list(sample.values())[0]

        num_turns = item["num_turns"]
        num_turns_list.append(num_turns)

        session_token = 0
        for turn in range(1, num_turns + 1):
            input_len = len(item[f"turn_{turn}"]["input_ids"])
            output_len = len(item[f"turn_{turn}"]["response_ids"])

            input_token_list.append(input_len)
            output_token_list.append(output_len)
        last_input = len(item[f"turn_{num_turns}"]["input_ids"])
        last_output = len(item[f"turn_{num_turns}"]["response_ids"])
        session_token += last_input + last_output

        session_total_tokens.append(session_token)

    # turns
    result["max_turns"] = round(max(num_turns_list), 1)
    result["min_turns"] = round(min(num_turns_list), 1)
    result["mean_turns"] = round(sum(num_turns_list) / len(num_turns_list), 1)

    # input
    result["max_input_token"] = round(max(input_token_list), 1)
    result["min_input_token"] = round(min(input_token_list), 1)
    result["mean_input_token"] = round(
        sum(input_token_list) / len(input_token_list), 1)

    # output
    result["max_output_token"] = round(max(output_token_list), 1)
    result["min_output_token"] = round(min(output_token_list), 1)
    result["mean_output_token"] = round(
        sum(output_token_list) / len(output_token_list), 1)

    # per session total
    result["max_session_total_tokens"] = round(max(session_total_tokens), 1)
    result["min_session_total_tokens"] = round(min(session_total_tokens), 1)
    result["mean_session_total_tokens"] = round(
        sum(session_total_tokens) / len(session_total_tokens), 1)
    result["all_session_total_tokens"] = round(sum(session_total_tokens), 1)

    return result


files = glob.glob("./**/*.jsonl", recursive=True)

print("Found files:")
for f in files:
    print(f)

files = [
    "./AgentBankData/AgentBank_hotpotqa_session_data.jsonl",
    "./AgentBankData/AgentBank_mbpp_before_session_data.jsonl",
    "./AgentBankData/AgentBank_mathqa_session_data.jsonl",
    "./AgentBankData/AgentBank_math_session_data.jsonl",
    "./AgentBankData/AgentBank_mbpp_session_data.jsonl",
    "./AgentBankData/AgentBank_alfworld_session_data.jsonl",
    "./AgentBankData/AgentBank_apps_session_data.jsonl",
    "./AgentBankData/AgentBank_intercode_bash_session_data.jsonl",
    "./AgentBankData/AgentBank_webarena_session_data.jsonl",
    "./AgentBankData/AgentBank_rearrange_session_data.jsonl",
    "./AgentBankData/AgentBank_alfred_session_data.jsonl",
    "./AgentBankData/AgentBank_humaneval_session_data.jsonl",
    "./AgentBankData/AgentBank_iqa_session_data.jsonl",
    "./AgentBankData/AgentBank_mind2web_session_data.jsonl",
    "./AgentBankData/AgentBank_triviaqa_session_data.jsonl",
    "./AgentBankData/AgentBank_strategyqa_session_data.jsonl",
    "./AgentBankData/AgentBank_intercode_sql_session_data.jsonl",
    "./AgentBankData/AgentBank_webshop_session_data.jsonl",
    "./AgentBankData/AgentBank_gsm8k_session_data.jsonl",
    "./Codeparrot/Code_session_data.jsonl",
    "./mashqa_data/mashqa_session_data.jsonl",
    "./SharedGPT-multi-turn/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl",
    "./SharedGPT-multi-turn/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl",
    "./SharedGPT-multi-turn/ShareGPT_cleaned_valid_with_unique_prefix_session_data.jsonl",
    "./SharedGPT-multi-turn/ShareGPT_cleaned_valid_session_data.jsonl",
    "./Multi-turn-time-stamp/SharedGPT_multi_turn_with_time_stamp_3735.jsonl",
    "./Multi-turn-time-stamp/SharedGPT_multi_turn_with_time_stamp_1052.jsonl",
    "./Multi-turn-time-stamp/ShareGPT_cleaned_valid_deduplicated_session_data_with_time_stamp_3735_29.0_min.jsonl",
    "./ToolBench/ToolBench_session_data.jsonl",
]
all_result = []

for path in files:
    with open(path, encoding="utf-8") as f:
        if "SharedGPT" in path:

            data = []
            for i, line in enumerate(f):
                if i >= 5000:
                    break
                data.append(json.loads(line))
        else:
            # 正常读取
            data = [json.loads(line) for line in f]
            data = data[:5000]
        result = analyze(data)  # 你之前写的统计函数
        result["file"] = os.path.basename(path)
        all_result.append(result)
        print(result)

df = pd.DataFrame(all_result)
cols = ["file"] + [c for c in df.columns if c != "file"]
df = df[cols]
df.to_csv("session_data_analysis_5000.csv", index=False, encoding="utf-8")
print("✅ Results saved to session_data_analysis.csv")
