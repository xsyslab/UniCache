# SPDX-License-Identifier: Apache-2.0
import ast
import json
import random
import statistics as stats
import uuid

from datasets import load_dataset


def parse_io(io_raw):
    if isinstance(io_raw, dict):
        return io_raw
    if io_raw is None:
        return {"inputs": [], "outputs": []}
    s = str(io_raw).strip()
    if not s:
        return {"inputs": [], "outputs": []}
    try:
        return json.loads(s)
    except Exception:
        # 偶尔是单引号或轻微格式问题
        try:
            return ast.literal_eval(s)
        except Exception:
            return {"inputs": [], "outputs": []}


def parse_solutions(sol_raw):
    if isinstance(sol_raw, list):
        return sol_raw
    if sol_raw is None:
        return []
    s = str(sol_raw).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else [s]
    except Exception:
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else [s]
        except Exception:
            return [s]


def build_raw_sample(sample):
    question = sample.get("question", "")
    io = parse_io(sample.get("input_output"))
    inputs = io.get("inputs", []) or []
    outputs = io.get("outputs", []) or []

    inputs = inputs[:1]
    outputs = outputs[:1]

    input_output_strings = ""
    for x in inputs:
        input_output_strings += str(x) + "\n"
    for y in outputs:
        input_output_strings += str(y) + "\n"
    input_format = "\nUse Standard Input format" if not sample.get(
        "fn_name") else "\nUse Call-Based format"
    sample_prompt = f"Question: {question} {input_format} {input_output_strings} \n\nAnswer:\n"
    sols = parse_solutions(sample.get("solutions"))
    result = []
    sample_id = "Code-" + str(uuid.uuid4())

    mu, sigma = 6.5, 1.5
    if len(sols) == 0:
        selected_indices = []  # 没有可选的
    else:
        constraint = int(random.gauss(mu, sigma))
        constraint = max(1, min(constraint, len(sols)))
        selected_indices = random.sample(range(len(sols)), constraint)

    for idx in selected_indices:
        sol = sols[idx]
        item = dict()
        item["id"] = f"{sample_id}-{idx}"
        item["conversations"] = []
        item["conversations"].append({"from": "user", "value": sample_prompt})
        item["conversations"].append({"from": "assistant", "value": str(sol)})
        result.append(item)
    return result


ds = load_dataset("codeparrot/apps",
                  split="test",
                  difficulties=["introductory"],
                  trust_remote_code=True)
print(ds)
raw_data = []
shared_count = []

for sample in ds:
    res = build_raw_sample(sample)
    shared_count.append(len(res))

    raw_data += res
print("Data num ", len(raw_data))
mean = stats.mean(shared_count)
median = stats.median(shared_count)
stdev = stats.stdev(shared_count)
min_val = min(shared_count)
max_val = max(shared_count)
print(
    f" Mean: {mean}, Median: {median}, Stdev: {stdev}, Min: {min_val}, Max: {max_val}"
)
# print(raw_data)
with open("Code.json", "w", encoding="utf-8") as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=2)
