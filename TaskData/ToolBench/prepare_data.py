# SPDX-License-Identifier: Apache-2.0
import json
import random
import statistics as stats
import uuid

SYSTEM_PROMPT = """System: You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names."""

# Read the JSON file
with open("G1_workload_updated_input_output_lengths_4096.json",
          encoding="utf-8") as f:
    data = json.load(f)

all_result = []
shared_count = []
for api_names, value in data.items():
    # generate the data
    if len(value) > 200:
        continue
    mu = 7.07
    sigma = 1.26
    constraint = int(random.gauss(mu, sigma))
    constraint = max(1, min(constraint, len(value)))
    shared_count.append(constraint)
    selected_items = random.sample(value, constraint)
    for item in selected_items:
        new_item = dict()
        new_item["id"] = "ToolBench-" + str(uuid.uuid4())
        new_item["conversations"] = []

        new_item["conversations"].append({
            "from": "system",
            "value": SYSTEM_PROMPT
        })

        query_raw = item["prompt"]
        if query_raw.startswith(SYSTEM_PROMPT):
            query_raw = query_raw[len(SYSTEM_PROMPT):].lstrip()
        suffix = "Assistant: "
        if query_raw.endswith(suffix):
            query_raw = query_raw[:-len(suffix)].rstrip()

        new_item["conversations"].append({"from": "user", "value": query_raw})
        new_item["conversations"].append({
            "from": "assistant",
            "value": item["output"]
        })
        all_result.append(new_item)
print("Data num ", len(all_result))
mean = stats.mean(shared_count)
median = stats.median(shared_count)
stdev = stats.stdev(shared_count)
min_val = min(shared_count)
max_val = max(shared_count)
print(
    f"✅ Mean: {mean}, Median: {median}, Stdev: {stdev}, Min: {min_val}, Max: {max_val}"
)
with open("ToolBench.json", "w", encoding="utf-8") as f:
    json.dump(all_result, f, ensure_ascii=False, indent=2)
