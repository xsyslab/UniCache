# SPDX-License-Identifier: Apache-2.0
import json
import random
import statistics as stats

from tqdm.auto import tqdm

file_list = [
    "train_webmd_squad_v2_full.json", "test_webmd_squad_v2_full.json",
    "val_webmd_squad_v2_full.json"
]
all_data = []
for path in file_list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
        all_data.extend(data)  # 合并进总列表


def build_raw_sample(item):
    prar = item["paragraphs"][0]
    doc = prar['context']
    res = []
    constraint = int(random.gauss(mu=5.74, sigma=1.0))
    for i in range(min(constraint, len(prar['qas']))):
        qa = prar['qas'][i]
        question = qa['question']
        answers = qa['answers'][0]["text"]
        sample = dict()
        sample["id"] = "QA-" + qa["id"]
        sample["conversations"] = []
        prompt_format = (
            f"Please answer the question based on the texts below.\n"
            f"{  doc}\n"
            f"Question: { question}\n"
            f"Answer: ")
        sample["conversations"].append({
            "from": "user",
            "value": prompt_format
        })
        sample["conversations"].append({"from": "assistant", "value": answers})
        res.append(sample)
    return res


shared_count = []

raw_data = []
for item in tqdm(all_data,
                 total=len(all_data),
                 desc="Building samples",
                 unit="sample"):
    samples = build_raw_sample(item)
    shared_count.append(len(samples))
    raw_data.extend(samples)

print(f"Got {len(raw_data)} valid samples")
mean = stats.mean(shared_count)
median = stats.median(shared_count)
stdev = stats.stdev(shared_count)
min_val = min(shared_count)
max_val = max(shared_count)
print(
    f"Mean: {mean}, Median: {median}, Stdev: {stdev}, Min: {min_val}, Max: {max_val}"
)
with open("mashqa.json", "w", encoding="utf-8") as f:
    json.dump(raw_data, f, indent=2, ensure_ascii=False)
