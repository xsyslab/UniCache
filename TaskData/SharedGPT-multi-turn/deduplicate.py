# SPDX-License-Identifier: Apache-2.0
import json
from collections import Counter

path = "ShareGPT_cleaned_valid.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print(f"{len(data)} Session")

first_turns = []
id_to_firstq = {}

for item in data:
    convs = item.get("conversations", [])
    if convs and convs[0].get("from") == "user":
        first_q = convs[0].get("value", "").strip()
        first_turns.append(first_q)
        id_to_firstq[item.get("id")] = first_q

# Count duplicates
counter = Counter(first_turns)
unique_ids = [id_ for id_, q in id_to_firstq.items() if counter[q] == 1]
print(f"Number of unique first user questions: {len(unique_ids)}")
print("Sample unique ids:", unique_ids[:10])  # show first 10

new_result = []

for item in data:
    if item.get("id") in unique_ids:
        new_result.append(item)

print(f"Number of unique first user questions: {len(new_result)}")
print("Sample unique ids:", new_result[:10])  # show first 10

output_path = "ShareGPT_cleaned_valid_deduplicated.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_result, f, ensure_ascii=False, indent=2)

print(f"Saved {len(new_result)} unique items to {output_path}")
