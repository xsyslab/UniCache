# SPDX-License-Identifier: Apache-2.0
import copy
import json
import math
import random
import uuid

path = "AgentBank_alfred.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)
# Target number of samples after scaling
target_num = 5000
original_num = len(data)
# Calculate how many times each sample should be duplicated on average

avg = math.ceil(target_num / original_num)
new_result = []
for item in data:
    for i in range(avg):
        new_item = copy.deepcopy(item)
        # Assign a new unique ID to avoid duplication conflicts
        new_item["id"] = f"{new_item['id']}-{uuid.uuid4()}"
        # Modify the first conversation message to ensure uniqueness
        # Avoid cross session prefix share
        new_item["conversations"][0]["value"] = str(
            uuid.uuid4()) + new_item["conversations"][0]["value"]
        new_result.append(new_item)

new_result = new_result[:target_num]
# shuffle
random.shuffle(new_result)

save_path = f"AgentBank_alfred_scale_{target_num}.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(new_result, f, indent=4, ensure_ascii=False)

print(f"✅ Done! Saved {len(new_result)} items to {save_path}")
