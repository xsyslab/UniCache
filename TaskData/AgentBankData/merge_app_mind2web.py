# SPDX-License-Identifier: Apache-2.0
import shutil

# 输入文件路径
file1 = "AgentBank_apps_session_data.jsonl"
file2 = "AgentBank_mind2web_session_data.jsonl"
output = "AgentBank_single_api_session_data.jsonl"

# === 方法 1：最简单高效（逐行合并） ===
with open(output, "w", encoding="utf-8") as fout:
    for fname in [file1, file2]:
        with open(fname, encoding="utf-8") as fin:
            shutil.copyfileobj(fin, fout)

print(f"✅ 已合并文件并保存为: {output}")
