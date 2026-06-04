Run the analysis script to compute statistics (turns, tokens, sessions) for all `*_session_data.jsonl` files.

```bash
python analyze_data.py
```

The script will automatically read all session_data.jsonl files under the project.

For SharedGPT files, only the first 5000 samples will be analyzed to avoid memory issues.

Results are aggregated per dataset and include:

Number of sessions

Turn statistics (max / min / mean)

Input token statistics (max / min / mean)

Output token statistics (max / min / mean)

Last turn token statistics (max / min / mean / total)
