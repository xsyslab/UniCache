1. Download `ShareGPT_V3_unfiltered_cleaned_split.json`

   ```bash
   wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
   ```

2. Clean the data
   Clean the raw ShareGPT dataset before running simulation:

   ```bash
   python clean_ShareGPT_V3.py
   ```

   Result: `ShareGPT_cleaned_valid.json`

3. Deduplicate conversations
   - The first turn of some conversation are duplicated in `ShareGPT_cleaned_valid.json`.
   - To remove duplicates, run the deduplication script `python deduplicate.py` and save the output as:
     
```
     ShareGPT_cleaned_valid_deduplicated.json
     ```
     
   - Then, use `data.sh` to convert it into session format:

     ```
ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
     ```
