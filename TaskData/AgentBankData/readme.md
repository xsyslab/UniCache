# AgentBank Simulation

## 1. Prepare raw data

```bash
python prepare_agentbank_data.py
```

## 2. Prepare the session data

```bash
bash data.sh
```

## 3. Run the simulation

```bash
bash run.sh
```

## 4. Scale data

To increase the dataset size, use the scaling script:

```bash
python Alfred_scale_up.py
```

Example: increase from 623 to 15,000.
For Alfred (multi-turn conversation), a unique prefix is added to the beginning of each conversation to avoid cross-session prefix sharing.
