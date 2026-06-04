# OPT Eviction Policy:
The OPT eviction policy always evicts the block whose **next reuse time** is farthest in the future.
+ If a block will never be reused, its next-reuse time is treated as $+\infty$, making it the highest-priority candidate for eviction.

+ Offline algorithm:
  + Requires full knowledge of the future access sequence.
  + This version is **theoretically optimal** but not implementable in practice because real systems cannot know future requests.

+ Online algorithm:

  + The online version approximates OPT by using only partial future information.
  + The algorithm tracks arrivals of all arrival requests.
  + Some requests remain in a **waiting queue**, meaning their future accesses are **partially known**.
  + Only the blocks that belong to requests already in the waiting queue can have a computable **next-reuse time**.
  + Blocks that belong to future, unseen requests cannot be assigned a correct reuse time. (defulalt:  $+\infty$)
    + These blocks fall back to an **LRU-based ordering** until more information becomes available.
  + **Worst case**: the online OPT algorithm degenerates to pure LRU,

## Offline-OPT Eviction Workflow
This section describes how to collect execution traces from vLLM, extract block use information, and run the OPT (optimal) eviction policy.
### 1. Collect Execution Trace

``` shell
bash run_profile.sh
```
During profiling, the simulator records the **relative execution order** of each request’s **prefill iterations**.

We only care about **prefill order** because prefill determines when each KV-cache block is reused.

For every prefill iteration, the simulator logs:

+ which request is executed

All trace records are saved into a JSON file: `trace_data.json`

Each top-level key in trace_data.json is an iteration index starting from 0, representing the **chronological order** in which requests execute their prefill steps.

Example:
```
  "0": {
    "id_ShareGPT-multi-turn-wNBG8Gp_51_turn_1": {
      "is_prompt": true,
      "prompt_token_len": 42
    }
  },
  "1": {
    "id_ShareGPT-multi-turn-wNBG8Gp_51_turn_2": {
      "is_prompt": true,
      "prompt_token_len": 219
    }
  },
```
In this example:

+  "0" is the first prefill execution.
+  "1" is the second prefill execution.

Thus, the keys ("0", "1", "2", …) encode the **relative prefill order**, which is the only information needed to compute block reuse times.

**Computing Block Use**

After obtaining `trace_data.json`, we compute the **use timeline** for each KV-cache block:

+ Each block is identified by its **content hash**.

+ A block is considered **used** only when a prefill iteration uses it.

+ Since OPT needs only the **relative reuse times**, the prefill order from trace_data.json is sufficient to reconstruct each block’s reuse timeline.

The block-use map is a dictionary:

``` arduino
{
  "<block_hash>": [prefill_iter1, prefill_iter2, ...]
}
```
Example:
``` json
{
  "-4494557957337493199": [0, 2]
}

```
This means:
+ The block with hash -4494557957337493199 was used at prefill iteration 0 and again at iteration 2

If a block is never used, the iteration list will be empty.




### 2. Run Offline OPT Eviction
``` shell
bash run.sh
```
The flag  `--opt-block-use-map-path` should be the path of the block use map file saved in **step1**



**Tracking Prefill Progress**

+ During inference, the system maintains a global counter called **PREFILL_CNT**, which records how many prefill requests have already been processed.
+ **PREFILL_CNT** represents the current time for OPT.



Note:

+ OPT relies on stable block content hashing.

+ To guarantee deterministic hashes, you must set: `PYTHONHASHSEED=0`. This ensures that the block content hashes remain consistent across different runs.


## Online-OPT

Online OPT only tracks requests that have already arrived or are currently waiting to be processed.


**Assigning Request Counters**
+ When a new request arrives, it is assigned a monotonically increasing request counter.
+ This counter represents the relative arrival and execution order of all requests in the system.
+ By assigning each request a unique req_counter value, we can determine the use time of every KV-cache block belonging to that request.

**Tracking Prefill Progress**

+ During inference, the system maintains a global counter called **PREFILL_CNT**, which records how many prefill requests have already been processed.
+ **PREFILL_CNT** represents the current time.


**Computing Next-Use Time**

For each block, we store a list of all use time.

To compute the block’s next reuse time:
+ Remove all use times ≤ PREFILL_CNT, since those reuses have already occurred.
+ Compare the remaining future use times with the current PREFILL_CNT.
  + If there exists a future use, the earliest one becomes the block’s next-use time.
  + If no future uses remain, the block’s next-use time is treated as $+\infty$.

**Eviction Decision**

+ The block with the smallest next-use time is evicted.
+ If multiple blocks have the same next-use time, the block with the largest last access time is evicted.

**Degeneration to LRU**
+ Online OPT relies on partial future-use information.
+ When most blocks belong to future, unseen requests, their reuse times cannot be computed and default to $+\infty$.
+ In this case, Online OPT degenerates to LRU.
