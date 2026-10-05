# Open Issues

What is not settled, not validated, or known to be wrong. Read this before trusting a number or starting new work.

> Split out of the single `sim/HANDOFF.md` during the Milestone 2 cleanup.
> **Section numbers are the original HANDOFF numbers and are deliberately not
> renumbered**, so cross-references like "supersedes §3 decision 4" still
> resolve. They are not sequential within a file. The other parts live in
> `ARCHITECTURE.md`, `DECISIONS.md`, `FINDINGS.md` and `OPEN_ISSUES.md`.

---

## 6. Validation status

Invariants pass across 3 seeds (3, 11, 42) × 2 context scales (480, 4000):

- no KV leak (`pinned_bytes == 0` at end)
- no refcount leak (all entries at 0)
- budget never exceeded
- every DAG node completes
- peak KV utilisation 99.5–99.9%

The engine reaches the contended state the old one could not. Trace evidence:

```
[115.388s]  ADMIT req16 t4/evaluate_plan MISS prefill=2160tok batch=7 kv=1344/1536MB
[115.493s]  EVICT t0:issue (24MB, held 109.0s, rent 0.00s)
[115.493s]  BLOCKED 1 req(s) on KV: req17 needs space; kv=1294/1536MB
            (pinned 867MB, retained 427MB)
[115.598s]  ADMIT req17 t3/evaluate_plan MISS prefill=2195tok batch=8 kv=1526/1536MB
```

---


## 9. Known limitations — read before trusting any number

1. **Prefill is unchunked and unbatched.** One prefill per iteration, so total
   prefill throughput is capped at ~268 tok/s regardless of B. Two consequences:
   a 26,597-token prefill stalls the whole batch for 99 seconds, and the system
   becomes prefill-bound so capacity is understated by roughly an order of
   magnitude. Real engines batch prefill tokens across requests and chunk them
   (vLLM defaults to 512-token chunks). **This is the top-priority fix.**
2. **Rent is charged only on blocking.** Under LRU the engine rarely blocks — it
   just evicts. The common real opportunity cost of holding E is "F was evicted
   and will be recomputed", which the current model does not capture. Rent is
   consequently too sparse to drive a policy. Reworking this changes what the
   competitive-ratio bound is a bound *on*, so it is a design decision, not a
   patch. **RESOLVED — design recorded in §13.** Implementation lands in
   Milestone 1 Phase 4; the old model stays selectable as `blocking_rent`.
3. **`batch_decode_coeff = 0.15` is a placeholder** and is load-bearing — it
   decides how much throughput a blocked request forgoes. Sweep it in `ablate.py`.
4. Rates are Mac/Metal, not CUDA.
5. `model_switch_seconds = 0.0` assumes both models stay resident.
6. `REPRESENTATIVE_ISSUE` in `measure_context.py` is invented, not real
   `parse_issue` output. Small share of the 480 but not zero.
7. Benchmark runs above are heavily oversaturated (p50 ~12,000s). Fine for
   demonstrating contention, wrong for measuring policy performance. Arrival rate
   needs tuning per config for ~70–90% utilisation.

---


## 15. Ollama prompt-cache model — UNVALIDATED (diagnosis superseded)

> The scores and diagnosis below predate 2026-10-04. The mechanism they blame
> was retracted; see FINDINGS §19 for what actually predicts a cache hit, and
> §20 below for why these two traces cannot settle it. Kept for the record.


`simulator/ollama_cache.py` models the confirmed mechanism (one slot per model,
0.10 prefix-similarity threshold, host-RAM cache of displaced states) and
replays both committed traces against the duration-based hit detector. It does
not reproduce the measurement, so it is **not** adopted as a baseline.

| rule | original | shared_first |
|---|---|---|
| displaced state destroyed on in-place reuse | 36 / 89 | 16 / 78 (10 FP) |
| displaced state always saved, slot path wins | 56 / 89 | 20 / 78 (10 FP) |

Both rules predict the affinity probe correctly, because in that probe A and B
were the same length and sat in symmetric positions — it cannot separate them.
The traces can, and neither survives.

### Where it fails, and the part that is not explained

`issue_parser` accounts for almost all of the gap: 45 measured hits in each
trace, 20 predicted on `original` and 0 on `shared_first`. These are real hits,
not an artefact of the detector — the prompts are the same size in both buckets
(~165 tokens), but hits take 0.04s and non-hits 0.55s, and the non-hits land on
the calibrated 268 tok/s exactly. An upper-bound scan confirms why they should
be cheap: for every one of those 45 calls, an **exact** duplicate prompt
(best-match prefix fraction 1.000, 100% exact) exists earlier in the trace,
a median of 152 calls back on `original` and 107 on `shared_first`. The same
HumanEval task is re-run under each of the four configs, so its issue_parser
prompt recurs byte-identically.

The model loses those states before they can be restored. On `shared_first`
every prompt shares a long prefix with every other by construction, so slot
reuse fires on 186 of 214 calls and the RAM cache is consulted 5 times — yet
reality still restores all 45. So the real server must reach a saved state that
the slot-priority rule says is unreachable, and no rule tried here does that
while remaining consistent with the probe.

Two candidates not yet tested, in order of promise:
1. More than one state is resident. `-np 1` is one *slot*, but the host cache
   may be consulted on every prompt, not only when the slot match is poor.
2. Similarity may be computed over tokens against the *slot's* length rather
   than the incoming prompt's. This changes which branch fires for the short
   issue_parser prompts specifically, which is exactly where the gap is.

Both are distinguishable by a probe where A and B have **different lengths** —
the symmetry that defeated the last one. Design that before modelling further.


## 11. CLOSED — "next steps, in order"

> Resolved during Milestone 1 and superseded by `docs/ROADMAP.md`. Items 1-5
> are done: chunked and batched prefill, the derived-rent rework (DECISIONS
> §13), re-baselining at 60-80% utilisation, and the full policy set in
> `simulator/policies.py`. Item 6's `run.py`/`sweep.py`/`ablate.py` became
> `experiments/run_retention_grid.py`. Kept for the record.


1. **Chunked + batched prefill.** Blocks everything else quantitative.
2. **Sample stable-context size from the real distribution** (§5) rather than one
   constant.
3. **Rework the rent model** to charge for eviction pressure — needs the user's
   decision first.
4. **Re-baseline at 70–90% utilisation.**
5. **Then Phase 2**: `simulator/policies.py` — retention (LRU, keep-all, ski-rental,
   ski-rental-with-predictions, offline Belady/OPT) and schedulers (FIFO,
   static-priority, EDF, least-slack-first).
6. **Phase 3**: `run.py`, `sweep.py`, `ablate.py`.

Note on scope: mechanism 2 (least-slack-first scheduling) is **not** affected by
the context-scale problem. Contention is real and heavy at every scale tested; at
toy scale it is simply *active-KV* contention, which is a scheduling and
admission problem rather than a retention one. That half can proceed independently.


## 14. MAARA writeup issues

Found while wiring trace logging. **No code was changed for any of these** —
they are recorded because they affect what the existing results can be claimed
to show.

### `single_agent` is not a single agent

The config name says single-agent, but `orchestrator.py` gates the mechanisms by
*exclusion*:

```python
use_rag = (config != "hetero_no_rag")        # orchestrator.py:54
if config != "hetero_no_sentinel":            # orchestrator.py:70
```

so anything not named `hetero_no_rag` gets RAG, and anything not named
`hetero_no_sentinel` gets the Sentinel. The four arms are actually:

| config | RAG | Sentinel | coder model |
|---|---|---|---|
| `single_agent` | yes | yes | qwen2.5-coder:7b |
| `homo_llama3` | yes | yes | llama3.2 |
| `hetero_no_rag` | **no** | yes | qwen2.5-coder:7b |
| `hetero_no_sentinel` | yes | **no** | llama3.2 |

`single_agent` runs the **full multi-agent pipeline** and differs from
`homo_llama3` only in the coder model. So the grid contains no single-agent
baseline, and **any claim of the form "multi-agent beats single-agent" is not
supported by `data/outcomes/results.jsonl`.** What the data does support is an
ablation of RAG and of the Sentinel, plus a coder-model comparison.

Also note there is no config named `hetero`, even though that is
`orchestrator.run`'s default and the value a bare `python orchestrator.py`
produces. Its rows would be functionally identical to `single_agent`.

### The run_experiments header is wrong

`experiments/collect_humaneval.py:2` claims:

> Runs all 6 configurations x 3 prompting strategies across 30 HumanEval tasks

Actual: **4 configurations x 2 prompting strategies x 15 tasks = 120 runs**,
which matches the 120 rows in `results.jsonl` exactly. If the writeup quotes
30 tasks or 6 configs from this comment, it is quoting the comment rather than
the experiment.

### Consequence

The grid is complete, so `run_experiments.py` resumes into a no-op. Collecting a
fresh trace needs either `MAARA_IGNORE_RESUME=1` or, better,
`MAARA_RESULTS_PATH` pointed at a new file — which gives an empty resume set and
keeps the original results untouched.

### The existing 120 runs are NOT contaminated

On 2026-09-29 a traced run retrieved `simulator/workload.py` as "relevant code" for a
HumanEval bug and overwrote it (see §15). The obvious follow-up question is
whether any *historical* result is affected. It is not. Timeline:

| event | when | source |
|---|---|---|
| `benchmark/results/run_*.json` written | 2026-04-18 15:36–16:02 | file mtimes |
| `data/outcomes/results.jsonl` last written | 2026-05-05 01:30 | file mtime |
| ...and committed | 2026-05-04 / 05-05 | `3215189`, `a01af03` |
| `sim/` first appeared on disk | **2026-09-15 16:12** | `simulator/__init__.py` mtime |
| `sim/` first committed | 2026-09-29 15:16 | `4ba38ac` |
| `tracing.py` created | 2026-09-29 | this session |

The simulator post-dates the last results row by **about four months**, and
`results.jsonl` has not been appended to since 2026-05-05. No run that produced
the 120 rows could have seen simulator code, because none of it existed.

Confirmed structurally as well. Applying orchestrator's `available_files` filter
to the tree at `472f18e` (the last commit before `sim/`) yields:

```
['target.py']
```

Applying the same filter to the tree today yields eleven files, including
`simulator/engine.py`, `tracing.py` and `tests/conftest.py`. That difference is
exactly the regression, and it did not exist when the 120 rows were produced.

**The existing `results.jsonl` and `benchmark/results/*.json` remain usable.**
The contaminated artefacts are only the 2026-09-29 traced run's outputs.

## 20. The 4-config repetition problem — the current traces cannot benchmark a cache

**Open. Blocks workstream 1's validation and workstream 2's workload.**

Both committed traces replay the *same 15 HumanEval tasks* under 4 configs in
one process. An `issue_parser` prompt therefore recurs **byte-identically**
about four times, and every one of those repeats is a cache hit (FINDINGS §19).

Consequence: the traces measure *experiment-harness repetition*, not the
workflow prefix reuse the project is about. Any cache or retention policy
scored on them is rewarded for something an operator would never see. This is
why the Milestone 1 hit fraction is not a usable baseline, and why the fit-set
scores in §15 should not be read as evidence about the mechanism.

Fix is the new trace (ROADMAP, Part 3): disjoint problems, 2 configs, Ollama
restarted between them.

## 21. New trace design — not yet collected

50+ distinct HumanEval problems not among the current 15; 2 configs (full
pipeline with the qwen coder; llama coder for model mixing); original layout;
zero_shot; single process start to finish; Ollama restarted between configs.

Open questions before collecting:
- Does `experiments/collect_humaneval.py` support problems beyond its embedded
  15? Its `TASKS` list is literal and truncated with `TASKS = TASKS[:15]`. If
  it does not, it must be extended first.
- The run must not be resumed mid-way. §14c and §18 are both consequences of
  a resumed or double-started run.

## 22. The `shared_first` trace has an unexplained slow segment

See FINDINGS §18 for the measurements. Segment 0 (86 calls) ran at a median
1.20x the calibrated prefill rate with a p10 of 0.21x, and 30% of its calls
were slower than a cold prefill; segment 1 (128 calls) ran at 1.73x. The two
are disjoint in time, so contention between them is ruled out.

Cause not established — memory pressure and thermal throttling are both
plausible and neither was logged. Consequence: the hit detector is unreliable
inside segment 0, which holds 19 of 21 apparent false positives, and that
trace's "78 measured hits" is an undercount. Capture machine state (`ollama
ps`, memory pressure, thermals) during the next long collection.

## 23. `llama3.2`'s `kv_bytes_per_token` is DERIVED, not measured

qwen2.5-coder:7b's 56 KiB/token was confirmed against the Ollama server log.
llama3.2's 112 KiB/token was scaled from it by architecture arithmetic and has
never been checked the same way. It drives every KV figure for the
llama-coder configs, and GQA makes the asymmetry counter-intuitive (the
smaller model is the memory hog), so it is exactly the constant most likely to
be wrong by a factor.

Fix: one server-log line from a llama3.2 call with a known prompt length.
Cheap, and worth doing during the new collection while the model is loaded.

## 24. The cache model's held-out recall gap is entirely `issue_parser`

**Open. To investigate after workstream 2 step 1.**

Held-out hit recall is 81% (44/54) and 80% (53/66), against 94% and 97% on the
fit traces (FINDINGS §22). The whole ~14-point gap sits in one role:

| role | `holdout_qwen` matched/measured | `holdout_llama` |
|---|---|---|
| coder | 25/25 | 47/49 |
| debugger | 13/13 | 6/6 |
| evaluator | 3/3 | 0/1 |
| **issue_parser** | **3/13** | **0/10** |

`issue_parser` holds all 23 false negatives and all 5 false positives across
both traces. On the fit traces its hits were exact-prompt repeats, which the
frozen rule captures perfectly; with disjoint problems its prompts no longer
recur exactly, so whatever reuse it gets is partial and not modelled.

The roles the workflow actually chains — coder and debugger — are near-perfect,
so this does not block the retention work. It does bound any claim about the
model's accuracy, and the claim should be stated per role until it is resolved.

Cheapest next step: `/tmp/ollama_holdout_A.log` and `_B.log` cover these exact
calls, so the server's own `f_keep` and `looking for better prompt` decisions
are on disk per call. Compare them against the model's decision for the 23
false negatives rather than inferring from timings.

## 25. Scoping prefix reuse cannot be done with a match-time guard

**Open. Blocks workstream 2 step 1a as originally planned.**

The plan was to enforce `reuse_scope` in `Engine._match_blocks`
(`simulator/engine.py:402-406`). That is unsafe, and the reason is structural.

`_match_blocks` returns the longest leading run of keys present in
`Engine.entries`, and `_try_admit` then computes `new_keys = chain[matched:]`
and **unconditionally overwrites** at the insertion site
(`simulator/engine.py:501-508`):

```python
for k in new_keys:
    self.entries[k] = KVEntry(key=k, ...)
    self._entry_bytes += block_bytes
```

Today that is safe because of an invariant: the key that stopped the match is
absent, and since keys are a hash chain, every later key is absent too. A
scope guard breaks the invariant — the match stops at a key that *is* present,
and because two runs of the same problem produce byte-identical prompts, every
subsequent key is present as well.

Measured on `original`: **16 node chains are emitted identically by more than
one task run, the longest being 32 blocks (runs 12 and 42).** Blocking a match
there would overwrite 32 live entries, resetting `refcount` to 0 while other
requests hold them and double-counting `_entry_bytes`. That is refcount and
budget corruption, not a wrong metric.

### The deeper point: the engine is not what is wrong

A content-keyed prefix cache *should* serve two identical prompts from the same
entry — that is what a real server does, and `simulator/ollama_cache.py` is
frozen on exactly that behaviour. The artefact is that the `original` trace
**contains** the same prompt four times, because the harness replayed 15
problems under 4 configs. The fix for that belongs in the workload, and it is
already done: the holdout traces have 0.0% same-task cross-run reuse
(FINDINGS §22).

### Safe implementation, if the scope flag is still wanted as a diagnostic

Salt the block chain at trace-load time by **problem occurrence index** rather
than guarding at match time. For the k-th occurrence of problem P, seed the
chain with k. Then two runs of the same problem get disjoint keys (no
collision, no overwrite), while different problems still share their boilerplate
prefix whenever their occurrence indices match. The engine is untouched, so
refcounting and byte accounting stay correct by construction.

Cost: slightly artificial, in that problem P's 2nd occurrence shares boilerplate
with Q's 2nd but not with Q's 1st. Aggregate sharing rates are preserved.
