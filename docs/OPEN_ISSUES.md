# Open Issues

What is not settled, not validated, or known to be wrong. Read this before trusting a number or starting new work.

> Split out of the single `sim/HANDOFF.md` during the Milestone 2 cleanup.
> **Section numbers are the original HANDOFF numbers and are deliberately not
> renumbered**, so cross-references like "supersedes §3 decision 4" still
> resolve. They are not sequential within a file. The other parts live in
> `ARCHITECTURE.md`, `DECISIONS.md`, `FINDINGS.md` and `OPEN_ISSUES.md`.

## Status index (2026-10-07)

| § | issue | status | next action |
|---|---|---|---|
| 6 | Engine invariants | **closed**, kept as a record | none: the suite enforces them |
| 9.1, 9.2, 9.3, 9.7 | Prefill, rent model, decode coefficient, arrival rate | **resolved** | none |
| 9.4 | Rates are Mac/Metal | **open**, a standing scope statement | the GPU profile, once GPU time is booked (GPU_TODO) |
| 9.5 | `model_switch_seconds = 0.0` | **open**, low impact | check `ollama ps` during the `final` collection |
| 9.6 | Invented `REPRESENTATIVE_ISSUE` | **not needed** for trace-driven results | none |
| 11 | Old "next steps" list | **closed** | none |
| 14 | MAARA writeup caveats | **open**, re-verified 2026-10-07 | state them in the writeup; fix the stale header comment |
| 15 | Ollama cache model | **resolved** | none (the remainder is §24) |
| 20 | 4-config repetition | **resolved** | none |
| 21 | New trace design | **resolved**; the `final` collection is **running** (started 19:50 on 2026-10-07, config A) | let it finish; run nothing heavy alongside it |
| 22 | `shared_first` slow segment | **not needed**: that trace no longer backs any claim | done: the `final` run logs machine state (`data/traces/final_machine_state.log`) |
| 23 | llama3.2 KV bytes DERIVED | **resolved** (`6f4ce7c`, DECISIONS §30) | none |
| 24 | `issue_parser` recall gap | **open**; the holdout logs are lost, but the `final` server log is being captured to `~/ollama_final_A.log` | copy that log somewhere durable, then compare per call |
| 25 | Scoping reuse with a match-time guard | **not needed**: the holdout traces have 0% cross-run reuse | none |
| 26 | "Ski-rental never fires" | **superseded** by §27 and FINDINGS §24–25 | none |
| 27 | No operating point where ski-rental beats keep-all | **open, highest impact** | run the retention grid at 8x/16x context inside the live band |

**Still open and relevant: 6** — 9.4, 9.5, 14, 24, 27, and the `final`
collection noted in 21.

---

## 6. Validation status

> **Closed; kept as a record.** These invariants (no KV or refcount leak,
> budget never exceeded) are now asserted throughout `tests/simulator/`.

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

1. **RESOLVED in Phase 5b (commit `b593cf5`):** chunked prefill (512 tokens,
   PLACEHOLDER) interleaved with decode is now the default; `--prefill-chunk 0`
   restores the old behaviour. Original text kept below.
   **Prefill is unchunked and unbatched.** One prefill per iteration, so total
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

   > **Swept 2026-10-07; still a placeholder, but the conclusions do not rest
   > on it.** `experiments/sweep_decode_coeff.py`, holdout_qwen 1x, util 0.8,
   > rate recalibrated per coefficient, 10 paired seeds, manifest-checked;
   > output `results/decode_coeff_sweep.json`.
   >
   > | coeff | keep-all recompute / makespan | ski-rental − keep-all recompute | ski − keep p95 | sched choice | lsf − fcfs p95 |
   > |---|---|---|---|---|---|
   > | 0.05 | 0.4% | +7.6s [+3.4, +11.8] | −1.8s [−6.1, +2.5] | 0.2% | −1.4s [−3.7, +0.9] |
   > | 0.15 | 0.4% | +11.2s [+5.6, +16.7] | +2.2s [−0.7, +5.0] | 0.3% | −1.0s [−2.0, +0.1] |
   > | 0.30 | 0.6% | +10.4s [+2.5, +18.4] | +7.9s [−4.1, +19.8] | 0.7% | −4.3s [−9.2, +0.7] |
   > | 0.50 | 0.7% | +11.7s [+4.7, +18.6] | +7.8s [−1.5, +17.1] | 1.0% | −2.3s [−7.1, +2.6] |
   >
   > Across a 10x range: recompute stays under 1% of makespan; ski-rental is
   > significantly *worse* than keep-all on GPU recompute at every value and
   > never significantly different on p95; the scheduler has a choice on at
   > most 1% of decisions and LSF is never significantly different from FCFS.
   > One trend to watch: LSF's point estimate is negative at every value and
   > the 0.15 interval only just spans zero, so a costlier decode leans
   > towards LSF without reaching significance here.
   >
   > Absolute values differ from FINDINGS §24 at 0.15 (keep-all recompute
   > 7.4s here against 13.4s there, p95 148.9s against 187.6s). This sweep
   > averages 10 seeds; how many seeds §24 used is unclear in the records.
   > Do not mix the two tables.
   > **Explained 2026-10-07:** §24 is seed 1 alone; the current code at 0.15,
   > seed 1, reproduces it exactly. Seed spread is large (keep-all recompute
   > 0.4s to 16.3s over seeds 1 to 3). §24 now carries a 20-seed table at the
   > measured coefficient.
   >
   > A **measurement** of the coefficient is still wanted (a probe was in
   > progress in `probes/` on 2026-10-07); the sweep only shows the
   > conclusions are insensitive across 0.05 to 0.50.
   >
   > **RESOLVED 2026-10-07: measured at 0.841 on the Mac** (DECISIONS §28),
   > outside the swept range. Rerun at 0.841, same driver and seeds, output
   > `results/decode_coeff_sweep_measured.json`:
   >
   > | coeff | keep-all recompute / makespan | ski-rental − keep-all recompute | ski − keep p95 | sched choice | lsf − fcfs p95 |
   > |---|---|---|---|---|---|
   > | 0.841 | 0.6% | +9.1s [+0.3, +17.9] | −1.9s [−13.7, +9.9] | 1.6% | −4.5s [−11.2, +2.2] |
   >
   > Every conclusion above holds at the measured value. What moves is
   > capacity: the util-0.8 rate falls to 0.0265/s and keep-all p95 rises to
   > 278s.
4. Rates are Mac/Metal, not CUDA. **Open; now explicit**: every number is
   tagged with the `mac_ollama_measured` profile (DECISIONS §27), and the GPU
   profile waits on booked GPU time (`docs/GPU_TODO.md`).
5. `model_switch_seconds = 0.0` assumes both models stay resident. **Open, low
   impact; early evidence supports it.** The `final` run's machine-state log
   (19:50 on 2026-10-07) shows qwen2.5-coder, llama3.2 and nomic-embed all
   resident at 100% GPU. One snapshot, so confirm across the whole run.
6. `REPRESENTATIVE_ISSUE` in `measure_context.py` is invented, not real
   `parse_issue` output. Small share of the 480 but not zero. **Not needed
   (2026-10-07):** the constant lives on in `probes/calibrate.py` and feeds
   only the synthetic workload's `stable_prefix_tokens`. Trace-driven results
   (`holdout_*`) use the recorded prompts.
7. **RESOLVED (commit `b593cf5`):** the drivers binary-search the arrival rate
   for a target GPU utilisation, and since DECISIONS §25 the rate must be
   calibrated at the configuration the run uses. Original text kept below.
   Benchmark runs above are heavily oversaturated (p50 ~12,000s). Fine for
   demonstrating contention, wrong for measuring policy performance. Arrival rate
   needs tuning per config for ~70–90% utilisation.

---


## 15. RESOLVED — Ollama prompt-cache model (was: UNVALIDATED)

> **Resolved 2026-10-05.** The model below was replaced by the v2 rule read
> from llama.cpp's source (`f_keep < 0.5`, DECISIONS §19, FINDINGS §20) and
> validated on held-out traces with no tuning: hit recall 81% (44/54) and 80%
> (53/66) (FINDINGS §22). The remaining gap is entirely `issue_parser`, tracked
> in §24. Everything below is kept for the record.

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

> **Open; re-verified 2026-10-07.** All three still hold in the code:
> `maara/orchestrator.py:83,99` still gates by exclusion, and the header of
> `experiments/collect_humaneval.py` still says 6 configs x 3 strategies x 30
> tasks. FINDINGS §5's root-only `os.listdir` is still in
> `maara/agents/navigator.py`. These are writeup caveats: changing MAARA
> would break comparability with the collected traces.

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

## 20. RESOLVED — The 4-config repetition problem — the current traces cannot benchmark a cache

> **Resolved 2026-10-05** by the held-out traces (`holdout_qwen`,
> `holdout_llama`): disjoint problems, one config per process, Ollama
> restarted between. The hit rate fell from 34.9% to 16.9% / 17.8% and the
> exact-repeat share from 46% to 7% / 2% (FINDINGS §22). The fit traces still
> carry the artefact and must not be used for cache or retention claims.

~~**Open. Blocks workstream 1's validation and workstream 2's workload.**~~

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

## 21. RESOLVED — New trace design (was: not yet collected)

> **Collected 2026-10-04/05** as `holdout_A_qwen2` (320 calls) and
> `holdout_B_llama` (370 calls), 54 tasks each, both passing
> `check_trace_clean` (ROADMAP, commit `0c8fe40`). The `TASKS[:15]` question
> was answered by moving tasks to `data/tasks/` (DECISIONS §20). A further
> sealed set of 92 tasks, `MAARA_TASK_SET=final`, was built 2026-10-07
> (commit `7d4a96c`) and has not been collected yet.

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

> **Not needed (2026-10-07).** `shared_first` is a fit trace carrying the
> replay artefact (§20) and backs no current claim, so its cause is no longer
> worth chasing. The one action that survives: log machine state during the
> `final` collection.

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

> **RESOLVED 2026-10-07** (`6f4ce7c`, DECISIONS §30): promoted to MEASURED,
> value unchanged, Mac profile only. Evidence follows. The rotated
> Ollama log `~/.ollama/logs/server-3.log` records llama3.2 loads, with
> `general.name = Llama 3.2 3B Instruct`, so it is the 3B, not the 1B. They
> log `llama_kv_cache: size = 448.00 MiB (4096 cells, 28 layers)` and
> `896.00 MiB (8192 cells)`, both f16, i.e. **114,688 B/token**, exactly the
> derived value. qwen2.5-coder in the same log: 224 MiB per 4096 cells =
> 57,344 B/token, as already confirmed. The prompt-cache size estimate agrees
> at about 114,770 B/token.

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

> **2026-10-07: those two logs are gone** (`/tmp` was cleared). They may
> survive in Ollama's rotated `~/.ollama/logs/server-*.log`; not yet checked.
> Otherwise redo this on the server log of the `final` collection, and copy
> that log into the repo or a durable path before analysing it.
>
> **Checked 2026-10-07: not recoverable.** The newest surviving log with
> slot decisions, `server-3.log`, ends at 22:37 on 2026-10-04. `holdout_A`
> started at 22:40 and `holdout_B` ran 23:36 to 00:05, and the logs that
> covered them have rotated away. This now waits on the `final` collection.

## 25. Scoping prefix reuse cannot be done with a match-time guard

> **Not needed (2026-10-07).** The problem it guarded against is gone from
> the workload: the holdout traces have 0.0% same-task cross-run reuse
> (FINDINGS §22), and no scope flag was built. Keep the salting design below
> in case a replayed trace is ever used again.

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

## 26. PARTLY RETRACTED — "the ski-rental release rule never fires"

> **Superseded (2026-10-07)** by §27 and FINDINGS §24–25. Ski-rental was run
> inside the live band (batch 4, KV 512MB), where it fires and still loses to
> keep-all. Its point 2 (predictions barely exercised on real traces) is
> carried by §27.

> **Two overstatements, both mine.** (a) The zero-releases measurement was taken
> at util 0.6; at util 0.8 the rule does fire (4-1315 releases). (b) I then
> concluded no KV budget could make rent live, having swept the wrong knob —
> the lever is **max_batch**, not KV. See §27 for the band that does exist.
> What survives: the rule is inert at batch >= 8, which is where every earlier
> comparison was run.


**Open. This is the biggest threat to mechanism 1's claim and should be the
next thing addressed.**

Instrumenting `SkiRentalPredictionsRetention.on_tick` over one `holdout_qwen`
replay at 1x, util 60%, KV 1024MB, batch 8:

| | |
|---|---|
| entry-ticks considered | 24,185,847 |
| of those, carrying a prediction | 4,850,553 (20.1%) |
| hinted entries with `rent_rate == 0` | **4,850,553 (100%)** |
| **voluntary releases made** | **0** |

So the voluntary "hold until accrued rent reaches the recompute cost, then let
go" rule — the actual ski-rental decision — **never fired once**. Every
difference between `ski-rental`, `ski-rental-pred` and `lru` in the grid comes
from the forced-eviction ordering they inherit, not from ski-rental at all.
That is why `ski-rental` and `ski-rental-pred` print byte-identical rows.

### Why

Rent only accrues under eviction pressure (DECISIONS §13). At this operating
point there is almost none, so `rent_rate` is 0 for every entry that has a
prediction. `_predicted_rent = hint_gap_s * rent_rate` is then 0, which reads
as "predicted near", so the threshold becomes `b / lambda` and nothing is ever
released. With no rent accruing, plain ski-rental's threshold `b` is never
reached either.

### What this means

This is the same failure mode as "nothing ever blocked on KV" (DECISIONS §8)
and "the scheduler has a choice on under 9% of decisions" (FINDINGS §23): the
engine is not in the regime where the mechanism can act, so the comparison
measures nothing. It is not a bug in the policy.

Two things follow, and the second matters more:

1. The grid must run where rent is live. A KV sweep to locate that band was in
   progress when this was written; the 1x/1024MB cell is definitively outside
   it.
2. **The learning-augmented half of mechanism 1 has never been exercised on
   real trace data.** Hints were absent entirely until the per-block fix
   (DECISIONS §21 follow-up), and now that they exist at 20% coverage they
   still reach only entries whose rent rate is zero. Any claim about
   ski-rental-with-predictions currently rests on synthetic runs only.

### Note on a related fix that was necessary but not sufficient

`HintBuilder.hint_for` read only node-level `next_reuse_gap_s`, which is None
for **320 of 320** nodes on `holdout_qwen` — so the hint was `None` for every
block and the predictions policy degenerated to the deterministic one for a
second, independent reason. It now prefers the per-block gap, lifting hint
coverage from 0% to 26.2% of blocks. Necessary, but as the table above shows,
not enough to make the rule fire.

## 27. The live-rent band is set by concurrency, not by KV budget

> **Open, highest impact (2026-10-07).** At 1x context the negative result is
> now established over 20 seeds (FINDINGS §24–25). What has never been run on
> real traces is the regime FINDINGS §7 says retention needs: stable context
> above ~2,000–4,000 tokens. The only saved grid is 1x, util 0.6, batch 8,
> which is outside the band. Next: `run_retention_grid` at 8x and 16x, batch
> 2–4, at the measured coefficient.

Joint sweep, `holdout_qwen` 1x, util 0.8, arrival rate fixed, FCFS:

| batch | retained KV | rent-active ticks | vol releases | ski-rental/keep-all | pred/keep-all |
|---|---|---|---|---|---|
| 2 | 96.6-99.1% | **50-59%** | 578-1315 | 1.107-1.408 | 1.117-1.417 |
| 4 | 95.1-98.6% | **14-16%** | 16-985 | 1.196-1.380 | **1.169**-1.264 |
| 8 | 93.8-98.3% | 2.6-3.5% | 0-420 | 1.204-1.375 | same as ski-rental |
| 16 | 93.7-98.3% | 0.9-1.0% | 0-56 | 1.218-1.365 | same |
| 32 | identical to 16 | identical | identical | identical | same |

Three results:

1. **KV budget is nearly irrelevant to whether retention is active.** Across
   512/1024/2048 MB the rent-active fraction barely moves; `max_batch` moves it
   by two orders of magnitude. Tightening KV actually *lowers* rent-active
   ticks, because the budget then goes to pinned ACTIVE requests and almost
   nothing is evictable. So the planned `--kv-mb` auto-calibration had no
   target to bisect toward; the knob is concurrency.
2. **Retained KV fraction does not discriminate** — it is 93-99% everywhere, so
   it is useless as a calibration target. `rent_active_tick_frac` is the one
   that tracks whether the mechanism can act.
3. **Batch 16 and 32 are byte-identical**, so effective concurrency on this
   trace caps out below 16: the queue never holds 16 ready requests. Every
   earlier retention comparison ran at batch 8 or 16, i.e. outside the band.

### The uncomfortable part

Even inside the band, **ski-rental never beats keep-all** — the ratio runs
1.107 to 1.408, always above 1. One cell is suggestive: at batch 4 / 512MB the
learning-augmented variant scores 1.169 against plain ski-rental's 1.380, which
is the first place predictions have measurably helped on real trace data. One
cell is not a result.

So mechanism 1 currently has no operating point on this workload where it beats
the incumbent. That is the finding to either overturn with the full grid or
report as negative.

### What this means for calibration

Calibrate on `rent_active_tick_frac` by varying `max_batch`, not `--kv-mb`. A
target band of roughly 15-50% puts the run between batch 4 and batch 2. Note
that batch 2-4 is a small-deployment regime — plausible for the single-GPU
local server MAARA actually runs on, but it must be stated rather than
presented as a general setting.
