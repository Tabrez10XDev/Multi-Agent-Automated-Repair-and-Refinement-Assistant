# Findings

What the measurements actually showed, including the results that went against the hypothesis. Corrections are kept in place rather than folded in, so the record shows what was believed and when.

> Split out of the single `sim/HANDOFF.md` during the Milestone 2 cleanup.
> **Section numbers are the original HANDOFF numbers and are deliberately not
> renumbered**, so cross-references like "supersedes §3 decision 4" still
> resolve. They are not sequential within a file. The other parts live in
> `ARCHITECTURE.md`, `DECISIONS.md`, `FINDINGS.md` and `OPEN_ISSUES.md`.

## Status index

Read this before quoting anything below. Retracted and superseded sections are
kept in place on purpose; this table says which numbers are still live.

**Every result here is a Mac + Ollama + Metal result** (the
`mac_ollama_measured` profile, DECISIONS §27). The simulator sections (marked
*sim*) were run with `batch_decode_coeff = 0.15`, a placeholder. The measured
value is **0.841** (DECISIONS §28). The conclusions still hold at 0.841
(OPEN_ISSUES §9.3), but absolute p95 roughly doubles, so quote their absolute
latencies as 0.15-era numbers. The other sections are direct trace or probe
measurements and are unaffected.

| § | status | current result |
|---|---|---|
| 4. Measurements | current | qwen2.5-coder:7b prefill 267.9 tok/s, decode 28.09 tok/s. This repo's stable prefix is 480 tokens. The 15.6s RAG build, not pytest, dominates the CPU gap between retries. |
| 5. Findings that affect MAARA | current | `get_test_file_contents` finds 0 test files on requests and flask (root-only `listdir`). Realistic stable prefixes are ~700–27,000 tokens, against 480 here. |
| 7. The central result | current, *sim*; latency claim qualified by §24 | Retention's rent/recompute crossover is at ~2,000–4,000 tokens of stable context, so it is dead at 480. This is a cost result, not a latency one: at 1x §24 finds recompute is 0.8–1.1% of makespan. The RTX 4090 and A100 rows are simulated configs beyond this repo's 16 GB ceiling (DECISIONS §29). |
| 14a. Reuse-figure corrections | current | Model-aware matching gives a reusable fraction of 13.7% (`original`) and 12.0% (`shared_first`). Reuse figures quoted before 2026-09-30 are wrong. |
| 14b. Shared-first layout | current (negative result) | Reordering cut reuse from 13.7% to 12.0% and saved prefill from 49.6s to 35.2s; coder→coder reuse fell from 7,626 to 703 tokens. Rule: most-stable content first. |
| 14c. Run-ID resume hazard | current | A resumed run under the same `MAARA_RUN_ID` appends a second `seq` sequence. Trace files now refuse to reopen without `MAARA_TRACE_APPEND=1`. |
| 14d. Ollama does prefix caching | partly superseded by §19, §20, §22 | Still true: Ollama caches prefixes (an identical repeat is 335x faster) and `prompt_eval_count` cannot see it. Superseded: the 32.7%/36.4% hit rates (§22: about 1 in 6 on disjoint problems) and the parallel-slot account of the hits (§19–20: host prompt-cache restores). |
| 17. 0.10 threshold | **retracted**, superseded by §19 | Overlap at or above 0.10 does not destroy the slot: A survives intact at 25% and 50% overlap (§19). |
| 18. Degraded `shared_first` segment | current | The trace has two disjoint segments, not concurrent ones. Segment 0 ran at a median 1.20x calibration (p10 0.21x) and holds 19 of the cache model's 21 false positives. Cause unknown. |
| 19. Exact prompt recurrence | data current; mechanism superseded by §20; hit-rate headline superseded by §22 | Every exact-prompt repeat is a measured hit (46/46, 47/47). Partial prefix hits run at 2.4–4.9x calibration against 14–15x for exact repeats. |
| 20. Cache rule from llama.cpp source | current | Save and cache search fire when `f_keep < 0.5` (literal in the source). 0.10 only chooses between slots and is inert at `-np 1`. Byte budget 8192 MiB, oldest-first eviction. |
| 21. Retry-loop state discard | current | With a llama coder, 19.5% of within-task same-model handoffs discard reusable state, 92% of them coder→coder retries. With a qwen coder it is 1.4%. That retry rate is the cause is unproven: it is confounded with the model. |
| 22. Held-out validation | current; hit rate refined by §28 | About **1 call in 6** hits Ollama's cache on the 54 holdout problems (16.9%, 17.8%); over all unseen traces including `final` it is about 1 in 8 (§28), almost all of it partial prefix reuse. Hit recall held out is 80–81% against 94–97% on the fit traces, and all of the shortfall is in `issue_parser`. |
| 23. LSF buys tail latency | **retracted**, superseded by §27 | Measured at ~2.2x the intended load. The 14.8% p95 win and the "oracle is worse" result are void. |
| 24. Retention cannot move latency | current, *sim* | Recompute is **under 1% of makespan** (keep-all 0.7% over 20 paired seeds at 0.841), so retention is a GPU-work mechanism, not a latency one. No deployable policy differs from keep-all on p95. Evict-now is significantly worse on both p95 (+36.9s) and recompute (+111.9s). Use the 20-seed table at the head of §24. The older table below it (0.8–1.1%, keep-all 13.4s) is **seed 1 only** at 0.15. The 0.6% in DECISIONS §28 is the separate 10-seed OPEN_ISSUES §9.3 sweep. |
| 25. Ski-rental negative result | current, *sim*; numbers are seed 1 only | Ski-rental holds 4.5 MB less than keep-all yet pays 20.2s more rent and 5.0s more recompute. Releasing shifts rent onto the remaining holders rather than removing it. The figures are the same single run as §24's older table. Over 20 seeds at 0.841, ski-rental recomputes +10.2s [+4.8, +15.6] more than keep-all, ties it on p95, and the conclusion holds (§24). |
| 26. Belady loses to LRU | current, *sim*; numbers are seed 1 only | Belady recomputes 29.4s against LRU's 13.4s, because eviction here is bulk and block sizes differ. `offline-belady` is not a valid minimum-recompute reference. Same single run as §24's older table, and not yet rerun over multiple seeds. |
| 27. Scheduler table, corrected | current, *sim*; holds at its point, bounded by §29 | **No scheduling rule differs significantly** from arrival order on p95, at batch 8 or 16, nor does the oracle (paired CIs, 20 seeds). The scheduler has a choice on only 0.3–0.4% of admissions. Still tied at 0.841 (DECISIONS §28). |
| 28. Sealed `final` set | current | On 92 unseen problems pass@1 is 75.0% (qwen coder) against 48.9% (llama coder). Over all 146 unseen problems it is 78.1% against 54.1%, a paired gap of +24.0 points [+15.0, +33.0]; this is a coder-model comparison. The frozen cache rule recalls 77% and 97% of hits, 86% pooled over every unseen trace. Hit rate over the unseen traces is about 1 call in 8 (12.8%). The policy drivers are not yet run on it (sealed, DECISIONS §26). |
| 29. Scheduling regime map | current, *sim*, 0.841 | Scheduling matters once requests queue: at batch 2-4, LSF and EDF cut p95 by 15-22% at §27's load and separate at the lowest load tested. At batch 8 they separate from load 1.10, and at batch 16 (KV-bound) from 1.5-1.75. The gain is task order: EDF here is oldest-task-first and matches LSF in 39 of 40 cells. The oracle estimator is slightly worse in overload. Miss rates do not improve (+1 to +5 points). |
| 30. Per-task deadlines | current, *sim*, 0.841 | With heterogeneous deadlines (3 classes, size-scaled), LSF, EDF and Dynamo-style client priority never beat oldest-task-first (`program-fcfs`) on p95, in any of 40 cells. They are worse in about 70% of cells, and they only move misses from the tight class to the loose one. The supported scheduling claim is program-level ordering. |
| 31. What-if: efficient batching | *sim*, what-if 0.15, not measured | HYPOTHETICAL coefficient 0.15. LSF still never beats `program-fcfs` on p95 (0 of 40 cells; worse in 22). Its tight-class gains (up to 14 points at batch 2) are paid for by the loose class. At batch 8–16 the scheduler rarely has a choice (cap-binding labels qualified by OPEN_ISSUES §28). Verdict: freeze `program-fcfs`. Deadlines were recalibrated to a fixed FCFS miss rate. |
| 32. Sealed `final_qwen`, partial | current, *sim*, 0.841, **out of sample**, 5 of 40 cells | On 91 unseen tasks at batch 2, `program-fcfs` cuts p95 by 14-33% against FCFS in all 5 cells (loads 0.71-1.15), and LSF is significantly worse than `program-fcfs` in all 5 (+18 to +57 s). Pre-registered S1 and S2 hold; S3 untestable at batch 2. Batches 4-16 not yet run. |

---

## 4. Measurements

All from `probes/calibrate.py` against live ollama on the user's Mac (Metal).

### Model rates — these overturned the user's stated constants

| | user's value | measured | note |
|---|---|---|---|
| qwen2.5-coder:7b prefill | 123 tok/s | **267.9** | user's figure ~2.2x low |
| qwen2.5-coder:7b decode | 29 tok/s | **28.09** | user's figure correct |
| llama3.2 prefill | never measured | **541.87** | |
| llama3.2 decode | never measured | **59.9** | |

Prefill measured at three lengths to separate throughput from fixed per-call
overhead. qwen: 292.8 tok/s @563, 289.6 @2213, 269.6 @7713 — flat, so it is a
real rate. Reported value is the marginal slope.

**Trap:** ollama's default `num_ctx` is 2048 and it **silently truncates** longer
prompts. A 6000-token prompt reports ~2050 tokens and every derived rate is
nonsense. `measure_context.py` now sets `num_ctx=8192` explicitly. Anyone
benchmarking ollama in this project must do the same.

### Prompt sizes

Measured against this repo: **stable prefix = 480 tokens** (1923 chars). The
user had assumed 2000. Composition is ~120 tokens of boilerplate plus the entire
contents of `test_target.py` (1301 bytes).

The stable/variable split is derived from `coder.py:113-131` and
`orchestrator.py:160-161`. Prompt order is: issue fields → test_context →
`Current code in {filename}` → retry_context. Because `relevant_code[original_file]`
is rewritten after every failed attempt, **the durable shared prefix ends
immediately before the source block**, even though the source sits mid-prompt.

### CPU stages

- `pytest`: **0.243s**
- RAG index build: **15.599s** — the RAG build, not pytest, dominates the
  non-GPU gap between retry attempts.

### KV bytes per token — DERIVED, NOT MEASURED

```
qwen2.5-coder:7b  2 × 28 layers × 4 KV heads × 128 dim × 2 bytes =  57,344 B (56 KiB)
llama3.2:3b       2 × 28 layers × 8 KV heads × 128 dim × 2 bytes = 114,688 B (112 KiB)
```

From published architecture configs. **Verify against GGUF metadata before
quoting.** Also confirm `llama3.2:latest` is the 3B, not the 1B.

The 3B model costs **twice as much KV per token** as the 7B because
qwen2.5-coder uses a more aggressive GQA ratio. This is not a typo and it
matters: the Sentinel (`evaluate_plan`, llama3.2, ~2400-token prompts) reserves
~262MB per request and is the biggest memory consumer in the pipeline.

### Workload calibration from real runs

`data/outcomes/results.jsonl` (121 runs) gives the retry distribution used by the
trace generator: attempts `{1: 74, 2: 13, 3: 33}`, p50 latency 24.5s, pass rate
0.77. The replan rejection probability is still a placeholder —
`evaluator_approved_on_attempt` is null for the `single_agent` and
`hetero_no_sentinel` arms, so it is not cleanly recoverable.

---


## 5. Findings that affect MAARA itself

### `get_test_file_contents` finds nothing on real repositories

`agents/navigator.py:42-52` uses `os.listdir(repo_path)` — repo root only, no
recursion. Verified against two repos SWE-bench actually uses:

| repo | test files in root | test files total |
|---|---|---|
| psf/requests | **0** | 9 (166 KB) |
| pallets/flask | **0** | 27 (216 KB) |

On either repo the function returns `{}`, so the `test_context` block of the
coder prompt is **empty** and MAARA attempts fixes without seeing the tests it
must satisfy. It works today only because `test_target.py` happens to sit in this
repo's root. **This is a real harness bug, independent of the simulator.**

### Inlining all test files does not scale

MAARA's current policy is to inline every test file it finds. On requests that
would be 40,825 tokens; on flask, 54,176. Both exceed practical context limits.
Relevant-test selection logic does not exist and would be needed for real repos.

### Realistic context sizes

Measured `chars_per_token = 4.077` on real test code. Per-file stable prefix:

| file | tokens |
|---|---|
| requests `tests/test_requests.py` | 26,597 |
| requests `tests/test_utils.py` | 7,710 |
| requests `tests/test_lowlevel.py` | 3,763 |
| requests `tests/test_structures.py` | 676 |
| flask `tests/test_basic.py` | 13,708 |

Realistic range is **~700–27,000 tokens**, against 480 in this repo. Note it is a
**distribution, not a constant** — which module is under repair determines the
context size. The trace generator currently uses a single value and should sample
from this distribution instead.

---


## 7. The central result

**The mechanism is dead at this repo's scale and alive at realistic scale.**

Context-size sweep (KV 2048MB, B=8):

| stable context | rent | recompute |
|---|---|---|
| 480 (this repo, measured) | 0.2s | 0.0s |
| 1000 | 4.5s | 0.0s |
| 2000 | 16.3s | 2.3s |
| 4000 | 17.6s | 88.4s |
| 8000 | 41.3s | 352.4s |

Crossover is ~2000–4000 tokens. This repo sits at 480; real repos sit at
3,700–27,000.

The sharper diagnostic is **what fraction of blocking retention could actually
fix**:

| regime | blocked iterations | retention could fix | rent | recompute |
|---|---|---|---|---|
| toy (ctx 4000, KV 1536MB, B=8) | 1557 | **2 (0.1%)** | 0.1s | 0.7s |
| realistic (ctx 7710, KV 40GB, B=64, 300 tasks) | 86 | **86 (100%)** | 2233s | 4510s |

At toy scale the budget is consumed by in-flight requests and retention has no
lever. At realistic scale every blocking event is retention-addressable and rent
and recompute are the same order of magnitude — the regime where ski-rental's
decision is non-trivial.

### Contention survives on modest hardware

ctx 7710, 300 tasks, rate 1.0:

| hardware | KV / B | blocked | retention could fix | rent | recompute |
|---|---|---|---|---|---|
| RTX 4090 fp16 | 7GB / 16 | 143 | 125 (87%) | 2022s | 5752s |
| RTX 4090 Q4 | 17GB / 32 | 114 | 114 (100%) | 2128s | 5391s |
| A100 40GB | 23GB / 48 | 109 | 109 (100%) | 2492s | 5007s |
| A100 80GB | 40GB / 64 | 86 | 86 (100%) | 2233s | 4510s |

Smaller GPUs show *more* blocking. 40GB is not required. Suggested headline
config is **RTX 4090 Q4 (17GB KV, B=32)** — nameable hardware, 100%
retention-addressable blocking.

---


## 14a. Corrections to the Phase 2 and Phase 4 reuse figures

Two errors were found after those phases were written. Numbers quoted from
them before 2026-09-30 are wrong.

**Prefix matching ignored the model.** A KV cache belongs to one model
instance, so a prefix qwen computed cannot serve a llama3.2 call however
identical the text. `compute_prefixes` compared across models and counted
those as reuse. Now model-aware, with the model in the cache key and a guard
in the engine. Effect on the original layout is small — reusable fraction
13.8% -> 13.7%, saved prefill 50.1s -> 49.6s — because almost all of its reuse
was same-role and therefore already same-model. Effect on the shared-first
layout is large: 20.2% -> 12.0%. `same_model_only=False` reproduces the old
figures.

**Segmentation merged two processes sharing a run_id.** `MAARA_RUN_ID` pins
the trace filename, so a run resumed under the same id appends a second `seq`
sequence starting again at 1. Sorting by `seq` interleaved them and fabricated
a task-run boundary on nearly every row — the shared-first trace segmented
into 305 blocks instead of 60. Now ordered by `t_start` with a new process
group wherever `seq` fails to advance.


## 14b. The shared-first prompt layout experiment (negative result)

`MAARA_PROMPT_LAYOUT=shared_first` reorders every prompt to put one common
system message and identically-rendered shared blocks first. It was built to
recover the 24% of prompt tokens that are repeated but unreachable by a prefix
cache. **It made things worse and should not be retried as designed.**

Reusable fraction (same-model matching) fell 13.7% -> 12.0%; saved prefill
49.6s -> 35.2s. Not retried.

### Order by volatility, not by role coverage

The blocks were ordered by how many roles carry each one, putting the code
block first because three roles have it. But the code is the one block that
**changes on every attempt**: `apply_and_test` writes the fix and
`orchestrator.py:160-161` re-reads it into `relevant_code`. Consecutive coder
calls therefore diverged 254 characters in, inside the very first block.

The result, by source -> target pair, in tokens:

| pair | original | shared_first |
|---|---|---|
| coder -> coder | **7,626** | 703 |
| debugger -> debugger | 3,466 | 3,402 |
| evaluator -> coder | — | 2,187 |
| issue_parser -> coder | — | 1,067 |

The new layout did create real cross-role sharing, in the arms where those
roles happen to run the same model. It just destroyed something 3.5x larger.
Reuse in this workload is overwhelmingly **same-role across retries**, not
cross-role, and the original layout (issue -> tests -> code -> retry context)
was already ordered stable-to-volatile by accident.

**The rule: most-stable content first, most-volatile last.** Role coverage is
the wrong sort key. A correct retry would be issue -> tests -> code ->
role-specific, keeping the common system message.

### It also broke the evaluator on 7 of 60 runs

Seven shared-first task runs died with `list index out of range` before any fix
attempt, against zero in the original. The cause is `agents/evaluator.py:66`:

```python
score_line = [l for l in raw.split("\n") if l.startswith("SCORE:")]
score_raw = score_line[0].split(":")[-1].strip()      # IndexError if absent
```

Moving the output-format instructions from the opening of the prompt to the end
made llama3.2 less reliably emit the `SCORE:` line. Two findings, not one: the
layout degrades instruction-following for the smaller model, **and**
`evaluate_plan` has no guard for a malformed response. The parser should
default rather than crash whichever layout is used.

Pass rate 50/60 -> 47/60. With n=15 per config that difference is **not
distinguishable from noise**; the 7 crashes are, because they are mechanical.


## 14c. Run-ID resume hazard

`MAARA_RUN_ID` pins the trace filename. A run interrupted and restarted under
the same id **appends a second `seq` sequence starting again at 1**, into the
same file, with the same `run_id`. Nothing in the data marks the boundary
except that `seq` stops advancing.

This happened to the shared-first run: 340 rows, 156 from one process and 184
from another, split by a 204-second gap. Segmenting by `seq` interleaved them
and produced 305 task-run blocks instead of 60, which invalidated every reuse
figure computed from that trace until it was found.

Two mitigations are in place:

* `segment_task_runs` orders by `t_start` and starts a new process group
  wherever `seq` fails to advance, so an already-collected trace is still
  usable.
* `tracing.py` now **refuses to open a trace file that already exists** unless
  `MAARA_TRACE_APPEND=1` is set, so the situation cannot be created silently.


## 14d. CORRECTION: Ollama does prefix caching, and the count hides it

**The "zero Ollama cache hits" reported in Phase 2 and Phase 4 was wrong.**
It came from a detector built on `prompt_eval_count`, assuming Ollama reports
only the tokens it recomputed. It does not. Measured directly with
`sim/check_ollama_cache` on 2026-09-30, `/api/generate`:

| call | prefill | vs cold |
|---|---|---|
| cold | 13.400s (281 tok/s) | — |
| identical repeat | **0.040s** | 335x faster |
| last line changed | **0.074s** | 181x faster |

`prompt_eval_count` was **constant throughout**. The cache is real and the
count is blind to it.

`detect_cache_hits` is rewritten around prefill DURATION against the
calibrated uncached rate: a call is a hit when its rate exceeds 1.5x the
model's uncached rate, and hidden tokens are
`prompt_tokens - prefill_seconds x uncached_rate`. Reclassified:

| trace | hits | hidden tokens | prefill saved |
|---|---|---|---|
| original | **89 / 272 (32.7%)** | 20,402 | **60.8s** |
| shared_first | **78 / 214 (36.4%)** | 15,831 | **49.5s** |

> **Superseded hit rates.** These include the harness replaying tasks under
> four configs. On disjoint problems the standing figure is about **1 call in
> 6**, almost entirely partial prefix reuse (§22).

By role, hits concentrate in `issue_parser` (45 of 89), which also explains the
2881 tok/s anomaly the earlier analysis could not: consecutive `issue_parser`
calls share a long fixed preamble and were hitting the cache. coder and
debugger sat at ~300 tok/s, the uncached rate.

### The single-slot hypothesis is NOT supported

Tested by recording, for every call, the previous call on the same model and
their shared prefix:

| | original | shared_first |
|---|---|---|
| prev same-model call shares >200 chars AND hit | 41 | 31 |
| shares >200 chars but MISSED | 34 | 85 |
| shares <=200 chars but HIT | **48** | **47** |

Only 46% of hits had the immediately previous same-model call sharing a long
prefix, and 54% hit without one. A strict one-slot-per-model cache cannot
produce that. `OLLAMA_NUM_PARALLEL` is unset, so the server runs its default
parallelism (4 in recent versions), which is consistent with several recent
prompts per model staying cached.

**Consequence:** do not model Ollama as a single slot. An `ollama_single_slot`
retention policy was planned on this hypothesis and deliberately NOT built,
because the data contradicts its premise. A small-N-slot model may be worth
testing, but N is not established.

## 17. RETRACTED — "the 0.10 similarity threshold, measured from the other side"

> **This section is wrong and is kept only as a record of the error.**
> Its causal claim — that overlap at or above 0.10 destroys the displaced
> state — was refuted on 2026-10-04 by the affinity rerun with a full-length
> B, and its correlational evidence was explained away by a simpler variable.
> The correct account is §19. Read that instead; what follows is preserved
> because the mistake is instructive and because figures were drawn from it.


The affinity probe established the threshold by *construction* — it built
prompts at chosen overlaps and watched what survived. The committed traces
confirm it from the other direction, on calls nobody designed, and the split is
clean enough to settle the rule.

Taking every `issue_parser` call in the `original` trace and splitting by
whether the duration detector scored it a cache hit:

| | n | median prefix overlap with the slot |
|---|---|---|
| measured HIT | 45 | **0.023** |
| measured miss | 37 | **0.237** |

Same prompts, same role, same ~165 tokens. The hits sit an order of magnitude
*below* the 0.10 threshold and the misses sit above it, which is exactly the
mechanism:

- **below 0.10** — the slot is not eligible for in-place reuse, so the
  displaced state is saved and the host RAM cache is consulted. The incoming
  prompt is an exact duplicate of one saved earlier (each HumanEval task is
  replayed under four configs), so it is restored whole: ~165 tokens in 0.04s.
- **at or above 0.10** — the slot is trimmed to the common prefix and extended
  in place. Only that ~24% is kept and the rest is recomputed: 0.55s, which is
  the calibrated uncached rate almost exactly.

### Lower similarity to the slot is what preserves the cache

This inverts the intuition that guides most cache design. Resembling the next
prompt does not protect your state — it is precisely what gets your state
overwritten, because similarity is the trigger for in-place reuse. A 90% match
is a cheap overwrite, not a survival.

Two consequences:

1. It explains §14b from the other end. Reordering prompts to put shared
   content first raised similarity between *consecutive* calls, which is the
   condition that destroys the slot. The layout did not merely fail to help;
   under this mechanism it was actively counterproductive.
2. Any retention policy aimed at this backend has to reason about similarity to
   the *next* arrival, not just reuse distance. That is a different signal from
   the one ski-rental retention currently uses.

Measured on `original`, n=82 `issue_parser` calls. `shared_first` cannot
reproduce this split: its layout pushes nearly every call above the threshold
by construction, which is why only 17 of its 62 `issue_parser` calls are
misses and the slot path fires on 186 of 214 calls overall.

## 18. The `shared_first` trace has a degraded segment, and it is not concurrency

§14c recorded that the `shared_first` trace was written by two processes
sharing a `run_id`. The obvious follow-up — that the two overlapped in time and
contended for the GPU — is **wrong**, and worth recording so it is not assumed
again.

The two segments are **disjoint**, separated by a 21.9s gap:

| segment | calls | t_start | t_end | span |
|---|---|---|---|---|
| 0 | 86 | 1790747053.8 | 1790748687.4 | 1633.6s |
| 1 | 128 | 1790748709.3 | 1790749322.0 | 612.7s |

No concurrency. But the two ran in visibly different performance regimes,
measured as prefill rate relative to the calibrated uncached rate:

| segment | n | median ratio | p10 | slower than cold | above the 1.5x hit threshold | median call |
|---|---|---|---|---|---|---|
| 0 | 86 | 1.20x | **0.21x** | **30%** | 13% | 5.9s |
| 1 | 128 | 1.73x | 1.16x | 2% | 52% | 3.7s |

Segment 0 delivered as little as a fifth of the calibrated rate, and 30% of its
calls were slower than a *cold* prefill. Within one contiguous band of 26 calls
the debugger took 20–22s for ~800 tokens, 0.13x of calibration. By contrast
`original` has only 4 such calls out of 272, all at 0.91–0.97x, i.e. no real
degradation at all.

### Why this matters for the cache work

The hit detector thresholds at 1.5x the *calibrated* uncached rate, which
assumes that rate held during the run. In segment 0 it did not, so a genuine
cache hit there can be slower than 1.5x calibration and get scored a miss. The
ground truth is unreliable in that segment.

This accounts for the cache model's apparent false positives on
`shared_first`: under the best-fitting variant, **19 of 21 fell in segment 0**,
and under the frozen rule 9 of 10 do. They are not model errors so much as
places where the instrument lost resolution — and the trace's "78 measured
hits" is itself an undercount for the same reason.

Cause of the slowdown is not established. Memory pressure or thermal
throttling on a long run are both plausible (segment 0 averaged 19s per call
against segment 1's 4.8s) and neither was logged. `validate_cache_model.py`
now detects and reports such segments rather than averaging over them.

## 19. What actually explains the cache hits: exact prompt recurrence

Supersedes §17, which is retracted.

### The correlation that misled us

§17 reported that `issue_parser` calls scored as cache hits had a median prefix
overlap with the slot of 0.023, and misses 0.237, and read the 0.10 threshold
as the cause. Both numbers are real. The causal story was not.

The affinity rerun with a full-length B killed it directly:

| shared | 5% | 25% | 50% | 90% |
|---|---|---|---|---|
| long A recompute | 0% | 0% | 0% | 11% |
| short A recompute | 3% | 4% | 4% | 14% |

A survives intact at 25% and 50% overlap — far above 0.10. So overlap of 0.237
destroys nothing, and §17's mechanism cannot be what separated those calls.

### The variable that actually separates them

Whether the **exact prompt had been seen before**. Splitting every call by
whether its prompt text is a first occurrence or a repeat, per model:

| trace | role | occurrence | n | result | median rate vs calibration | median prefill |
|---|---|---|---|---|---|---|
| `original` | issue_parser | first | 37 | **all miss** | 1.26x | 0.647s |
| `original` | issue_parser | repeat | 45 | **all hit** | 14.41x | 0.043s |
| `shared_first` | issue_parser | first | 17 | **all miss** | 1.22x | 0.555s |
| `shared_first` | issue_parser | repeat | 45 | **all hit** | 15.23x | 0.044s |

Perfect separation, no exceptions. And it holds for every role, not just
`issue_parser`: **every exact-prompt repeat in either trace is a measured hit**
— 46 of 46 in `original`, 47 of 47 in `shared_first`.

The 45 "unexplained" `issue_parser` hits were never mysterious. Each HumanEval
task is replayed under four configs, so its `issue_parser` prompt recurs
byte-identically; the first time is cold and the later times are restored from
the host prompt cache. The overlap figures correlated only because a
first-occurrence `issue_parser` call tends to follow an `evaluator` call
(sharing a preamble, hence 0.237) while a repeat tends to follow `coder` or
`debugger`. Ordering, not destruction.

### What this does to Milestone 1's headline cache figure

Milestone 1 reported that roughly **one call in three hit Ollama's cache** —
89 of 272 (32.7%) on `original`, 78 of 214 (36.4%) on `shared_first`, worth
~60.8s and ~49.5s of prefill. That number is **mostly an artefact of the
experiment harness, not evidence of workflow prefix reuse.**

Both traces replay the same 15 HumanEval tasks under 4 configs in one process,
so prompts recur byte-identically across configs. Of the hits, 46 of 89 and 47
of 78 are exact-prompt repeats of that kind. An operator running these
workflows once would not see them.

What remains after removing them is genuine but much smaller: partial prefix
reuse, running at 2.4-4.9x the calibrated rate rather than 14-15x. The honest
headline is therefore not "a third of calls hit the cache" but "a third of
calls hit the cache, and most of that is the benchmark repeating itself."

> **Superseded by §22.** Measured on disjoint problems, the standing figure is
> about **1 call in 6**, almost entirely partial prefix reuse; the "1 in 3" was
> about half replay artefact.

This is the reason for the new trace design in OPEN_ISSUES §21: disjoint
problems and an Ollama restart between configs, so the measured reuse is the
workflow's own.

### Partial hits are real but much weaker

The calls that are first occurrences yet still score as hits (43 in `original`,
31 in `shared_first`) run at a median 2.4-4.9x calibration, against 14-15x for
exact repeats. That is genuine partial prefix reuse — a novel prompt sharing a
long prefix with something cached — and it is an order of magnitude smaller
than an exact restore. Any model of this backend needs both terms, and should
not conflate them.

### What llama-server's own binary says

Strings from `llama-server` and `libllama-server-impl.dylib` as shipped with
Ollama 0.33.3 (`libllama.0.3.0.dylib`, `libggml.0.22.0.dylib`; no build tag):

```
idle slots will be saved to prompt cache upon starting a new task
saving idle slot to prompt cache
updating prompt cache
prompt cache update took %.2f ms
prompt cache is enabled, size limit: %d MiB
 - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)
 - cache size limit reached, removing oldest entry (size = %.3f MiB)
 - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)
slot ...: id %2d | task %d |  - checking sim = %.3f (%zu/%zu) > %.3f
slot ...: id %2d | task %d | selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f
save idle slots to the prompt cache on new task, and clear them when using
  unified KV (default: enabled, requires cache-ram)
```

Four things follow, and they align with the corrected measurements:

1. **The save is unconditional.** "idle slots will be saved to prompt cache
   upon starting a new task" has no similarity condition attached. This is the
   direct support for saving-on-displacement regardless of the 0.10 threshold.
2. **Two separate quantities exist.** `f_sim_best` chooses the slot against a
   threshold (`thold`); `f_keep` is logged separately as how much of that
   slot's cache is retained. §17 treated one number as doing both jobs.
3. **Eviction is oldest-first, bounded by both bytes and tokens** — three
   distinct "removing oldest entry" messages, one for a size limit in MiB and
   one for a token limit. The byte bound matches the capacity probe (~39 x
   207 MiB ~= 8 GiB); the token bound was not previously modelled at all.
4. **There is no prompt-cache state dedup.** Searching both binaries for
   dedup, replacement or identical-state handling returns only
   `__PRESET_DEDUP_CACHE_MODELS` / `dedup-cache-models` / "hiding cache model
   name=%s (deduplicated by a preset)", which concern **model names in a
   preset list**, not cached states. The dedup hypothesis has no support in the
   shipped binary. That does not refute it — absence of a string is not
   absence of logic — but nothing was found, and it should not be assumed.

The `(%zu/%zu)` in the `checking sim` line prints the fraction's numerator and
denominator, so a single server log at `--verbose` would settle whether
similarity is measured against the incoming prompt or the cached one. That is
cheaper than any further probe and is worth capturing on the next run.

## 20. The cache rule, read out of llama.cpp's source

Supersedes the mechanism in §19; §19's empirical findings stand.

### The rule, cited

`llama.cpp` `tools/server/server-context.cpp`:

```c
const float f_sim_cur = float(lcp_len) / task.tokens.size();
const float f_keep    = (f_sim_best*task.tokens.size())
                        / ret->prompt.tokens.size();
if (f_keep < 0.5f) { update_cache = true; }
```

Three things settled, each of which we had wrong at some point:

1. **`f_sim` is over the incoming prompt; `f_keep` is over the cached/slot
   prompt.** Two different denominators, confirmed independently by a log
   triple: slot `n_tokens = 379`, `checking sim = 0.130 (3/23)` (3/23,
   incoming), `f_keep = 0.008` (3/379, slot).
2. **The gate is 0.5 on `f_keep`, literal in the source.** Not 0.10, which is
   `--slot-prompt-similarity` for choosing *between* slots and is inert at
   `-np 1`. Reading 0.10 as the retention threshold is the root of §17.
3. **`f_keep < 0.5` triggers `update_cache`** — both the save and the
   "looking for better prompt" search. Above 0.5 the slot is reused in place
   and the remainder is discarded, never saved.

Independent agreement from `server.log`, 758 slot-selection events:
`f_keep >= 0.5` searched the cache in 0.8% of 128 events; `f_keep < 0.5` in
64.8% of 630.

### Cache limits

`cache state: 197 prompts, 6507.8 MiB (limits: 8192.000 MiB, 4096 tokens, 149762 est)`

- **8192 MiB** byte budget — matches the capacity probe (~39 x 207 MiB).
- **N tokens** is a *per-prompt* cap tracking `num_ctx` (4096, 8192 and 16384
  all observed), not a cache total.
- **`est`** is a derived total-token estimate, printed as raw bytes
  (`8589934592` = 8 GiB) while the cache is empty. Not a limit.
- Eviction is oldest-first: `making room for prompt cache entry` fired 22
  times; no size- or token-limit eviction messages appear in these logs.
- The cache reached **197 resident prompts**, so it is nowhere near scarce at
  this workload's state sizes.

### Scores, with reuse split by source

| trace | hits | FP | agreement | exact-prompt repeats | partial prefix reuse |
|---|---|---|---|---|---|
| `original` | 84/89 | 1 | 97.8% | **46/46** | 38/43 |
| `shared_first` | 76/78 | 21 | 89.3% | **47/47** | 29/31 |

The split is the point. The exact-repeat column is harness artefact
(OPEN_ISSUES §20) and the model captures all of it; the partial column is the
only part reflecting workflow reuse. 19 of `shared_first`'s 21 false positives
sit in its degraded segment (§18).

**Char/token approximation.** The server computes both ratios on tokens; this
model approximates with characters scaled by each call's chars-per-token ratio.
Calls with `f_keep` within ±0.05 of the gate are therefore uncertain: **3 of
272 (1.1%) on `original`, 12 of 214 (5.6%) on `shared_first`**. Small, but
`shared_first`'s layout pushes more calls toward the gate, so it is the trace
where the approximation matters most.

## 21. The server discards reusable state in the retry loop — at a rate set by how often the coder retries

**Recomputed 2026-10-05 on the held-out traces.** The original 25/152 figure
came from traces containing cross-config replay, so it is no longer the
headline. The two holdout traces disagree, and the disagreement is the finding.

For every within-task, same-model handoff — the calls the workflow actually
chains (`coder -> debugger -> coder`, retries, replans) — counting those whose
`f_keep >= 0.5`, where the slot is reused in place and the remainder is
discarded with no cached copy:

| trace | handoffs | `f_keep >= 0.5` | coder→coder share of crossings | median `f_keep` | coder pass rate |
|---|---|---|---|---|---|
| `holdout_llama` (llama coder) | 262 | **51 (19.5%)** | **47/51 (92%)** | 0.107 | 34/54 |
| `holdout_qwen` (qwen coder) | 212 | **3 (1.4%)** | 0/3 | 0.012 | 45/54 |
| `original` (fit, has replay) | 152 | 25 (16.4%) | 24/25 (96%) | 0.053 | — |
| `shared_first` (fit, has replay) | 94 | 9 (9.6%) | 0/9 | 0.204 | — |

### The headline, and why the two traces differ

**Taking the held-out llama-coder trace as the headline: 19.5% of within-task
same-model handoffs discard reusable state, and 92% of those are
`coder -> coder` retries.** That reproduces the `original` pattern closely
(16.4%, 96%), so the effect is not an artefact of cross-config replay — the
replay-free trace shows it just as strongly.

But the qwen-coder trace shows almost none of it (1.4%). The difference is not
the workload — both ran the same 54 problems — it is **how often the coder
retries**:

| | qwen coder | llama coder |
|---|---|---|
| coder calls | 79 | 103 |
| debugger calls | 34 | 69 |
| tasks passed | 45/54 | 34/54 |

The weaker coder fails more, retries more, and a retry is precisely the handoff
whose prompt differs from its predecessor only by an appended diagnosis — so
`f_keep` clears 0.5 and the server overwrites the slot in place. The strong
coder mostly succeeds first time, so the retry handoff barely occurs.

**This causal story is consistent with all four traces but is not proven.** It
is a correlation across two configs, and coder model and retry rate are
confounded here: we cannot tell whether the llama coder crosses the gate more
because it retries more, or because its prompts happen to be shaped differently.
Separating them needs a run where the same model is made to retry more (e.g. by
tightening the test suite), which has not been done.

### Why it matters for the thesis

The handoff most certain to want its predecessor's state is the one the server
drops the remainder of, with no cached copy to fall back on. A policy that knew
the workflow would reuse that state could keep it; the server cannot know and
does not.

The dependence on retry rate sharpens rather than weakens the claim: the
opportunity is largest exactly where the pipeline is struggling, which is also
where latency hurts most. But it also means the benefit is **not uniform** — a
deployment with a strong coder would see little of it, and any headline number
has to name the coder it was measured with.

## 22. Held-out validation: the rule generalises, and the Milestone 1 hit rate halves

Two traces collected 2026-10-04/05 on 54 problems disjoint from the fit set,
two configs in two processes with Ollama restarted between them. The frozen v2
rule was run against them with **no tuning**.

| trace | calls | measured hits | matched | FP | FN | **hit recall** | agreement |
|---|---|---|---|---|---|---|---|
| `holdout_A_qwen2` (qwen coder) | 320 | 54 | 44 | 1 | 10 | **81%** (44/54) | 96.6% |
| `holdout_B_llama` (llama coder) | 370 | 66 | 53 | 4 | 13 | **80%** (53/66) | 95.4% |
| `original` (fit) | 272 | 89 | 84 | 1 | 5 | 94% (84/89) | 97.8% |
| `shared_first` (fit) | 214 | 78 | 76 | 21 | 2 | 97% (76/78) | 89.3% |

**Read the recall column, not the agreement column.** Agreement is dominated
by true negatives — only ~17% of held-out calls are hits, so a model that
predicted "miss" for everything would already score ~83%. Recall is the number
that says whether the model finds the reuse: **81% and 80% held out, against
94% and 97% on the traces it was fit to.**

So the rule does generalise, but it loses about 14 points of recall on unseen
data. That gap is the honest measure of how much the fit-set scores were
flattered, and all of it sits in one role (below).

### The repetition artefact, confirmed quantitatively

OPEN_ISSUES §20 predicted that Milestone 1's hit rate was inflated by the
harness replaying 15 tasks under 4 configs. Removing that — disjoint problems,
one config per process, cache cleared between — gives:

| trace | calls | hits | hit rate | exact repeats | partial reuse | exact share |
|---|---|---|---|---|---|---|
| Milestone 1 `original` | 255 | 89 | **34.9%** | 41 | 48 | **46%** |
| `holdout_A_qwen2` | 320 | 54 | **16.9%** | 4 | 50 | **7%** |
| `holdout_B_llama` | 370 | 66 | **17.8%** | 1 | 65 | **2%** |

**The corrected statement, which supersedes the Milestone 1 figure wherever it
appears: about 1 call in 6 hits Ollama's cache on disjoint problems, and it is
almost entirely partial prefix reuse. The earlier "1 in 3" was about half
replay artefact.**

The headline halves from ~35% to ~17% and the exact-repeat share collapses from
46% to 2-7%.

The encouraging half: partial reuse is not only intact but larger in absolute
terms (48 → 50 and 65 hits). Genuine workflow prefix reuse is real, robust, and
was being obscured by the artefact sitting on top of it.

### Where the rule still fails: `issue_parser`

The residual is concentrated in one role, in both traces:

| role | A matched/measured | B matched/measured |
|---|---|---|
| coder | 25/25 | 47/49 |
| debugger | 13/13 | 6/6 |
| evaluator | 3/3 | 0/1 |
| **issue_parser** | **3/13** | **0/10** |

The roles the workflow actually chains are near-perfect. `issue_parser` is
where 23 of the 23 false negatives and all 5 false positives live. With
disjoint problems its prompts no longer recur exactly, so whatever reuse it
gets is not exact-match restoration and is not yet modelled. That is the next
thing to explain, and it is now a narrow question rather than a diffuse one.

## 23. RETRACTED — "LSF buys tail latency, and a perfect estimator does not help"

> **Measured in accidental overload; the numbers below are void.** At the time,
> `calibrate_rate` took no `batch` argument, so it calibrated the arrival rate
> against `run_one`'s default `batch=16` (0.0753/s) and the table then ran at
> `batch=8`. The run was therefore at about **2.2x the intended load**. With
> calibration and execution agreeing the rate is 0.0343/s, p95 is ~133s rather
> than 274-322s, and the scheduler's `choice` fraction is **0.3%, not 34.7%**.
> The claimed 14.8% p95 win and the "oracle is worse" result are both gone.
> Corrected in §27. Kept as a record of the error.


`experiments/run_scheduler_table.py`, `holdout_qwen`, scale 1x, util 80%,
kv 1024MB, batch 8, retention held fixed at `lru`, **20 seeds**,
`slack_factor` calibrated to 1.875 so FCFS misses 29.0% of deadlines.

| scheduler | p50 | p95 | miss | slack@done | MAPE | choice |
|---|---|---|---|---|---|---|
| fcfs | **115.8** | 321.6 | 32.1% | 30.3 | 37.8% | 34.7% |
| sjf | 119.8 | 293.6 | 31.3% | 34.1 | 37.6% | 33.1% |
| edf | 118.6 | 274.7 | **29.9%** | **39.5** | 37.7% | 30.8% |
| lsf | 120.5 | **274.0** | **29.9%** | 38.0 | 37.7% | 31.5% |
| static-priority | 119.7 | 293.1 | 31.0% | 34.0 | 37.6% | 33.3% |
| lsf [ORACLE] | 131.0 | 272.2 | 34.3% | 32.8 | 0.0% | 33.1% |

### LSF trades median for tail, as intended

p95 falls **14.8%** (321.6s → 274.0s) and the deadline miss rate from 32.1% to
29.9%, while p50 *rises* 4% (115.8 → 120.5). That is the expected shape: LSF
deprioritises work with slack, which costs the median slightly and protects the
tail. EDF is statistically indistinguishable from LSF here.

The effect is real but modest — a 15% p95 improvement, not a transformation.

### A perfect estimator makes LSF WORSE

The oracle row has zero estimation error (MAPE 37.7% → 0.0%) and is **worse on
the miss rate** (34.3% vs 29.9%) and on p50 (131.0 vs 120.5), gaining only
0.7% on p95. This is counter-intuitive and it undercuts the premise of
workstream 3.

Likely mechanism: with true remaining critical paths, LSF orders decisively
toward tasks that genuinely have the most work left, which starves short tasks
and pushes them past their deadlines. The online estimator's stage-typical
prompt sizes and expected-retry term effectively *regularise* the ordering,
and that smoothing is worth more than accuracy at this operating point.

**Treat this as provisional.** It is one trace at one operating point, and the
oracle's `remaining_s` is the realised critical path while the online one is an
expectation over retries — related but not the same quantity. It should be
re-checked at other utilisations and on `holdout_llama` before workstream 3 is
rescoped. But the burden of proof has shifted: a better estimator can no longer
be assumed to help.

### The operating point had to be engineered, and that is itself a result

At the engine's default `max_batch=16` the scheduler has a choice on under 9%
of admission decisions, so **no scheduling rule can distinguish itself** — the
first version of this table showed all five schedulers within noise and the
oracle row byte-identical to the online one. `max_batch=8` yields ~34% choice
and lands FCFS in the target miss band.

| max_batch | choice | p95 | FCFS miss |
|---|---|---|---|
| 16 | 8.9% | 339.9 | 22.2% |
| 8 | 41.8% | 396.8 | 27.8% |
| 4 | 71.8% | 560.4 | 44.4% |
| 2 | 90.5% | 903.3 | 63.0% |

So any scheduling claim about this workload has to state the concurrency limit
it holds at. At generous batch sizes the queue is nearly always trivial and
scheduling is a non-question.

## 24. Why retention cannot move latency on this workload

Diagnostic before the grid, `holdout_qwen` 1x, util 0.8, batch 4, KV 512MB,
FCFS — inside the live-rent band (OPEN_ISSUES §27).

### The decisive number: recompute is ~1% of makespan

> **Regenerated 2026-10-07: 20 paired seeds, measured `batch_decode_coeff`
> 0.841 (DECISIONS §28).** The table below this note is a **single seed**:
> rerunning the current code at 0.15 with seed 1 reproduces it exactly
> (keep-all 13.4s / 187.6s / makespan 1603.9s), while seeds 2 and 3 give
> 16.3s / 146.8s and 0.4s / 91.7s. Its absolute numbers are one draw; the
> table here supersedes them. Same point (holdout_qwen 1x, util 0.8, batch 4,
> KV 512MB, FCFS), rate recalibrated to 0.0265/s, manifest in
> `results/finding24_measured.json` (`experiments.sweep_decode_coeff --coeffs
> 0.841 --seeds 20 --ret-policies keep-all ski-rental evict-now offline-ref`).
>
> | policy | recompute | prefill | decode | idle | makespan | p95 | vs keep-all p95 | vs keep-all recompute |
> |---|---|---|---|---|---|---|---|---|
> | keep-all | 13.8s | 409.2s | 1201.8s | 542.7s | 2178.3s | 277.7 | | |
> | ski-rental | 24.0s | 419.4s | 1201.6s | 540.2s | 2185.8s | 278.3 | +0.6 [-7.0, +8.2] | +10.2 [+4.8, +15.6] |
> | evict-now | 125.7s | 521.0s | 1197.9s | 477.9s | 2220.7s | 314.7 | +36.9 [+24.8, +49.0] | +111.9 [+103.6, +120.2] |
> | offline-ref | 24.6s | 420.0s | 1201.7s | 541.6s | 2187.9s | 278.7 | +1.0 [-6.4, +8.4] | +10.8 [+6.0, +15.6] |
>
> Means over seeds; intervals are paired 95% CIs (DECISIONS §24). Keep-all
> recompute is **0.7% of makespan** (mean of per-seed ratios). Decode is 55%
> of makespan, prefill 19%, idle 25%. The conclusion stands and is sharper:
> no deployable policy differs from keep-all on p95, evict-now is
> significantly worse on both, and ski-rental and the offline reference both
> recompute significantly more than keep-all.


| policy | recompute | prefill | decode | idle | makespan | recomp/makespan | p95 |
|---|---|---|---|---|---|---|---|
| keep-all | 13.4s | 408.7s | 883.1s | 285.8s | 1603.9s | **0.8%** | 187.6 |
| ski-rental | 18.4s | 413.8s | 882.8s | 283.8s | 1606.7s | **1.1%** | 185.8 |
| evict-now | 127.4s | 522.8s | 856.9s | 207.4s | 1612.4s | 7.9% | 220.3 |
| offline-ref | 29.4s | 424.8s | 878.8s | 286.6s | 1616.3s | 1.8% | 184.7 |

Decode is 55% of makespan, prefill 25%, idle 18%. Avoided recompute is the
*only* thing a retention policy can win, and it is **0.8-1.1%**. So the
latency spread across every deployable policy is 185.8-187.6s p95 — **within
1%** — and the offline reference, the best achievable, beats keep-all's p95 by
1.5%.

**Retention is not a latency mechanism on this workload.** It is a GPU-work
mechanism: 13.4s against evict-now's 127.4s is a 9.5x difference in wasted
prefill, which matters for throughput and cost per task, not for what a user
feels.

### The decision space is not empty, but the headroom is in the wrong term

Answering the direct question: keep-all's realized ratio against its own
hindsight is **1.20-1.55**, not ~1.0, so there *is* cost headroom.

| batch | KV | keep-all ratio | ski-rental ratio | rent-active |
|---|---|---|---|---|
| 2 | 512 | 1.545 | 1.554 | 44.9% |
| 2 | 1024 | 1.465 | 1.225 | 37.4% |
| 4 | 512 | 1.461 | 1.300 | 12.6% |
| 4 | 1024 | 1.362 | 1.197 | 14.2% |
| 8 | 512 | 1.450 | 1.475 | 3.5% |

**But `realized_ratio` is not comparable across policies** and these numbers
are misleading on their own. It divides by each run's *own* hindsight, and
ski-rental accrues more rent, which inflates `min(rebuild, rent)` per key and
therefore its own denominator. That is why ski-rental looks better here while
costing more in absolute terms. Against the **shared keep-all denominator**
(109.1 recompute-s), which is what the grid uses:

| policy | cost | ratio | vs keep-all | p50 | p95 |
|---|---|---|---|---|---|
| keep-all / lru | 159.4 | 1.461 | 1.000 | 49.6 | 187.6 |
| evict-now | 127.4 | 1.168 | **0.800** | 54.1 | 220.3 |
| ski-rental | 184.6 | 1.693 | 1.158 | 50.9 | 185.8 |
| ski-rental-pred | 184.2 | 1.689 | 1.156 | 50.9 | 184.0 |
| offline-ref | 116.3 | 1.067 | **0.730** | 50.8 | 184.7 |

### The composite objective inverts the ranking

`recompute + rent` rates **evict-now as 20% better than keep-all** — while
evict-now is 17% worse on p95 and 9.5x worse on GPU work. It also rates
`offline-ref` best, yet offline-ref's recompute (29.4s) is *higher* than
keep-all's (13.4s): it trades recompute for rent. The reference is optimising
an objective that does not correspond to anything a user or an operator cares
about.

So latency and GPU recompute are the reportable metrics. Rent stays as an
internal diagnostic.

### Why ski-rental pays more rent while holding less — not a bug

It holds **4.5MB less** (470.8 → 466.3 mean retained) and pays **20.2s more
rent** (146.0 → 166.2), at an essentially identical eviction rate (5.058 vs
5.070). Traced over all 862 of its voluntary releases:

| | |
|---|---|
| releases | 862 |
| released and never needed again | **776 (90%)** |
| released then rebuilt later | 86 (10%), p50 gap 31.7s |
| rebuilt more than once | 6 |
| rent at release vs break-even | p50 0.0597s vs p50 b 0.0597s — **fires exactly at break-even** |
| `useful_tokens / tokens` at release | **1.000** — break-even is priced on the full block, no mispricing |
| keys rebuilt *and* charged >1.5x their rebuild cost in rent | 57 |

No bug: the rule fires exactly where it is designed to, and 90% of its releases
are correct in hindsight. The explanation is the **rent model**, not the policy.
Ski-rental's threshold is the entry's own recompute cost, so it releases the
**cheapest-to-rebuild** entries first, leaving an expensive-to-rebuild residue.
Derived rent is proportional to the victim's recompute-per-byte and normalised
by the evictable pool (DECISIONS §13), so a more expensive residue accrues rent
faster. Releasing shifts rent onto whoever still holds rather than removing it.

The structural consequence: **rent is nearly invariant to the policy in the
regime where the policy operates** — 146s vs 166s — and only collapses when
almost nothing is held (evict-now: 0s). An objective dominated by a term the
policy cannot reduce will mislead, and here it does: ski-rental spends 5.0s of
extra recompute to chase 20.2s of rent it cannot actually avoid.

### Framing the Ollama comparison

The Ollama-vs-block gap (239.3s vs 159.4s, DECISIONS §22) is an **architecture**
result: shared block prefix caching against one slot plus a host tier.
**`keep-all` already captures all of it** — no clever policy is needed.

So workflow-aware retention's own contribution must be measured as
**block/ski-rental against block/keep-all**, never against Ollama. Quoting the
1.50x architecture gap as if a retention policy earned it would be wrong. On
that honest comparison, ski-rental is currently **1.158x worse** than keep-all
on cost and level on latency.

## 25. Ski-rental retention: an analysed negative result

Ski-rental's premise is that **holding has a cost the policy can avoid by
releasing**. On this workload that premise is false, which is why voluntary
release cannot pay off — and the reason is structural rather than a tuning
failure.

Releasing does not remove rent, it **shifts** it. Derived rent is normalised by
the evictable pool, so when an entry is released the remaining holders absorb a
larger share of the same pressure. Measured at `holdout_qwen` 1x, util 0.8,
batch 4, KV 512MB:

| | keep-all | ski-rental |
|---|---|---|
| mean retained | 470.8 MB | **466.3 MB** (4.5 MB less) |
| rent | 146.0s | **166.2s** (20.2s more) |
| GPU recompute | 13.4s | 18.4s |
| eviction rate | 5.058/s | 5.070/s |
| p95 | 187.6s | 185.8s |

It holds *less* and pays *more*. The policy is not malfunctioning: over all
862 of its voluntary releases, **90% were never needed again**, the rule fires
exactly at break-even (p50 accrued rent 0.0597s against p50 break-even
0.0597s), and `useful_tokens / tokens` is 1.000, so the break-even is not
mispriced. Only 57 keys were rebuilt *and* charged more than 1.5x their rebuild
cost in rent.

The mechanism is selection. Ski-rental's threshold is each entry's own
recompute cost, so it releases the **cheapest-to-rebuild** entries first and
leaves an expensive-to-rebuild residue — which accrues rent *faster*, because
rent scales with the victim's recompute-per-byte. Net effect: it spends 5.0s of
real GPU recompute chasing 20.2s of rent it cannot actually avoid.

**Conclusion.** Ski-rental is the wrong shape of mechanism for this setting.
Hold-or-release only pays when holding carries a private, avoidable cost; here
the cost is shared and roughly invariant (DECISIONS §23). The measured failure
in the real server is not that it holds too long — it is that it discards state
the workflow is about to reuse (§21), which is an **ordering** problem. That is
what §26 tests instead.

## 26. Belady ordering loses to LRU here, and Belady's optimality does not apply

`pred-belady` ranks forced-eviction victims by predicted next use, furthest
first; `offline-belady` does the same with the true next use. Measured at
`holdout_qwen` 1x, util 0.8, batch 4, KV 512MB, FCFS:

| policy | p50 | p95 | GPU recompute | forced evictions |
|---|---|---|---|---|
| keep-all / lru | 49.6 | 187.6 | **13.4s** | 6813 |
| ski-rental | 50.9 | 185.8 | 18.4s | 5984 |
| pred-belady sigma=0 | 50.8 | **184.7** | 29.4s | 7094 |
| pred-belady sigma=0.5 | 52.9 | 190.5 | 46.6s | 7332 |
| offline-belady | 50.8 | **184.7** | 29.4s | 7094 |

`pred-belady` at sigma=0 is identical to `offline-belady`, which is the
consistency check passing: perfect predictions reproduce the oracle. Noise
degrades it gracefully (29.4 -> 46.6s recompute).

**But Belady loses to LRU on the metric it is supposed to optimise** — 29.4s of
recompute against 13.4s. The oracle is not at fault; that was checked
directly. Of 7094 forced evictions under `offline-belady`, 6030 were entries
the oracle said would never be reused and **0%** of those were ever rebuilt,
while 35.7% of the 1064 evictions with a real future gap were. The ordering and
the next-use information are both correct.

### Why: Belady's optimality does not hold in this engine

Belady's rule is optimal for **uniform-size items evicted one at a time**. This
engine satisfies neither condition. Blocks differ in size across models — qwen
is 56 KiB/token and llama3.2 112 KiB/token, and `holdout_qwen` contains both —
and eviction is **bulk**: the engine frees whatever is needed to admit an
entire prompt, not one block per miss. Choosing a set of victims to free N
bytes while minimising future misses is a knapsack-shaped problem, and
furthest-next-use is not its solution.

**Consequence for the plan: `offline-belady` is not a valid "minimum recompute
under the memory budget" reference.** It is a well-defined, verified ordering
with a correct oracle, and it is beaten by LRU, so it cannot be an upper bound
on what retention can achieve. This is the same trap as `offline-ref`
(FINDINGS §24), arrived at from the other direction: a reference that optimises
the wrong objective, or the right objective under assumptions the engine
violates, will be beaten by a trivial policy.

A genuine minimum-recompute reference here needs either a size-aware offline
solver or an explicitly-labelled lower bound that is not claimed to be
achievable.

## 27. Scheduler table, corrected: no scheduling rule differs significantly

Supersedes the retracted §23. `holdout_qwen` 1x, util 80%, KV 1024MB,
retention fixed at `lru`, **20 seeds**, `slack_factor` calibrated to 0.844 so
FCFS misses 27.8%. Arrival rate 0.0343/s, calibrated **at the same batch the
table runs at** — the error that invalidated §23.

> **Correction (DECISIONS §25).** As first run, that held only for the batch-8
> rows: the driver calibrated at batch 8 whatever batch it ran. Recalibrated at
> batch 16 the rate is bit-identical (0.0343/s), and the 20-seed batch-16 rerun
> (`results/scheduler_table_holdout_qwen_b16.json`) is still not significant
> for every rule. The conclusion stands; its batch-16 CIs differ from the table
> below by at most 0.01s.

| batch | row | p50 | p95 | miss | missed tasks | already infeasible at first pick |
|---|---|---|---|---|---|---|
| 8 | fcfs | 47.3 | 133.2 | 31.4% | 16.9 | 1.0 (**6%**) |
| 8 | lsf | 47.6 | 133.0 | 31.4% | 16.9 | 1.0 (6%) |
| 8 | edf | 47.5 | 132.4 | 31.6% | 17.1 | 1.0 (6%) |
| 8 | lsf [ORACLE] | 47.4 | 132.5 | 31.6% | 17.1 | 1.0 (6%) |
| 16 | fcfs | 47.4 | 132.5 | 31.5% | 17.0 | 1.0 (6%) |
| 16 | lsf | 47.5 | 132.6 | 31.5% | 17.0 | 1.0 (6%) |

### Paired 95% confidence intervals on the p95 difference against FCFS

Paired, because every row runs the same 20 seeds.

| batch | row | p95 difference | 95% CI | verdict |
|---|---|---|---|---|
| 8 | lsf | -0.25s | [-0.92, +0.42] | **not significant** |
| 8 | edf | -0.78s | [-1.98, +0.43] | **not significant** |
| 8 | lsf [ORACLE] | -0.73s | [-2.36, +0.91] | **not significant** |
| 16 | lsf | +0.03s | [-0.07, +0.13] | **not significant** |
| 16 | edf | +0.04s | [-0.05, +0.14] | **not significant** |
| 16 | lsf [ORACLE] | +0.08s | [-0.78, +0.93] | **not significant** |

**Every scheduling rule is statistically indistinguishable from arrival order,
at both concurrency limits.** So is the oracle. The reason is in the activity
line: at the correctly calibrated rate the scheduler has a choice on **0.3-0.4%
of admission decisions** — the ready queue almost never holds more than one
request. A scheduler with nothing to choose between cannot differ from FCFS.

### The overload hypothesis is not supported

The proposed explanation for §23's oracle result was that under overload
LSF/EDF keep serving tasks that will miss anyway, cascading. Classifying every
missed task by whether it was **already infeasible at its first pick**, judged
by the true critical path for all rows so the classification is identical
across them: only **6%** (1.0 of 17) were doomed from the start, in every row
alike. There is no pool of hopeless tasks being serviced, so cascading misses
are not what happened. The §23 oracle result was the overload artefact itself,
not a property of LSF.

### What would make scheduling measurable

Genuine overload, which is what §23 had by accident. That is a legitimate
regime to study, but it has to be *chosen* and labelled, with the arrival rate
calibrated at the concurrency the run actually uses. The methodological rule
this cost us: **calibrate every parameter at the configuration the measurement
runs at**, and check the activity line before believing any comparison.

---

## 28. The sealed `final` set: pass@1, and the cache rule on unseen problems

Collected 2026-10-07 (commit `676ba59`) on the 92 `final` problems
(DECISIONS §26), with the same two configs as the held-out traces, zero-shot,
one process each and Ollama restarted between. Both traces pass
`check_trace_clean`. Machine state was logged per minute in
`data/traces/final_machine_state.log`. The first 1 to 3 minutes of
`final_A_qwen` overlapped two pytest runs, so its earliest call timings are
contended; pass/fail is unaffected.

| trace | config | tasks | calls | wall clock |
|---|---|---|---|---|
| `final_A_qwen` | `single_agent` (qwen coder) | 92 | 810 | 95 min |
| `final_B_llama` | `homo_llama3` (llama coder) | 92 | 924 | about 70 min |

The policy drivers have **not** been run on these traces. They are kept out
of `TRACE_SETS` and refuse to load until a freeze commit is named
(`SEALED_TRACE_SETS`, `tests/simulator/test_sealed_traces.py`).

### pass@1, with intervals

Wilson 95% intervals per config. The difference is paired on the same
problems: a 95% normal interval on per-task differences, and an exact
McNemar test on the discordant tasks.

| set | qwen coder | llama coder | qwen − llama | qwen-only / llama-only | McNemar p |
|---|---|---|---|---|---|
| `final` (92) | 69 = 75.0% [65.3, 82.7] | 45 = 48.9% [38.9, 59.0] | +26.1 pts [+14.8, +37.4] | 29 / 5 | 4e-5 |
| holdout (54) | 45 = 83.3% [71.3, 91.0] | 34 = 63.0% [49.6, 74.6] | +20.4 pts [+5.4, +35.4] | 15 / 4 | 0.019 |
| **both (146)** | 114 = **78.1%** [70.7, 84.0] | 79 = **54.1%** [46.0, 62.0] | **+24.0 pts** [+15.0, +33.0] | 44 / 9 | 1e-6 |

The qwen coder beats the llama coder on unseen problems, by about 24 points
with an interval well clear of zero. Both configs run the full multi-agent
pipeline (OPEN_ISSUES §14), so this is a **coder-model comparison**, not
multi-agent against single-agent. These are implement-from-stub rates over
canonical HumanEval and carry DECISIONS §20's caveat.

### The frozen cache rule, third test

The v2 rule (DECISIONS §19) was run unchanged, as §26 rule 2 allows. The
held-out traces were rerun beside it and reproduce §22 exactly.

| trace | calls | measured hits | matched | FP | FN | **hit recall** | precision | agreement |
|---|---|---|---|---|---|---|---|---|
| `final_A_qwen` | 810 | 69 | 53 | 5 | 16 | **77%** (53/69) | 91% | 96.2% |
| `final_B_llama` | 924 | 121 | 117 | 4 | 4 | **97%** (117/121) | 97% | 98.8% |
| `holdout_A_qwen2` (§22) | 320 | 54 | 44 | 1 | 10 | 81% | 98% | 96.6% |
| `holdout_B_llama` (§22) | 370 | 66 | 53 | 4 | 13 | 80% | 93% | 95.4% |

Three things this adds to §22:

1. **The rule generalises to a third, larger set.** Recall is 77% and 97%
   with no tuning. Pooled over every unseen trace it is 267/310 = 86%.
2. **`issue_parser` is still the weak role** (OPEN_ISSUES §24): recall 1/8
   on qwen and 3/7 on llama, and it holds all 9 false positives. On llama it
   accounts for every miss.
3. **New: the qwen trace misses some coder and debugger hits** (6 and 3),
   which never happened on the holdout. Not investigated yet. It is the next
   thing to look at, by inspection only, with no change to the rule.

The hit rate is lower than §22's "about 1 call in 6": 8.5% of calls on qwen
and 13.1% on llama, against 16.9% and 17.8% on the holdout. Treat "1 in 6"
as a holdout figure. Across all four unseen traces the rate is 310/2424 =
**12.8%, about 1 call in 8**.

---

## 29. Scheduling regime map: ordering matters once requests queue, and the gain is task order, not slack

The question FINDINGS §27 could not answer from one operating point: is
there **any** load and concurrency where a scheduling rule beats FCFS?

**Setup.** `experiments/scheduling_map.py` at commit `36de19e`, output
`results/scheduling_map.json` (manifest inside: profile
`mac_ollama_measured`, `batch_decode_coeff` 0.841 MEASURED, every rate
and slack calibration recorded and checked before each cell; the tree was
dirty only in `data/traces/final_machine_state.log`, which no simulator
code reads). `holdout_qwen` 1x, KV 1024 MB, retention fixed at LRU, 20
paired seeds, five schedulers (FCFS, SJF, EDF, LSF, LSF with the oracle
estimator). Figures: `results/figures/regime/scheduling_p95.png` and
`scheduling_miss.png`.

- **Load axis:** arrival rate / the rate calibrated to GPU busy 0.8 at the
  same batch size, so 1.00 is §27's point. 0.78, 0.85, 1.22 are calibrated
  directly at util 0.6, 0.7, 0.9; above that, multiples of the util-0.8
  rate (busy fraction saturates and cannot be calibrated past 1). 2.20 is
  §23's accidental overload; 3.00 is one point beyond.
- **Deadlines:** slack_factor calibrated once per batch at util 0.8 (FCFS
  misses 23-25%), then held fixed along the load axis.
- **Miss split:** "infeasible" = the deadline was already out of reach,
  on the uncontended critical path, when the task's first request was
  admitted. No scheduler could have saved those.

### The answer: where LSF and EDF first separate from FCFS

| batch | LSF first separates (p95) | gap there | EDF first separates | p95 change at load 1.00 | at 2.20 |
|---|---|---|---|---|---|
| 2 | at or below 0.78 (lowest tested) | −35.7s [−54.6, −16.8] | ≤ 0.78, −37.6s | 296 → 232s (−22%) | 948 → 750s (−21%) |
| 4 | at or below 0.78 | −17.2s [−26.7, −7.7] | ≤ 0.78, −19.7s | 265 → 226s (−15%) | 879 → 688s (−22%) |
| 8 | 1.10 | −15.4s [−28.5, −2.2], miss +1.1 pts | 1.10, −12.6s | 251 → 246s (n.s.) | 834 → 686s (−18%) |
| 16 (KV-bound) | 1.75 | −24.2s [−40.1, −8.3] | 1.50, −5.7s | no change | 808 → 756s (−6%) |

Batch 8 and 16 are **not** identical and were not merged. Batch 16's cap
never binds at any load (at most 12 in flight): the 1024 MB KV budget, not
`max_batch`, limits concurrency there. Batch 8 reaches its cap only
fleetingly (0.0% of time at cap everywhere), so it is effectively KV-bound
too, with a lower ceiling than 16.

### What this adds to §27

1. **§27 holds at its own point.** At batch 8 or 16 and load ≤ 1.0 the
   scheduler has a choice on ≤ 2.2% of decisions (flagged inactive) and no
   rule separates, the same null.
2. **Outside that point the null breaks, and by a lot.** Cap concurrency at
   2 or 4, or push load past about 1.1 at batch 8, and a queue forms. Then
   every rule beats FCFS on p95 at every load tested, by 15-22% at load 1.0
   and up to about 200s in overload, every interval clear of zero.
3. **The gain is task order, not slack.** Every task's deadline is its
   arrival plus one common constant (`slack_factor` × the estimator's
   isolated task time), so **EDF here is exactly oldest-task-first**, while
   FCFS is oldest-*request*-first and interleaves the stages of different
   tasks. EDF and LSF differ significantly in only 1 of 40 cells (batch 4,
   load 3.00, EDF better by 11.7s [1.9, 21.4]; paired over the same seeds).
   So what helps is finishing older workflows before starting newer ones,
   and the critical-path slack term adds nothing measurable. SJF also helps,
   by less.
5. **A perfect estimator makes LSF slightly worse in overload.** LSF with
   the oracle estimator is significantly *worse* than online LSF in 10 of
   40 cells, by 3 to 20s, all at load ≥ 1.25 (largest: batch 4, load 3.00,
   +20.4s [13.8, 27.0]), and never significantly better. This is the
   "oracle is worse" pattern §23 reported and then retracted because §23 was
   accidentally overloaded. Here the overload is deliberate and calibrated,
   so the pattern is real, but only in overload. Not yet explained.
6. **p95 wins do not buy deadlines.** Miss rates are unchanged or slightly
   worse under LSF and EDF (+1 to +5 points, significant in many cells).
   At low load nearly every FCFS miss was feasible at admission (infeasible
   ≤ 1%), so the misses were in principle preventable, but none of the rules
   prevents them. Prioritising old tasks rescues the tail, and newer tasks
   slip past their deadlines in exchange.

### Caveats

- Mac profile. The util-0.8 rate is 0.0265/s at **every** batch size,
  because measured batching is nearly serial (0.841, DECISIONS §28) and a
  bigger cap adds almost no capacity. A GPU that batches efficiently would
  move the KV-bound boundary and the load axis.
- `holdout_qwen` is a fitting set for policy work (DECISIONS §26). These
  numbers guide which policies to freeze. The headline run is the single
  one on `final`, after the freeze.
- Arrivals are a Poisson re-timing of a sequential trace (trace_loader).
- Forced evictions are about 6,000-7,000 blocks per run in every cell
  (LRU under 1024 MB) and do not depend on the scheduler.

### The full table

**batch 2**, slack_factor 1.188 (FCFS miss 24.7% at util 0.8)

| load | choice | at cap | in-flight max/p95 | FCFS p95 | FCFS miss (infeasible) | LSF − FCFS p95 | EDF − FCFS p95 | LSF − FCFS miss | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.78 | 20.7% | 47% | 2/2.0 | 207s | 23% (0%) | **-35.7 [-54.6, -16.8]** | **-37.6 [-53.8, -21.3]** | +0.1 [-1.8, +2.0] pts |  |
| 0.85 | 26.3% | 50% | 2/2.0 | 229s | 26% (0%) | **-40.8 [-61.7, -20.0]** | **-41.8 [-59.7, -23.8]** | +0.9 [-1.1, +3.0] pts |  |
| 1.00 | 37.6% | 54% | 2/2.0 | 296s | 34% (1%) | **-63.8 [-88.6, -39.1]** | **-63.3 [-87.4, -39.1]** | +1.9 [-0.1, +4.0] pts |  |
| 1.10 | 44.5% | 58% | 2/2.0 | 343s | 39% (2%) | **-81.9 [-111.6, -52.2]** | **-78.5 [-108.0, -49.1]** | +2.0 [-0.6, +4.6] pts |  |
| 1.22 | 54.7% | 63% | 2/2.0 | 396s | 46% (3%) | **-95.8 [-129.6, -62.1]** | **-91.0 [-123.6, -58.5]** | +2.0 [-0.3, +4.4] pts |  |
| 1.25 | 57.8% | 64% | 2/2.0 | 410s | 48% (3%) | **-96.7 [-130.3, -63.2]** | **-92.7 [-124.5, -60.9]** | **+2.4 [+0.4, +4.5]** pts |  |
| 1.50 | 75.3% | 69% | 2/2.0 | 581s | 63% (10%) | **-147.7 [-187.4, -108.0]** | **-144.6 [-181.7, -107.5]** | **+4.4 [+2.8, +6.0]** pts |  |
| 1.75 | 81.8% | 70% | 2/2.0 | 753s | 72% (20%) | **-189.8 [-220.4, -159.3]** | **-189.4 [-218.5, -160.3]** | -0.2 [-2.2, +1.8] pts |  |
| 2.20 | 88.8% | 72% | 2/2.0 | 948s | 79% (34%) | **-197.7 [-221.0, -174.5]** | **-198.4 [-221.7, -175.1]** | **+1.5 [+0.2, +2.8]** pts |  |
| 3.00 | 93.1% | 74% | 2/2.0 | 1161s | 84% (46%) | **-196.0 [-210.1, -182.0]** | **-198.5 [-213.3, -183.7]** | +0.6 [-0.7, +1.8] pts |  |

**batch 4**, slack_factor 1.188 (FCFS miss 24.1% at util 0.8)

| load | choice | at cap | in-flight max/p95 | FCFS p95 | FCFS miss (infeasible) | LSF − FCFS p95 | EDF − FCFS p95 | LSF − FCFS miss | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.78 | 4.8% | 0% | 4/3.0 | 191s | 22% (0%) | **-17.2 [-26.7, -7.7]** | **-19.7 [-29.7, -9.7]** | +0.8 [-0.5, +2.2] pts | inactive |
| 0.85 | 7.0% | 1% | 4/3.0 | 209s | 26% (0%) | **-21.7 [-34.3, -9.1]** | **-21.5 [-35.1, -8.0]** | +0.7 [-0.5, +2.0] pts |  |
| 1.00 | 14.7% | 1% | 4/3.0 | 265s | 33% (0%) | **-39.0 [-53.7, -24.2]** | **-40.5 [-55.8, -25.2]** | **+2.0 [+0.7, +3.4]** pts |  |
| 1.10 | 20.0% | 1% | 4/3.0 | 311s | 38% (0%) | **-52.1 [-76.2, -27.9]** | **-55.1 [-79.6, -30.5]** | **+2.7 [+1.0, +4.4]** pts |  |
| 1.22 | 27.4% | 1% | 4/3.0 | 352s | 46% (1%) | **-64.9 [-94.5, -35.2]** | **-64.1 [-91.8, -36.4]** | **+1.9 [+0.2, +3.5]** pts |  |
| 1.25 | 29.7% | 1% | 4/3.0 | 364s | 47% (1%) | **-65.6 [-96.3, -35.0]** | **-64.1 [-95.2, -33.0]** | **+3.0 [+1.2, +4.7]** pts |  |
| 1.50 | 52.5% | 1% | 4/3.0 | 509s | 60% (5%) | **-108.4 [-141.3, -75.5]** | **-113.9 [-149.5, -78.2]** | **+4.7 [+3.4, +6.0]** pts |  |
| 1.75 | 66.8% | 0% | 4/3.0 | 678s | 70% (13%) | **-169.9 [-193.9, -145.9]** | **-174.5 [-203.7, -145.3]** | **+3.6 [+2.2, +5.0]** pts |  |
| 2.20 | 77.9% | 1% | 4/3.0 | 879s | 78% (27%) | **-191.9 [-214.2, -169.5]** | **-199.2 [-222.5, -175.9]** | **+2.6 [+1.5, +3.7]** pts |  |
| 3.00 | 86.0% | 0% | 4/3.0 | 1089s | 84% (40%) | **-196.1 [-213.2, -179.1]** | **-207.8 [-222.6, -193.1]** | **+1.9 [+1.0, +2.9]** pts |  |

**batch 8**, slack_factor 1.188 (FCFS miss 22.8% at util 0.8)

| load | choice | at cap | in-flight max/p95 | FCFS p95 | FCFS miss (infeasible) | LSF − FCFS p95 | EDF − FCFS p95 | LSF − FCFS miss | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.78 | 0.2% | 0% | 6/4.0 | 186s | 23% (0%) | -1.1 [-4.1, +1.8] | -1.1 [-4.0, +1.9] | +0.1 [-0.1, +0.3] pts | inactive, cap never binds |
| 0.85 | 0.5% | 0% | 6/4.2 | 207s | 27% (0%) | -1.2 [-3.5, +1.1] | -0.3 [-1.7, +1.1] | -0.1 [-0.4, +0.2] pts | inactive, cap never binds |
| 1.00 | 2.2% | 0% | 7/4.5 | 251s | 34% (0%) | -5.4 [-12.9, +2.1] | -4.9 [-11.7, +1.9] | +0.4 [-0.1, +0.8] pts | inactive, cap never binds |
| 1.10 | 3.8% | 0% | 7/4.7 | 288s | 39% (0%) | **-15.4 [-28.5, -2.2]** | **-12.6 [-23.1, -2.0]** | **+1.1 [+0.3, +1.9]** pts | inactive, cap never binds |
| 1.22 | 7.1% | 0% | 6/4.8 | 338s | 46% (0%) | **-29.7 [-46.3, -13.1]** | **-32.8 [-51.1, -14.6]** | +0.9 [-0.2, +2.0] pts | cap never binds |
| 1.25 | 8.2% | 0% | 6/4.8 | 344s | 47% (0%) | **-28.3 [-45.1, -11.6]** | **-26.9 [-43.0, -10.8]** | +1.4 [-0.0, +2.8] pts | cap never binds |
| 1.50 | 21.4% | 0% | 8/5.0 | 475s | 62% (3%) | **-56.0 [-76.1, -36.0]** | **-59.5 [-81.1, -37.9]** | **+2.5 [+0.9, +4.1]** pts |  |
| 1.75 | 38.6% | 0% | 7/5.1 | 633s | 71% (6%) | **-102.6 [-122.9, -82.3]** | **-103.7 [-125.7, -81.7]** | **+1.7 [+0.7, +2.6]** pts | cap never binds |
| 2.20 | 54.5% | 0% | 7/5.0 | 834s | 79% (17%) | **-148.5 [-167.1, -129.9]** | **-154.7 [-171.6, -137.8]** | **+1.9 [+1.0, +2.9]** pts | cap never binds |
| 3.00 | 66.0% | 0% | 7/5.1 | 1024s | 85% (31%) | **-151.1 [-165.2, -137.0]** | **-157.1 [-173.6, -140.6]** | **+2.1 [+1.3, +3.0]** pts | cap never binds |

**batch 16 (KV-bound)**, slack_factor 1.188 (FCFS miss 22.8% at util 0.8)

| load | choice | at cap | in-flight max/p95 | FCFS p95 | FCFS miss (infeasible) | LSF − FCFS p95 | EDF − FCFS p95 | LSF − FCFS miss | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.78 | 0.1% | 0% | 8/4.0 | 185s | 23% (0%) | -0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] pts | inactive, cap never binds |
| 0.85 | 0.1% | 0% | 8/4.2 | 205s | 27% (0%) | -0.0 [-0.1, +0.0] | -0.0 [-0.1, +0.0] | +0.0 [+0.0, +0.0] pts | inactive, cap never binds |
| 1.00 | 0.1% | 0% | 10/5.0 | 253s | 34% (0%) | +0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] pts | inactive, cap never binds |
| 1.10 | 0.2% | 0% | 10/5.4 | 287s | 39% (0%) | -1.1 [-3.3, +1.2] | -1.1 [-3.4, +1.2] | +0.0 [+0.0, +0.0] pts | inactive, cap never binds |
| 1.22 | 0.4% | 0% | 10/6.1 | 334s | 47% (0%) | -1.2 [-3.8, +1.4] | -1.5 [-4.8, +1.8] | +0.0 [+0.0, +0.0] pts | inactive, cap never binds |
| 1.25 | 0.5% | 0% | 10/6.2 | 342s | 48% (0%) | -0.8 [-3.7, +2.1] | -1.2 [-3.6, +1.1] | +0.2 [-0.2, +0.6] pts | inactive, cap never binds |
| 1.50 | 2.7% | 0% | 11/7.7 | 467s | 64% (0%) | -4.9 [-10.7, +0.9] | **-5.7 [-10.9, -0.5]** | +0.2 [-0.1, +0.5] pts | inactive, cap never binds |
| 1.75 | 8.0% | 0% | 11/8.7 | 618s | 72% (2%) | **-24.2 [-40.1, -8.3]** | **-24.5 [-40.3, -8.8]** | +0.3 [-0.1, +0.7] pts | cap never binds |
| 2.20 | 19.7% | 0% | 12/9.1 | 808s | 82% (6%) | **-52.3 [-65.0, -39.6]** | **-53.8 [-68.2, -39.4]** | +0.2 [-0.1, +0.5] pts | cap never binds |
| 3.00 | 34.0% | 0% | 11/9.1 | 1007s | 88% (15%) | **-87.4 [-99.9, -74.8]** | **-90.5 [-103.1, -77.9]** | +0.1 [-0.1, +0.3] pts | cap never binds |

Bold = paired 95% CI excludes zero. Seconds unless marked pts (percentage points of deadline-miss rate).

- batch 2: lsf first separates at load 0.78 on p95_vs_fcfs: p95 -35.7s [-54.6, -16.8], miss +0.1% [-1.8%, +2.0%]
- batch 2: edf first separates at load 0.78 on p95_vs_fcfs: p95 -37.6s [-53.8, -21.3], miss -1.6% [-3.2%, +0.0%]
- batch 4: lsf first separates at load 0.78 on p95_vs_fcfs: p95 -17.2s [-26.7, -7.7], miss +0.8% [-0.5%, +2.2%]
- batch 4: edf first separates at load 0.78 on p95_vs_fcfs: p95 -19.7s [-29.7, -9.7], miss +0.5% [-0.5%, +1.4%]
- batch 8: lsf first separates at load 1.10 on p95_vs_fcfs, miss_vs_fcfs: p95 -15.4s [-28.5, -2.2], miss +1.1% [+0.3%, +1.9%]
- batch 8: edf first separates at load 1.10 on p95_vs_fcfs, miss_vs_fcfs: p95 -12.6s [-23.1, -2.0], miss +0.9% [+0.3%, +1.6%]
- batch 16: cap never binds at any load (KV-bound: the KV budget, not max_batch, limits concurrency)
- batch 16: lsf first separates at load 1.75 on p95_vs_fcfs: p95 -24.2s [-40.1, -8.3], miss +0.3% [-0.1%, +0.7%]
- batch 16: edf first separates at load 1.50 on p95_vs_fcfs: p95 -5.7s [-10.9, -0.5], miss +0.2% [-0.1%, +0.5%]

---

## 30. With per-task deadlines, nothing beats oldest-task-first

The test §29 called for: give tasks heterogeneous deadlines, so EDF and LSF
are no longer just task age, and compare them with the program-level
baseline instead of request-level FCFS.

**Setup.** Same map as §29 (commit `92490cd`, clean tree, profile
`mac_ollama_measured`, coefficient 0.841, `holdout_qwen` 1x, KV 1024 MB,
LRU, 20 paired seeds), run with `--slo-mix
tight:1.0:0.3,standard:2.0:0.4,loose:4.0:0.3`. Output
`results/scheduling_map_slo.json`, figures
`results/figures/regime/scheduling_slo_*.png`.

- **Deadlines:** arrival + slack_factor × class multiplier × size scale ×
  isolated time. The class is drawn per task, the same draw for every
  policy. The size scale is the first prompt against its stage's typical
  prompt, which is visible at arrival. slack_factor 1.875 was calibrated
  so FCFS misses 22-27% at util 0.8.
- **Two new rules:**
  - `program-fcfs`: oldest task first, no estimates.
  - `hint-priority`: the client-declared class first, then task age. This
    mimics NVIDIA Dynamo's agent hints, where the client sets a static
    priority and the server computes no slack.

### Result: 40 cells, paired against `program-fcfs`

| rule | p95 significantly better / worse | overall miss significantly better / worse | tight-class miss, mean change (range) |
|---|---|---|---|
| LSF | 0 / 28 | 0 / 22 | −2.8 pts (−12.7 to +0.3) |
| EDF | 0 / 27 | 1 / 17 | −3.3 pts (−13.0 to +0.3) |
| hint-priority | 0 / 28 | 3 / 13 | −5.9 pts (−28.5 to +0.3) |
| LSF-oracle | 0 / 29 | 0 / 26 | −3.0 pts |

1. **No deadline-aware rule beats oldest-task-first on p95, in any cell.**
   They are significantly worse in about 70% of cells. Where the queue is
   real (batch 2-4), LSF is +9 to +58s and hint-priority up to +304s.
2. **They do not reduce total misses either.** What they do is move misses
   between classes: tight-class misses fall a few points (hint-priority by
   up to 28), and loose-class misses rise to pay for it. That is a
   reallocation, not a gain.
3. **A perfect output-length estimator does not help**: LSF-oracle is no
   better than LSF.
4. **Against request-level FCFS the §29 picture repeats:** every rule
   separates at the lowest load tested at batch 2-4, at load 1.10 at
   batch 8, and at 1.5-1.75 at batch 16. `program-fcfs` is the largest
   single gain everywhere.

### Why, and the caveat that limits it

The tight class misses about 53% of its deadlines under every rule, even at
load 1.0 (batch 4: FCFS 53%, program-fcfs 54%, LSF 53%). With nearly serial
batching (DECISIONS §28), a tight task is close to infeasible once it shares
the GPU at all, so there is little for a smarter rule to save. Meanwhile
reordering for deadlines breaks up older workflows, and that is what the
p95 pays for. The remaining-work estimate is still per stage, not per task
(the open item from §29), so this tests heterogeneous deadlines, not better
estimates. Mac-only: a GPU with efficient batching would leave tight tasks
feasible more often, and that is the regime where this verdict could change.

**Consequence for the writeup and the freeze.** The supported scheduling
claim is **program-level ordering**: finish older workflows first. That
gives a 15-22% p95 cut at the operating load whenever requests queue, and
needs no estimates. On this hardware, neither slack (LSF) nor client
priority hints (Dynamo-style) improve on it.

## 31. What-if: efficient batching (0.15, HYPOTHETICAL)

**This is not a GPU measurement.** The question: §30's verdict could be an
artefact of the Mac's nearly serial batching (coefficient 0.841, DECISIONS
§28). Under vLLM-class batching, does LSF start to beat `program-fcfs`? The
answer decides whether GPU calibration time is worth spending on scheduling.

**Setup.** Same map as §30 (commit `70ae06d`, clean tree, profile
`mac_ollama_measured` with `--batch-decode-coeff 0.15`, recorded in the
manifest's cost provenance as `OVERRIDDEN from profile value 0.841`, and
every calibration at 0.15; `holdout_qwen` 1x, KV 1024 MB, LRU, 20 paired
seeds, the same `--slo-mix`). Rates and slack were recalibrated: the
util-0.8 rate is 0.0304/s at batch 2 and 0.0343/s at batch 4-16, against
0.0265/s everywhere on the Mac. Output `results/scheduling_map_slo_c015.json`,
figures `results/figures/regime/scheduling_slo_c015_{p95,miss}[_vs_program].png`.

### Result: 40 cells, paired against `program-fcfs`

| rule | p95 significantly better / worse | overall miss significantly better / worse | tight-class miss, mean change (range) |
|---|---|---|---|
| LSF | 0 / 22 | 0 / 17 | −3.1 pts (−13.7 to +0.5) |
| EDF | 0 / 19 | 2 / 9 | −3.5 pts (−14.2 to +0.5) |
| hint-priority | 0 / 22 | 5 / 5 | −5.9 pts (−29.5 to +0.5) |
| LSF-oracle | 2 / 22 | 1 / 19 | −3.0 pts |

1. **LSF still never beats oldest-task-first on p95.** It is significantly
   worse in 22 of 40 cells, against 28 on the Mac; the drop comes from more
   cells going inactive, not from LSF improving. The oracle's two wins (batch 8,
   load 1.10 and 1.23) are 3-4s, with CIs that barely exclude zero.
2. **The tight-class gain is the same reallocation as §30.** At batch 2, LSF
   cuts tight-class misses by 9-14 points, but loose-class misses rise by 6-30
   points and the overall miss rate rises in 8 of those 9 cells (−0.7 to
   +5.1 points).
3. **Efficient batching removes the queue rather than making room for slack.**
   The cap never binds in 9 of 10 cells at batch 8 or in any cell at batch 16.
   Inactive cells (choice on under 3% of decisions) rise from 1/4/7 (§30,
   batch 4/8/16) to 2/6/9. At load 1.00, `program-fcfs` has a choice on 0.3%
   of decisions at batch 8 and 0.1% at batch 16.

> **Qualified (2026-10-08, OPEN_ISSUES §28):** point 3's cap-binding counts
> came from the active-model batch, which undercounts. The cap was enforced on
> resident requests across both models. "The cap never binds" may be wrong; the
> choice fractions are unaffected. Don't quote point 3 until the map is rerun.

**Tight-class feasibility.** The tight class still misses 53-60% of its
deadlines under `program-fcfs` at load 1.00 (batch 2/4/8/16: 53.5%, 60.2%,
57.0%, 56.5%), against about 53% on the Mac. Caveat: slack is recalibrated so
that FCFS misses 22-27% at util 0.8, so the deadlines tighten along with the
faster service (slack_factor 1.19 at batch 4-16 and 1.53 at batch 2, against
1.875). This test therefore holds relative deadline pressure fixed, as §30
did. It does not test "the same absolute deadlines on faster hardware".

**Verdict:** program-level ordering holds under efficient batching too:
freeze `program-fcfs`.

## 32. Sealed `final_qwen`, partial: program-level ordering holds out of sample

> **Superseded (2026-10-22) by §33**, which reports the full 40-cell run
> and `final_llama`; the five cells below are unchanged in it.
>
> **Partial (2026-10-15/16).** 5 of 40 cells, batch 2 only. The run
> stopped when the laptop slept on battery and the session ended; it was not
> stopped on a result. The sealed retention run and the `final_llama` map did
> not start. Finishing reruns the same freeze commit on the remaining cells.

**Setup.** Freeze commit `f4c889b` (DECISIONS §31 and its amendment), run 1 as
pre-registered: `scheduling_map --trace-set final_qwen --slo-mix
tight:1.0:0.3,standard:2.0:0.4,loose:4.0:0.3`, 20 paired seeds, KV 1024 MB,
LRU, coefficient 0.841, 91 tasks (HumanEval/75 excluded, DECISIONS §31).
Calibration was normal: util 0.6-0.9 landed at busy 0.60-0.92, slack_factor
1.875 (FCFS miss 26.7% at util 0.8). Log, force-added because a sealed run is
one-shot: `results/sealed/scheduling_map_final_qwen.log`. No JSON was written.

Paired p95 differences, seconds [95% CI]; * excludes zero:

| load | FCFS p95 | `program-fcfs` vs FCFS | LSF vs `program-fcfs` | EDF vs `program-fcfs` |
|---|---|---|---|---|
| 0.71 | 212.6 | −29.1 [−43.3, −14.9]* (−14%) | +18.5 [+7.0, +30.0]* | +12.2 [+2.0, +22.5]* |
| 0.85 | 287.5 | −66.8 [−91.0, −42.6]* (−23%) | +38.5 [+27.1, +49.9]* | +27.8 [+16.0, +39.7]* |
| 1.00 | 397.9 | −112.4 [−145.8, −79.0]* (−28%) | +48.7 [+36.9, +60.4]* | +47.0 [+35.7, +58.4]* |
| 1.10 | 492.0 | −151.2 [−191.8, −110.6]* (−31%) | +52.3 [+37.8, +66.8]* | +46.5 [+31.6, +61.5]* |
| 1.15 | 557.6 | −183.8 [−225.8, −141.8]* (−33%) | +56.5 [+43.9, +69.2]* | +49.9 [+35.0, +64.9]* |

Against the pre-registered comparisons:

- **S1 holds:** `program-fcfs` beats FCFS on p95 in all 5 cells, and its miss
  rate is lower in all 5 (−1.5 to −3.0 points, each significant).
- **S2 holds, more strongly than expected:** LSF is not merely never better
  than `program-fcfs`; it is significantly worse in all 5 cells.
- **S3 not testable at batch 2:** `program-fcfs-cap`'s cap is 3, above the
  batch limit, so it equals `program-fcfs` exactly in every cell.
- `hint-priority` is significantly worse than `program-fcfs` on p95 in all 5
  cells (+33 to +267 s).

At load 1.00, LSF and EDF cut p95 by 16% against FCFS, inside §29's 15-22%
range on the fitting set, so nothing here suggests the holdout result was
overfit. Batches 4-16 are
untested out of sample.

## 33. Sealed `final` set, complete: program-level ordering holds out of sample

> **Current (2026-10-22).** Supersedes §32. Runs 1 and 3 of DECISIONS §31,
> each run once from freeze commit `f4c889b`, clean tree, reported as they
> came out.

**Setup.** As §32: `scheduling_map --slo-mix
tight:1.0:0.3,standard:2.0:0.4,loose:4.0:0.3`, 20 paired seeds, KV 1024 MB,
LRU, profile `mac_ollama_measured` (coefficient 0.841), busy load axis,
batches 2/4/8/16 × 10 loads = 40 cells per trace. `final_qwen` has 91 tasks
(HumanEval/75 excluded, §31 amendment); `final_llama` has 92. Results,
force-added because a sealed run is one-shot:
`results/sealed/scheduling_map_final_qwen_full.{json,log}` (5,823 s) and
`results/sealed/scheduling_map_final_llama.{json,log}`.

Cells where the rule is significantly better / worse / tied on p95 task
latency (paired 95% CI over 20 seeds), by batch 2 / 4 / 8 / 16:

| comparison | `final_qwen` | `final_llama` | both, of 80 |
|---|---|---|---|
| `program-fcfs` vs FCFS | 10/0/0, 10/0/0, 8/0/2, 6/0/4 | 10/0/0, 10/0/0, 10/0/0, 8/0/2 | **72 / 0 / 8** |
| LSF vs `program-fcfs` | 0/10/0, 0/9/1, 0/6/4, 0/4/6 | 0/10/0, 0/10/0, 0/9/1, 0/3/7 | **0 / 61 / 19** |
| LSF-oracle vs `program-fcfs` | 0/30/10 in total | 0/33/7 in total | 0 / 63 / 17 |
| EDF vs `program-fcfs` | 0/28/12 in total | 0/31/9 in total | 0 / 59 / 21 |
| `hint-priority` vs `program-fcfs` | 0/31/9 in total | 0/36/4 in total | 0 / 67 / 13 |

`program-fcfs` vs FCFS at load 1.00, seconds [95% CI] (change in FCFS p95):

| batch | `final_qwen` | `final_llama` |
|---|---|---|
| 2 | −112.4 [−145.8, −79.0] (−28%) | −267.4 [−328.5, −206.2] (−36%) |
| 4 | −60.5 [−83.9, −37.0] (−17%) | −190.5 [−241.7, −139.4] (−28%) |
| 8 | −25.9 [−44.2, −7.5] (−8%) | −112.8 [−166.9, −58.6] (−17%) |
| 16 | −2.1 [−5.7, +1.5] (−1%, n.s.) | −28.8 [−51.8, −5.8] (−4%) |

1. **S1 holds on both traces.** `program-fcfs` is never significantly worse
   than FCFS, and better in 72 of 80 cells. Largest cuts: −38% (`final_qwen`,
   batch 2, load 1.50) and −40% (`final_llama`, batch 2, load 1.25). The 8
   ties are at batch 8–16 below load 1.15, where the scheduler rarely has a
   choice (cells flagged inactive).
2. **S2 holds.** LSF is better than `program-fcfs` in 0 of 80 cells and worse
   in 61. Oracle output lengths do not change this (0 better, 63 worse), so
   the loss is not estimator error.
3. **The gain shrinks with batch size**, as on the fitting set (§29): a larger
   batch drains the queue that ordering acts on.
4. **Same pattern on unseen tasks.** The holdout map at the same setup
   (`results/scheduling_map_slo_cap.json`, commit `404e44d`) gives 31/0/9 for
   `program-fcfs` vs FCFS and 0/28/12 for LSF vs `program-fcfs`.

**Verdict:** program-level ordering is the result; slack-based ordering is a
negative result, out of sample, on two coder models. Mac costs only; on GPU
costs, see DECISIONS §33.

## 34. Concurrency cap (pre-registered S3): helps at batch 8–16, costs ~5% in overload at batch 4

> **Current (2026-10-22).** S3 of DECISIONS §31, first evaluated on the
> sealed set. No expectation was registered.

**Rule.** `program-fcfs-cap` (`AdaptiveCapProgramFCFS`, EPS 0.05, frozen)
admits the smallest k with `mu(k) >= 0.95 mu(max_batch)`, where
`mu(k) = k / (1 + (k-1)c)`. **k depends only on the batching coefficient c**:
at 0.841 it is 2 at batch 4 and 3 at batch 8–16. At batch 2 it equals
`program-fcfs` exactly. Sources: the §33 sealed JSONs and the holdout
`results/scheduling_map_slo_cap.json`.

Cap vs `program-fcfs` on p95, significantly better / worse / tied:

| batch (k) | `final_qwen` | `final_llama` | holdout `qwen` |
|---|---|---|---|
| 4 (2) | 2 / 6 / 2 | 2 / 5 / 3 | 0 / 6 / 4 |
| 8 (3) | 6 / 3 / 1 | 8 / 1 / 1 | 4 / 2 / 4 |
| 16 (3) | **10 / 0 / 0** | **10 / 0 / 0** | 9 / 1 / 0 |
| sealed, batch 4–16 | | | **38 / 15 / 7 of 60** |

1. **Batch 16: better in every sealed cell.** The change in p95 is −10% to
   −16% (`final_qwen`) and −20% to −33% (`final_llama`) up to load 1.5,
   narrowing to −2% to −7% at load 3.0. Largest: −363.6 s [−409.1, −318.2]
   (`final_llama`, load 1.50).
2. **Batch 8: better below deep overload.** The change is −4% to −9%
   (`final_qwen`, load ≤ 1.25) and −1% to −17% (`final_llama`, load ≤ 1.75),
   becoming +2% worse at load ≥ 1.75 (`final_qwen`) and 3.0 (`final_llama`).
3. **Batch 4: worse in overload.** At load 0.74–0.85 it is −5% to −6% better.
   From load 1.15–1.25 up it is +3% to +6% worse. Interpretation, not
   measured: capping 4 to 2 gives up `1 - mu(2)/mu(4)` = 4.3% of
   throughput, and in overload throughput sets the tail, so the loss matches
   the throughput given up.
4. The holdout run shows the same signs, with fewer cells separating.

**Verdict:** a useful mechanism at large batch sizes on a nearly serial
batcher, and a regression at batch 4 under overload. A load-aware version was
tried before the freeze and removed (the docstring explains why), so it isn't
claimed here. Because k is a function of c, the cap's value on a better
batcher is untested; that is H3 of DECISIONS §33.

## 35. Retention at 8x context: retention matters, ski-rental still loses to keep-all

> **Preliminary (2026-10-22).** Holdout (fitting) set, 10 seeds, as
> DECISIONS §31 requires for 8x. Not sealed. 8x is a simulator-only context
> scale (GPU_TODO).

**Setup.** `run_retention_grid --trace-set holdout_qwen --scales 8
--headrooms 0.25,0.5,1.0 --util 0.8 --batch 4 --seeds 10`, FCFS, profile
`mac_ollama_measured`, with the calibration bracket scaled by context
(`87d21da`). The budget is the active peak (2,227 MB) plus the headroom
times the reusable working set (1,274 MB). Output:
`results/retention_headroom_8x.{json,log}` (4,470 s).

Paired change against keep-all, seconds [95% CI]; * excludes zero. Keep-all
p95 is 2,191–2,210 s, and its recompute is 185–365 s:

| headroom (KV MB) | evict-now p95 | ski-rental p95 | ski-rental recompute | offline-ref p95 |
|---|---|---|---|---|
| 0.25 (2,545) | +222.1 [+127.5, +316.7]* (+10.0%) | +171.2 [+69.0, +273.3]* (+7.7%) | +164.1* | +65.2* |
| 0.5 (2,864) | +234.8 [+143.1, +326.5]* (+10.7%) | +95.7 [+19.4, +172.0]* (+4.4%) | +118.4* | +49.8 |
| 1.0 (3,501) | +241.3 [+145.5, +337.2]* (+11.0%) | −2.4 [−31.2, +26.5] | +136.1* | −11.8 |

Contrast, the sealed 1x run (DECISIONS §31 run 2,
`results/sealed/retention_headroom_final_qwen.json`, 20 seeds, budgets
418–563 MB). Evict-now is +31.0 to +44.8 s on p95 (each significant).
Ski-rental ties keep-all on p95 (−2.8 to +3.5 s) but recomputes +9.9 to
+16.6 s more in every cell. Ski-rental-pred does the same (+9.3 to +17.1 s).
**R1 holds:** neither variant is better than keep-all on recompute or on p95.

1. **At 8x, retention matters.** Dropping everything (evict-now) costs 10–11%
   on p95 and 648–828 s of recompute. At 1x it cost about 8–12% on p95 with
   about a sixth of the recompute (+114 to +139 s).
2. **Ski-rental's releases are the wrong ones.** It recomputes 118–164 s more
   than keep-all at every budget, and is significantly worse on p95 at the two
   tighter budgets (+7.7%, +4.4%). The learning-augmented variant is no
   better (+232.2*, +101.2*, +1.7 s).
3. **LRU equals keep-all in every cell**, and the offline reference loses to
   it at the tightest budget (+65.2 s*), consistent with §26.

**Verdict:** the regime FINDINGS §7 predicted, where retention matters, does
appear at 8x. Rent-to-buy release still doesn't beat holding KV until memory
pressure forces eviction. Keep-all/LRU is best at every budget tested.

## 36. Batching-cost sweep, pass 1 (Mac rates): the conclusions do not depend on the coefficient

> **Current (2026-10-23).** Pass 1 of DECISIONS §33, scored exactly as
> pre-registered. The coefficient is swept, not measured: 0.841 is the Mac's
> MEASURED value and 0.5, 0.3, 0.15 are OVERRIDDEN what-ifs. Pass 2 (the same
> runs on G1 GPU rates) waits for the GPU calibration file.

**Setup.** As DECISIONS §33: `experiments/run_gpu_sweep.sh` at `cbe313c`
(the commit after the pre-registration `d06b932`), clean tree in every
manifest. `holdout_qwen`, profile `mac_ollama_measured`, `--load-axis
capacity` (load 1.0 = saturation throughput at that batch and coefficient),
the SLO mix of §30, 20 paired seeds, 40 cells per coefficient. Outputs:
`results/gpu_sweep/mac_ollama_measured/` (map runs 1,441–1,590 s each;
`sweep_decode_coeff` 610 s).

Cells significantly better / worse / tied on p95 (40 per coefficient):

| comparison | 0.841 | 0.5 | 0.3 | 0.15 |
|---|---|---|---|---|
| `program-fcfs` vs FCFS (H1) | 28 / **0** / 12 | 29 / **0** / 11 | 29 / **0** / 11 | 29 / **0** / 11 |
| LSF vs `program-fcfs` (H2) | **0** / 25 / 15 | **0** / 27 / 13 | **0** / 25 / 15 | **0** / 27 / 13 |
| LSF-oracle vs `program-fcfs` | 0 / 24 / 16 | 0 / 26 / 14 | 0 / 26 / 14 | 0 / 27 / 13 |
| cap vs `program-fcfs`, batch 16 (H3) | **9** / 0 / 1 | 6 / 0 / 4 | 4 / 0 / 6 | **4** / 0 / 6 |
| cap k at batch 4 / 8 / 16 | 2 / 3 / 3 | 4 / 6 / 9 | 4 / 7 / 12 | 4 / 8 / 14 |

At load 1.0, the change in p95 for `program-fcfs` vs FCFS, by batch
2 / 4 / 8 / 16: −22 / −19 / −9 / −0% at 0.841, −24 / −19 / −8 / −0% at 0.5,
−26 / −19 / −10 / −1% at 0.3, and −25 / −19 / −10 / −1% at 0.15 (batch 16 is
n.s. throughout).

Retention point (`sweep_decode_coeff`: batch 4, KV 512 MB, util 0.8 on the
busy axis, recalibrated per coefficient), ski-rental minus keep-all:

| coefficient | keep-all recompute, % of makespan | p95, s | recompute, s |
|---|---|---|---|
| 0.841 | 0.7% | +0.57 [−7.02, +8.17] | +10.18 [+4.77, +15.59]* |
| 0.5 | 0.7% | +3.30 [−2.36, +8.95] | +10.08 [+5.85, +14.32]* |
| 0.3 | 0.6% | +7.49 [+1.41, +13.56]* | +11.75 [+7.43, +16.07]* |
| 0.15 | 0.4% | +3.14 [+1.26, +5.01]* | +13.48 [+8.75, +18.22]* |

Scored:

- **H1 held.** `program-fcfs` is worse than FCFS in 0 of 160 cells. On the
  capacity axis its gain is nearly identical at every coefficient: −19% at
  batch 4 and −22% to −26% at batch 2, at load 1.0.
- **H2 held.** LSF beats `program-fcfs` in 0 of 160 cells (limit: 2 per
  coefficient), and LSF-oracle in 0 of 160.
- **H3 held.** The cap is better at batch 16 in 9 cells at 0.841 and 4 at
  0.15. Its gains there shrink from −6% to −12% (0.841, loads 0.5–1.5) to
  −5% to −8% (0.5), −1% to −5% (0.3) and −1% to −2% (0.15). This is
  partly by construction: k rises toward `max_batch` as the coefficient
  falls, and at 0.15 the cap equals `program-fcfs` exactly at batch 4–8.
- **H4 held.** Recompute stays at 0.4–0.7% of makespan, and ski-rental is
  never better than keep-all on p95. It is significantly worse at 0.3 and
  0.15, and recomputes 10–13 s more at every coefficient.

Exploratory, not pre-registered:

1. **Batch 4 under overload depends on hardware.** The cap's batch-4 losses
   (1 better, 5 worse at 0.841, the §34 pattern) disappear at 0.5 and below,
   because k reaches 4 and the cap stops binding.
2. **EDF beats `program-fcfs` in one cell** at 0.5 and one at 0.15, against
   25–26 cells worse. That is within the about 1 in 20 expected by chance
   (§31).

**Verdict:** across the plausible range of batching cost, from the Mac's
nearly serial 0.841 to vLLM-class 0.15, program-level ordering keeps its
gain, LSF never helps, and ski-rental never beats keep-all. Only the cap's
value is hardware-dependent, and it fades as batching gets cheaper. A future
measured GPU coefficient only picks a column of this table.

## 37. Batching-cost sweep, pass 2a (G1 GPU rates): ordering holds, the cap does not carry over

> **Provisional (2026-10-23).** Pass 2a of DECISIONS §33 (amendment), scored as
> pre-registered. Rates are the RTX 5060 Ti G1 run-2 values (qwen
> 3021.98/62.78, llama3.2 4001.12/98.25 tok/s), passed with `--rates` because
> the box is unreachable, and recorded as OVERRIDDEN. The coefficient is swept,
> not measured. This is superseded if the committed `measured.json` differs.

**Setup.** As §36, with profile `gpu_5060ti` and `--rates`, at `4b15417`,
clean tree in every manifest. Outputs: `results/gpu_sweep/gpu_5060ti_g1rates/`.
Saturation throughput is 2.7–3.8× the Mac's at the same batch and coefficient
(for example 0.095 against 0.034 tasks/s at batch 16 and 0.841). Caveats:
- CPU-side stages (RAG, pytest) keep Mac timings.
- The rates came from a card that later failed (Xid 79), and qwen decode
  differed 13% between G1 runs 1 and 2.

Cells significantly better / worse / tied on p95 (40 per coefficient):

| comparison | 0.841 | 0.5 | 0.3 | 0.15 |
|---|---|---|---|---|
| `program-fcfs` vs FCFS (H1) | 29 / **0** / 11 | 30 / **0** / 10 | 29 / **0** / 11 | 30 / **0** / 10 |
| LSF vs `program-fcfs` (H2) | **0** / 24 / 16 | **0** / 29 / 11 | **0** / 26 / 14 | **0** / 29 / 11 |
| LSF-oracle vs `program-fcfs` | 0 / 24 / 16 | 0 / 26 / 14 | 0 / 28 / 12 | 0 / 28 / 12 |
| cap vs `program-fcfs`, batch 16 (H3) | **4** / 1 / 5 | 4 / 1 / 5 | 0 / 1 / 9 | **0** / 2 / 8 |
| cap vs `program-fcfs`, all batches | 6 / 12 / 22 | 4 / 5 / 31 | 0 / 7 / 33 | 0 / 2 / 38 |

At load 1.0, the change in p95 for `program-fcfs` vs FCFS, by batch
2 / 4 / 8 / 16: −25 / −21 / −10 / −0% at 0.841, −27 / −21 / −8 / −0% at 0.5,
−28 / −20 / −9 / −1% at 0.3, and −27 / −21 / −8 / −1% at 0.15 (batch 16 is
n.s.).

Retention point (`sweep_decode_coeff`): keep-all recompute is **0.1–0.2% of
makespan**, against 0.4–0.7% on Mac rates; the 11× faster prefill makes
recompute nearly free. Ski-rental minus keep-all on p95:

| coefficient | ski-rental − keep-all p95, s | recompute, s |
|---|---|---|
| 0.841 | +0.82* | +1.17* |
| 0.5 | +0.38* | +1.13* |
| 0.3 | −0.23 (n.s.) | +1.29* |
| 0.15 | +0.60* | +1.15* |

Evict-now costs only +2 to +5 s against p95s of 47–101 s.

Scored:

- **H1 held.** `program-fcfs` is worse than FCFS in 0 of 160 cells. Its gain
  matches the Mac's: −20% to −21% at batch 4 and −25% to −28% at batch 2, at
  load 1.0.
- **H2 held.** LSF beats `program-fcfs` in 0 of 160 cells, and so does
  LSF-oracle.
- **H3 held.** The cap is better at batch 16 in 4 cells at 0.841 and 0 at
  0.15.
- **H4 held.** Recompute is 0.1–0.2% of makespan, and ski-rental is never
  better than keep-all on p95.

Exploratory, not pre-registered:

1. **The cap's value depends on the hardware, not only on the coefficient.**
   At the same 0.841, the cap wins 9 batch-16 cells on Mac rates (§36) but 4
   here (−6% to −8%). At batch 4 it is worse in 8 cells (+5% to +18%), and at
   batch 8 it is worse in overload (+8% to +12% at loads 1.25–2.0). On GPU
   rates, below batch 16, the cap is never a net gain: it is worse wherever
   it binds, and identical where k reaches the batch limit.
2. At the sweep's scheduler point (batch 8, 0.841), LSF beats plain FCFS by
   3.09 s [0.55, 5.63]. This is not a gain over `program-fcfs`, which LSF
   never beats in the map.

**Verdict:** on the GPU's measured speeds, at every batching cost from 0.15 to
0.841, the Milestone 2 conclusions hold: oldest-workflow-first keeps a
20–28% p95 cut at batch 2–4, slack scheduling never helps, and retention is
irrelevant at 1x. The concurrency cap (§34) is a Mac-cost result and should
not be claimed for GPU deployment.
