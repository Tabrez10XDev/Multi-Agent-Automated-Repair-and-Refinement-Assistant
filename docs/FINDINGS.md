# Findings

What the measurements actually showed, including the results that went against the hypothesis. Corrections are kept in place rather than folded in, so the record shows what was believed and when.

> Split out of the single `sim/HANDOFF.md` during the Milestone 2 cleanup.
> **Section numbers are the original HANDOFF numbers and are deliberately not
> renumbered**, so cross-references like "supersedes §3 decision 4" still
> resolve. They are not sequential within a file. The other parts live in
> `ARCHITECTURE.md`, `DECISIONS.md`, `FINDINGS.md` and `OPEN_ISSUES.md`.

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

## 17. The 0.10 similarity threshold, measured from the other side

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
