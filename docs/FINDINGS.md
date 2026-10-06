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

## 23. Scheduler table: LSF buys tail latency, and a perfect estimator does not help

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

| policy | recompute | prefill | decode | idle | makespan | recomp/makespan | p95 |
|---|---|---|---|---|---|---|---|
| keep-all | 13.4s | 408.7s | 883.1s | 285.8s | 1603.9s | **0.8%** | 187.6 |
| ski-rental | 18.4s | 413.8s | 882.8s | 283.8s | 1606.7s | **1.1%** | 185.8 |
| evict-now | 127.4s | 522.8s | 856.9s | 207.4s | 1612.4s | 7.9% | 220.3 |
| offline-ref | 29.4s | 424.8s | 878.8s | 286.6s | 1616.3s | 1.8% | 184.7 |

Decode is 55% of makespan, prefill 25%, idle 18%. Avoided recompute is the
*only* thing a retention policy can win, and it is **under 1%**. So the
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
