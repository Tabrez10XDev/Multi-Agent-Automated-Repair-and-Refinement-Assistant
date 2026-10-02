# Simulator rebuild — handoff

Context for an agent picking this up: you know MAARA (the multi-agent coding
harness in this repo). You do not know `sim/`. This explains what it is, what was
measured, what was wrong, and what to do next.

---

## 1. What this is

The user's Master's capstone is **"Adaptive LLM Inference Scheduling"** — a
workflow-aware controller for LLM serving, with two mechanisms:

1. **Ski-rental KV cache retention.** Keep-vs-evict as an online rent-or-buy
   problem. Deterministic 2-competitive, randomized e/(e-1), plus a
   learning-augmented variant using a workflow-graph hint about reuse time.
2. **Least-slack-first scheduling**, where slack comes from the workflow DAG's
   critical path rather than arrival time.

`sim/` is a trace-driven simulator for studying those. It does **not** call any
model at run time. MAARA is the *workload being modelled*, not a dependency.

**A previous version of this simulator was lost and had a fatal flaw:** it
processed one request at a time. Holding a cache entry therefore never blocked
anyone, the "rent" side of ski-rental was always zero, and retention could show
no effect. Avoiding that is the entire point of the rebuild. Any change that
lets the engine drift back toward serial execution defeats the project.

Build order the user specified: **Phase 1 engine skeleton → stop for review →
Phase 2 policies → Phase 3 experiment drivers.** Phase 1 is complete. Phase 2 has
not been started.

---

## 2. Files

| file | contents |
|---|---|
| `sim/workload.py` | `CostModel`, `StageTokens`, `CpuStageSeconds`, `WorkloadConfig`, and `generate_trace()` — unrolls MAARA's DAG into a replayable trace |
| `sim/engine.py` | Continuous-batching engine: shared KV budget, wait queue, prefix cache with refcounts, eviction, rent accounting. Baseline `LRURetention` + `FIFOScheduler` |
| `sim/measure_context.py` | Calibration harness. Measures rather than assumes |
| `sim/measured.json` | Its output. Auto-loaded by `workload.py`; `CostModel.provenance()` prints MEASURED vs PLACEHOLDER |

Run: `./venv/bin/python -m sim.engine --help`

---

## 3. Design decisions — locked by the user, do not silently revise

1. **Contention comes from many concurrent tasks.** Open Poisson arrivals of
   independent MAARA task instances sharing one inference server. A single task's
   DAG is almost fully sequential and can never fill a batch. Intra-task fan-out
   (`generate_fix` loops per file, `coder.py:95`) is a secondary source.
2. **One shared KV pool, per-model rates.** Requests carry a model tag;
   `CostModel.homogeneous=True` collapses everything onto the qwen rates as an
   ablation. The user asked whether to go homogeneous; the answer given was
   build hetero-capable and flip a flag, since `hetero` and `homo_llama3` are
   already arms in `benchmark/results.jsonl`.
3. **Active KV is pinned; only completed prefix entries are evictable.** No
   preemption of running requests. Consequence: the engine reserves prompt +
   full output at admission, because with no preemption a request that runs out
   of KV mid-decode has nowhere to go.
4. **Rent is emergent, never priced.** There is deliberately no per-token memory
   knob. **Do not add a memory-price config value** — it would let the retention
   policy read the answer it is supposed to infer.
   > **REVISED — see §13.** The *no-price* principle stands, but the mechanism
   > ("charge rent while a retained entry keeps a request waiting") has been
   > replaced by derived, eviction-pressure-scaled rent. Do not implement the
   > blocking-only version from this line.
5. **CPU stages consume wall-clock but hold no KV.** `navigate` (ChromaDB +
   nomic-embed-text) and `apply_and_test` (pytest) define the reuse gap.

---

## 4. Measurements

All from `sim/measure_context.py` against live ollama on the user's Mac (Metal).

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

`benchmark/results.jsonl` (121 runs) gives the retry distribution used by the
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

## 8. Bugs already fixed — do not reintroduce

1. **Double-counted KV.** Full output was reserved at admission *and* grown per
   decoded token, inflating usage ~2x. Growth removed; reservation is correct
   under the no-preemption decision.
2. **Iteration-averaged metrics.** `mean_batch` / `mean_kv_util` averaged per
   iteration, so a 99s prefill and a 0.035s decode step counted equally. Symptom
   was resident reading identically (13.95) across three very different configs.
   Now time-weighted; the KV split moved 407/810 → 532/726MB.
3. **`prefill_sample_tokens` read from the wrong response object** in
   `measure_rates`.
4. **ollama `num_ctx` truncation** (see §4).

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

## 10. Oracle fields — do not leak to online policies

`generate_trace()` samples loop trip counts up front and emits the realized path,
so the trace object literally contains the future. `Node.next_reuse_node`,
`Node.next_reuse_gap_s`, and `Node.critical_path_remaining` are oracle fields.
They exist for offline Belady/OPT and for constructing the *noisy hint* the
learning-augmented ski-rental variant receives. An online policy reading them
directly invalidates the comparison.

---

## 11. Next steps, in order

1. **Chunked + batched prefill.** Blocks everything else quantitative.
2. **Sample stable-context size from the real distribution** (§5) rather than one
   constant.
3. **Rework the rent model** to charge for eviction pressure — needs the user's
   decision first.
4. **Re-baseline at 70–90% utilisation.**
5. **Then Phase 2**: `sim/policies.py` — retention (LRU, keep-all, ski-rental,
   ski-rental-with-predictions, offline Belady/OPT) and schedulers (FIFO,
   static-priority, EDF, least-slack-first).
6. **Phase 3**: `run.py`, `sweep.py`, `ablate.py`.

Note on scope: mechanism 2 (least-slack-first scheduling) is **not** affected by
the context-scale problem. Contention is real and heavy at every scale tested; at
toy scale it is simply *active-KV* contention, which is a scheduling and
admission problem rather than a retention one. That half can proceed independently.

## 12. On vLLM

Position taken with the user: model vLLM's semantics, do not implement inside it.
vLLM requires CUDA (user is on Metal), and the competitive-ratio method needs
offline Belady, which cannot be computed on a live server. Useful detail for the
writeup: **vLLM's automatic prefix caching evicts with LRU**, so the LRU baseline
is the real incumbent rather than a strawman. If GPU access appears, spend it
calibrating the cost model (especially `batch_decode_coeff`), not on a
reimplementation.

---

## 13. Derived rent design (supersedes §3 decision 4, resolves §9.2)

Decided with the user before Milestone 1. This replaces blocking-only rent.

### The four rules

1. **Rent scales with eviction pressure.** The accrual rate is proportional to
   how fast the cache is actually being forced to evict, measured live from
   engine state.
2. **Rent for holding entry E = the recompute cost of the next victim that would
   be evicted instead of E** — the marginal victim, selected from current budget
   state, **excluding E itself**. If E were released, that victim would survive;
   the damage E causes by staying is precisely that victim's recompute.
3. **Rent = 0 when nothing else is evictable.** No victim means no pressure and
   no damage. This covers the case where the budget is full of pinned in-flight
   KV: retention had no lever, so charging it would blame the wrong mechanism.
   That time is reported separately as residual blocking.
4. **Rent accrues continuously over time**, throughout any interval where the
   budget is full and pending work would require an eviction — not only while a
   request is hard-blocked.

### Why not the obvious version

The first formulation set the rate equal to the marginal victim's recompute cost
directly. That gives the rate units of seconds per second, i.e. dimensionless,
so break-even lands at `E_recompute / victim_recompute` **time units** — and
which time unit is arbitrary. With measured numbers, two 7,710-token prefixes
(28.8s recompute each) break even at `t* = 1.0 second`, while the measured p50
reuse gap is ~21s. Ski-rental would evict almost everything immediately and lose
to LRU. Re-express the identical rule in milliseconds and it evicts 1000x
sooner. A rule whose behaviour depends on the unit you write it in has a missing
factor, and the missing factor is the eviction rate.

So: `rate = marginal_victim_recompute_seconds × evictions_per_second`. Accrued
rent then reads as "recompute-seconds actually being forced while I hold this",
which is directly comparable to E's own recompute cost — the ski-rental
comparison, in consistent units.

### Still no price knob

§3 decision 4's *principle* survives even though its mechanism is replaced:
there is no configurable memory price. Both terms in the rate are observed —
the victim's recompute comes from `CostModel` prefill physics (the same physics
the offline reference uses), and the eviction rate is counted from the engine.

One honest caveat: the eviction rate needs smoothing, so there is an EWMA time
constant. It is an **estimator** parameter, not a price — it changes how noisily
an observable is measured, not how much memory costs. It should still be listed
as a PLACEHOLDER and swept.

### RESOLVED in Phase 4 — retention unit and footprint scaling

Decided: **the retention unit is the block-quantised prefix head, plus a
separately-keyed tail**, following vLLM. KV is allocated in fixed blocks
(`EngineConfig.block_tokens`, default 16, PLACEHOLDER), and a finished call
leaves *all* its prompt blocks cached — not only the part a later call happens
to match. So each completed call leaves two things behind:

* a **head** entry: the shared prefix, rounded up to a whole block, genuinely
  reusable;
* a **tail** entry: every block past the divergence point, which costs exactly
  as much memory and can never be matched.

Online, a head and a tail are indistinguishable at creation time. That is the
decision retention actually has to get right, and modelling only the head would
have assumed the problem away.

Rent is charged on **held** bytes; benefit accrues on **useful** tokens:

```
rent_rate(E) = E.bytes x victim_recompute_s_per_byte x eviction_rate
break-even   = recompute_saved(useful tokens) / rent_rate(E)
```

The footprint question resolves cleanly, and the cancellation worry was real
but harmless. Rent and benefit are *both* linear in tokens, so entry size
cancels in every retention unit — break-even is the same for a 64-token and a
2,048-token entry. What does move it is `useful / held`:

| retention unit | useful/held | break-even at eviction rate r |
|---|---|---|
| hold only the shared prefix | 1.0 | `1 / r` |
| block-quantised head (chosen) | `shared / ceil(shared/16)*16` | `(useful/held) / r` |
| hold whole context | 0.138 (measured) | `0.138 / r` |

Measured on the committed trace, the median reusable prefix is **4 tokens**, so
the median head occupies a full 16-token block for a useful/held of **0.25**.
Block quantisation therefore quarters the break-even at this workload's median
and is negligible above one block. `results/figures/gap_histogram.png` draws
both bands: solid for the chosen unit, dotted for the exact-prefix reference.

### OPEN QUESTION as originally raised (kept for the record)

As written, rule 2 charges a 20,000-token entry exactly the same rent as a
500-token one, because the rate depends only on the *victim's* recompute cost
and the eviction rate. That is wrong in the direction that matters: a large
entry displaces proportionally more of the budget, so it should accrue rent
faster. Two candidate forms:

```
(a)  rent_rate(E) = (E.bytes / victim.bytes) × victim_recompute × eviction_rate
(b)  rent_rate(E) = per_byte_rent_price × E.bytes
     where per_byte_rent_price = (victim_recompute / victim.bytes) × eviction_rate
```

These are algebraically identical; (b) just names the intermediate quantity.
Note that under either form, if E and the victim are the same size the rule
reduces to rule 2 as originally written, so this is a generalisation rather than
a contradiction.

Unresolved consequence to check when implementing: E's break-even time becomes
`E_recompute / rent_rate(E)`. Both numerator and denominator scale with E's
size, so the scaling may cancel and leave break-even size-invariant — which
would defeat the purpose. Whether it cancels depends on whether recompute is
linear in tokens (it is, under the current CostModel) and whether bytes are
linear in tokens (they are). **Decide and justify at Phase 4, with a worked
example at two entry sizes.**

### Consequence for the competitive-ratio claim

Classic ski-rental assumes **constant** rent. Derived rent is not constant: it
moves with load and with which victim is currently marginal. The textbook
2-competitive guarantee therefore does **not** transfer, and must not be claimed.
Measure the realized ratio against the offline reference instead, and label that
baseline "offline reference" rather than "optimal" — exact OPT under continuous
batching with load-dependent rent is not proven here.

---

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
supported by `benchmark/results.jsonl`.** What the data does support is an
ablation of RAG and of the Sentinel, plus a coder-model comparison.

Also note there is no config named `hetero`, even though that is
`orchestrator.run`'s default and the value a bare `python orchestrator.py`
produces. Its rows would be functionally identical to `single_agent`.

### The run_experiments header is wrong

`benchmark/run_experiments.py:2` claims:

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

On 2026-09-29 a traced run retrieved `sim/workload.py` as "relevant code" for a
HumanEval bug and overwrote it (see §15). The obvious follow-up question is
whether any *historical* result is affected. It is not. Timeline:

| event | when | source |
|---|---|---|
| `benchmark/results/run_*.json` written | 2026-04-18 15:36–16:02 | file mtimes |
| `benchmark/results.jsonl` last written | 2026-05-05 01:30 | file mtime |
| ...and committed | 2026-05-04 / 05-05 | `3215189`, `a01af03` |
| `sim/` first appeared on disk | **2026-09-15 16:12** | `sim/__init__.py` mtime |
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
`sim/engine.py`, `tracing.py` and `tests/conftest.py`. That difference is
exactly the regression, and it did not exist when the 120 rows were produced.

**The existing `results.jsonl` and `benchmark/results/*.json` remain usable.**
The contaminated artefacts are only the 2026-09-29 traced run's outputs.

## 15. Ollama prompt-cache model — UNVALIDATED, does not match measurement

`sim/ollama_cache_model.py` models the confirmed mechanism (one slot per model,
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
