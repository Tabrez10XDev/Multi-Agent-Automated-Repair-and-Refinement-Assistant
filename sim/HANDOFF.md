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
   knob. The engine charges rent by observing that a retained entry kept a
   request waiting. **Do not add a memory-price config value** — it would let the
   retention policy read the answer it is supposed to infer.
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
   patch. **Unresolved; needs the user's input.**
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
