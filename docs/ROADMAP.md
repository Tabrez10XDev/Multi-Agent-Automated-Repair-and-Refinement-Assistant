# Milestone 2 roadmap

Status as of 2026-10-04. One line per workstream; detail lives in the linked
sections of FINDINGS / OPEN_ISSUES / DECISIONS.

| # | Workstream | Status |
|---|---|---|
| 1 | Ollama prompt-cache baseline | **finishing now** — rule being pinned |
| 2 | Retention comparison | next, runs on the new trace |
| 3 | Better output-length estimator | not started |
| 4 | vLLM on an RTX 4060 | not started; early feasibility check lands during 2 |

## 1. Ollama prompt-cache baseline — finishing now

Model the cache the real system already has, so any retention policy is
measured against what exists rather than against a strawman.

Done: mechanism confirmed from the server binary and probes; capacity measured
(~8 GiB, byte-bounded LRU); the restore-does-not-consume bug fixed
(DECISIONS §17); the 0.10-destroys-the-slot claim retracted (FINDINGS §17 →
§19); exact prompt recurrence identified as what actually predicts a hit.

Remaining:
1. `--affinity-sweep` to pin the restore-to-trim cutoff (between 50% and 90%).
2. Settle the similarity denominator from existing `checking sim = %.3f (%zu/%zu)`
   server log lines.
3. Add the token-count eviction limit, absent from the model.
4. Re-score on both fit traces with **exact-repeat hits and within-task partial
   reuse scored separately** — conflating them is what produced the inflated
   Milestone 1 figure (FINDINGS §19).
5. Re-freeze before any new trace exists.

## 2. Retention comparison — next

keep-all, LRU, evict-now, ski-rental, ski-rental-with-predictions, and
`ollama_prompt_cache` (only if validated) × {FCFS, LSF}, over context {1x, 8x,
16x} and target utilisation {0.6, 0.8}.

Blocked on the new trace (below), because the current two traces cannot support
it: their cache-hit structure is dominated by cross-config exact repeats
(OPEN_ISSUES §20).

First step is a correctness check, not a sweep: whether a prefix block from one
task run can hit for a different task run in the simulator. If it can, that is
cross-run reuse the real system would not get, and earlier numbers depended on
it to an unknown degree.

Carries the **vLLM feasibility check** (workstream 4) as a side task.

## 3. Better output-length estimator — not started

LSF needs output length to compute slack. Current estimator is an online mean
with a PLACEHOLDER noise term. Planned: the oracle line in the scheduler table
as an upper bound, 20 seeds, and `slack_factor` calibrated so FCFS misses
20–30% of deadlines.

## 4. vLLM on an RTX 4060 — not started

Out of scope for Milestone 1 and still not committed to. An **early feasibility
check during workstream 2**, not a full port: does an RTX 4060's 8 GB hold
qwen2.5-coder:7b plus a KV budget large enough to reach the contended regime
the simulator studies? If not, the simulator stays the only venue and that
should be stated rather than discovered late.

## New trace collection (Part 3)

Serves two purposes at once: held-out validation for workstream 1 and the main
workload for workstream 2.

Design: 50+ distinct HumanEval problems **not** among the current 15; 2 configs
(full pipeline with the qwen coder, and the llama coder for model mixing);
original prompt layout; zero_shot; one process start to finish; **Ollama
restarted between configs** so no cache state carries over.

The restart and the disjoint problem set are the point — they remove the
cross-config exact repeats that made the existing traces unusable as a cache
benchmark.
