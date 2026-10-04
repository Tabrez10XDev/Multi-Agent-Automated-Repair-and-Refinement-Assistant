# Milestone 2 roadmap

Status as of 2026-10-04. One line per workstream; detail lives in the linked
sections of FINDINGS / OPEN_ISSUES / DECISIONS.

| # | Workstream | Status |
|---|---|---|
| 1 | Ollama prompt-cache baseline | **finishing now** — rule being pinned |
| 2 | Retention comparison | next, runs on the new trace |
| 3 | Better output-length estimator | not started |
| 4 | vLLM on an RTX 4060 | not started; early feasibility check lands during 2 |

## 1. Ollama prompt-cache baseline — DONE

Rule read out of llama.cpp source and re-frozen as v2 (DECISIONS §19,
FINDINGS §20): `f_keep < 0.5f` gates both the save and the cache search;
`f_sim` is over the incoming prompt, `f_keep` over the slot. Scores 84/89
(`original`), 76/78 (`shared_first`), with exact repeats and partial prefix
reuse reported separately.

Held open deliberately: validation on the new trace, because over half the
fit-set hits are harness repetition (OPEN_ISSUES §20).

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
