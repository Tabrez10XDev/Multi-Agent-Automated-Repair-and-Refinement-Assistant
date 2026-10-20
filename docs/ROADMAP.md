# Milestone 2 roadmap

Status as of 2026-10-20. One line per workstream; detail lives in the linked
sections of FINDINGS / OPEN_ISSUES / DECISIONS.

| # | Workstream | Status |
|---|---|---|
| 1 | Ollama prompt-cache baseline | **done** (FINDINGS §20, §22, §28) |
| 2 | Retention comparison | **1x done** (FINDINGS §24-25); 8x calibration fixed, grid not complete (OPEN_ISSUES §27) |
| 3 | Better output-length estimator | **closed on evidence**: the oracle estimator does not help LSF, and LSF loses to `program-fcfs` (FINDINGS §29-30) |
| 4 | vLLM on an RTX 5060 Ti 16 GB | deferred; plan in `docs/GPU_TODO.md` |
| - | Sealed `final` evaluation | policies frozen (DECISIONS §31, `f4c889b`); `final_qwen` 5 of 40 cells done (FINDINGS §32) |

Milestone 2 headline (DECISIONS §31): program-level ordering. Remaining, in
order: finish the sealed runs from `f4c889b`, then the 8x retention grid.
The sections below are the original plan, kept for the record.

## 1. Ollama prompt-cache baseline — DONE

Rule read out of llama.cpp source and re-frozen as v2 (DECISIONS §19,
FINDINGS §20): `f_keep < 0.5f` gates both the save and the cache search;
`f_sim` is over the incoming prompt, `f_keep` over the slot. Scores 84/89
(`original`), 76/78 (`shared_first`), with exact repeats and partial prefix
reuse reported separately.

Held-out scores: **hit recall 81% (44/54) and 80% (53/66)**, agreement 96.6%
and 95.4%. Recall is the number to quote — agreement is dominated by true
negatives, since only ~17% of held-out calls are hits. Against 94%/97% recall
on the fit set, so the rule loses ~14 points on unseen data, all of it in
`issue_parser` (OPEN_ISSUES §24).

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

## 4. vLLM on an RTX 5060 Ti 16 GB — not started

> **Retargeted 2026-10-07:** the expected GPU is an RTX 5060 Ti with 16 GB,
> not a 4060 with 8 GB, and it is the ceiling for this repo (DECISIONS §29).
> The step-by-step plan and the 16 GB budget are in `docs/GPU_TODO.md`. The
> original text is kept below.

Out of scope for Milestone 1 and still not committed to. An **early feasibility
check during workstream 2**, not a full port: does an RTX 4060's 8 GB hold
qwen2.5-coder:7b plus a KV budget large enough to reach the contended regime
the simulator studies? If not, the simulator stays the only venue and that
should be stated rather than discovered late.

## New trace collection (Part 3) — DONE

Collected 2026-10-04/05. `holdout_A_qwen2` (320 calls, 54 tasks, 28.0 min) and
`holdout_B_llama` (370 calls, 54 tasks, 29.3 min), one process each with Ollama
restarted between, both passing `check_trace_clean`, registered as the
`holdout_qwen` / `holdout_llama` trace sets.

Served both purposes: held-out validation of the cache rule (96.6% / 95.4%
agreement, no tuning — FINDINGS §22) and the workload for workstream 2.
