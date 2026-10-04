# Tour

Where things are, and what happens when you run them. Two flows matter: a
benchmark run that produces a trace, and a simulator run that turns a trace
into metrics. Everything else is support for those two.

---

## The shape of it

```
HumanEval task ──> maara/ (agents) ──> Ollama ──> data/traces/*.jsonl.gz
                                                        │
                                                        v
                                        simulator/ (engine + policies)
                                                        │
                                                        v
                                        results/ (metrics, figures)
```

The left half calls a real model and takes hours. The right half is pure
computation, takes seconds, and is what you iterate on. They are joined only
by the trace files in `data/traces/`, which is why those are committed and
treated as measured data.

---

## Flow 1: one benchmark run, task to trace

`make collect CONFIRM=1` → `experiments/collect_humaneval.py`.

**1. Per-task sandbox.** For each of the 15 HumanEval tasks, the driver creates
a throwaway directory and writes exactly two files into it: `target.py` (the
buggy code) and `test_target.py` (the tests). The agent's `repo_path` *is* that
directory. This matters — an earlier run used the repo root as `repo_path`, and
the retriever indexed `simulator/workload.py` as "relevant code" for a
HumanEval bug and rewrote it. `maara/sandbox.py` and the per-task sandbox are
both consequences of that.

**2. The pipeline**, in `maara/orchestrator.py::_run_impl`:

| step | agent | model | what it does |
|---|---|---|---|
| 1 | `get_file_structure` | — | lists the sandbox (filtered by `is_agent_visible`) |
| 2 | `parse_issue` | llama3.2 | issue text → structured plan |
| 3 | `navigate` / `update_index` | — | RAG retrieval of relevant code |
| 4 | `evaluate_plan` | llama3.2 | the Sentinel gate; can reject and force a replan (up to `MAX_REPLAN = 2`) |
| 5 | `generate_fix` | qwen2.5-coder:7b | plan + code + diagnosis → patched code |
| 6 | `apply_and_test` | — | writes the patch, runs pytest |
| 7 | `diagnose` | qwen2.5-coder:7b | on failure, explains why; loops back to 5 |

Steps 5–7 repeat up to `max_attempts = 3`. Which agents run depends on the
config arm — the orchestrator's only ablation gates are
`config != "hetero_no_rag"` and `config != "hetero_no_sentinel"`.

**3. Tracing.** `maara/tracing.py` wraps every Ollama call and appends one JSONL
row: `call_id`, `agent_role`, `model`, `prompt_tokens`, `prompt_eval_duration`
(converted from Ollama's nanoseconds to seconds), `t_start`/`t_end`,
`parent_call_ids`, and the attempt and replan counters. Prompt *text* goes to a
separate sidecar file keyed on `call_id`, so the main trace stays small enough
to read and the prompts are still there when you need them.

**4. The gate.** `make check-traces` runs `analysis/check_trace_clean.py`, which
scans every prompt for strings that could only come from outside the sandbox —
`simulator/`, `maara/`, `CostModel`, `prefill_rate`. It still lists the
pre-cleanup directory names, because the committed traces were collected under
the old layout and dropping those entries would make the gate stop detecting
the exact contamination it exists for.

Outcome rows (pass/fail, attempts, latency) append to `data/outcomes/`.
`make pass-at-1` turns those into the pass@1 table.

---

## Flow 2: one simulator run, trace to metrics

`./venv/bin/python -m simulator.engine --scheduler lsf --retention ski-rental --kv-mb 512`

**1. Workload.** Either `simulator/workload.py::generate_trace()` synthesises a
trace from the measured distributions, or `simulator/trace_loader.py::load_trace`
replays a real one. Replay does real work: it segments rows into task runs (by
`run_id` *and* sequence resets — two processes once shared a `run_id` and
segmentation silently produced 305 blocks instead of 60), fingerprints which
config arm each run came from, and recovers shared prefixes by
longest-common-prefix on the prompt text, **per model**, since a prefix shared
between a qwen call and a llama3.2 call is not reusable.

**2. The engine** (`simulator/engine.py`) is a continuous-batching simulator:
a fixed KV budget, a wait queue, chunked prefill, and a block-hash prefix cache
where each 16-token block is keyed by a hash chain over all prior blocks, so
partial hits work the way vLLM's do.

Each iteration it admits what fits, decodes a token for everything resident,
and charges **rent** against retained entries — the cost of holding memory
somebody else needs, derived from what an eviction actually costs its victim
(`bytes × victim_recompute_s_per_byte × eviction rate`, normalised by the
evictable pool) rather than from a tuned constant. See `DECISIONS.md` §13.

**3. Policies** (`simulator/policies.py`) make two decisions:

- **Retention**: keep or evict. `ski-rental` buys (keeps) once accrued rent
  reaches the entry's own recompute cost. Baselines: `lru`, `keep-all`,
  `evict-now`, and `offline-ref`, which is the hindsight optimum and not a
  policy you could run.
- **Scheduling**: who goes next. `lsf` picks least slack, using an output-length
  estimator and the remaining critical path.

Oracle fields (`next_reuse_gap_s`, `output_tokens`, `critical_path_remaining`)
are visible to the offline reference and **forbidden** to online policies;
`ReqView` enforces that rather than trusting the author to remember.

**4. Out.** Metrics to stdout; `analysis/analyze_trace.py` writes
`results/trace_stats_<set>.json` and the figures.

One caveat that will bite you: at default settings nothing blocks on KV, so
retention cannot matter and every policy scores the same. The engine says so
explicitly when it happens. `--kv-mb 512` is a contended setting; much lower
and the worst-case feasibility guard refuses to start.

---

## What is being worked on now

`docs/ROADMAP.md` has the Milestone 2 workstreams and their status. Read it
before starting anything — two of the four are blocked on a trace that has not
been collected yet.

## Where each Milestone 1 finding is recorded

| finding | where |
|---|---|
| Model rates overturned the assumed constants | `FINDINGS.md` §4 |
| qwen's KV bytes/token confirmed against the server log; llama3.2's still DERIVED | `FINDINGS.md` §4, `OPEN_ISSUES.md` §9 |
| Reusable prefix is tiny (median 4 tokens, 13.7% of prompt tokens) | `FINDINGS.md` §14a |
| ~24% of prompt tokens repeat but are unreachable by a prefix cache, because they recur out of order | `FINDINGS.md` §14a |
| Retention only matters at 2000–4000 token contexts | `FINDINGS.md` §7 |
| Shared-first prompt layout backfired; order by volatility, not role coverage | `FINDINGS.md` §14b |
| It also broke the evaluator on 7 of 60 runs | `FINDINGS.md` §14b |
| Ollama *does* prefix-cache; `prompt_eval_count` hides it | `FINDINGS.md` §14d |
| The single-slot hypothesis was not supported | `FINDINGS.md` §14d |
| Two processes sharing a `run_id` invalidated a figure set | `FINDINGS.md` §14c |
| `single_agent` is not a single agent | `OPEN_ISSUES.md` §14 |
| The Ollama cache model does not reproduce measured hits | `OPEN_ISSUES.md` §15 |
| Derived-rent design and the four rules | `DECISIONS.md` §13 |
| Bugs fixed, not to be reintroduced (nine of them, including the eviction double-count deadlock, the pressure snapshot taken after admission, and `used_bytes` as the real hot loop) | `DECISIONS.md` §8 |
| Competitive-ratio claim does not survive non-constant rent | `DECISIONS.md` §13 |

Section numbers are the original HANDOFF numbers, kept so cross-references
still resolve. They are not sequential within a file.
