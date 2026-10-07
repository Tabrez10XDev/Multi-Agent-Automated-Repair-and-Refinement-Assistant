# Architecture

What the simulator is, what it is made of, and the rules its parts obey.

> Split out of the single `sim/HANDOFF.md` during the Milestone 2 cleanup.
> **Section numbers are the original HANDOFF numbers and are deliberately not
> renumbered**, so cross-references like "supersedes §3 decision 4" still
> resolve. They are not sequential within a file. The other parts live in
> `ARCHITECTURE.md`, `DECISIONS.md`, `FINDINGS.md` and `OPEN_ISSUES.md`.

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

Layout as of the Milestone 2 cleanup. The split is by **provenance**: `data/`
holds measured inputs that cannot be regenerated, `results/` holds output that
can, and `probes/` is the only place that talks to a live model server.

| path | contents |
|---|---|
| `maara/` | The agent pipeline that generates the workload. `orchestrator.run` is the entry point |
| `maara/agents/` | The role agents: coder, debugger, evaluator, issue_parser, navigator, retriever, test_runner |
| `maara/sandbox.py` | One definition of what the coding agent may see and touch (`EXCLUDED_DIRS`, `is_agent_visible`) |
| `maara/prompt_layout.py` | `MAARA_PROMPT_LAYOUT` block-ordering switch |
| `maara/tracing.py` | One JSONL row per LLM call, plus the prompt sidecar |
| `simulator/workload.py` | `CostModel`, `StageTokens`, `CpuStageSeconds`, `WorkloadConfig`, `generate_trace()` — unrolls MAARA's DAG into a replayable trace |
| `simulator/engine.py` | Continuous-batching engine: shared KV budget, wait queue, block-hash prefix cache with refcounts, eviction, rent accounting |
| `simulator/policies.py` | Schedulers (FCFS, SJF, EDF, LSF, static priority) and retention policies (LRU, keep-all, evict-now, ski-rental, ski-rental-pred, offline-ref) |
| `simulator/trace_loader.py` | Replay of real traces: segmentation, config fingerprinting, prefix recovery, cache-hit detection |
| `simulator/ollama_cache.py` | Model of Ollama's own prompt cache. Rule frozen as v2 from llama.cpp source (DECISIONS §19); held-out recall 81% / 80%, misses all in `issue_parser` (FINDINGS §22, OPEN_ISSUES §24) |
| `probes/calibrate.py` | Calibration harness. Measures rather than assumes. **Calls the model** |
| `probes/ollama_cache_probe.py` | Probes for slot count, capacity and prefix affinity. **Calls the model** |
| `analysis/analyze_trace.py` | Token distributions, gap histogram, break-even band, figures |
| `analysis/check_trace_clean.py` | Gate: verifies a collected trace saw only its sandbox |
| `analysis/pass_at_1.py` | pass@1 per configuration from the committed outcome rows |
| `experiments/collect_humaneval.py` | Runs the HumanEval tasks and collects traces. **Calls the model** |
| `experiments/run_retention_grid.py` | retention x scheduler grid with utilisation calibration |
| `data/traces/` | The committed traces and prompt sidecars. Measured; never regenerate |
| `data/calibration/measured.json` | Calibration output, auto-loaded by `workload.py`. `CostModel.provenance()` prints MEASURED vs PLACEHOLDER |
| `data/outcomes/` | HumanEval pass/fail rows from the real runs |
| `data/demo_target/` | The fixture the orchestrator demo repairs |
| `archive/` | Retired code, with reasons in `archive/README.md`. Do not import |

Run `make help` for one entry point per task, or:

```
./venv/bin/python -m simulator.engine --help
```

---

## 10. Oracle fields — do not leak to online policies

`generate_trace()` samples loop trip counts up front and emits the realized path,
so the trace object literally contains the future. `Node.next_reuse_node`,
`Node.next_reuse_gap_s`, and `Node.critical_path_remaining` are oracle fields.
They exist for offline Belady/OPT and for constructing the *noisy hint* the
learning-augmented ski-rental variant receives. An online policy reading them
directly invalidates the comparison.

---


## 12. On vLLM

Position taken with the user: model vLLM's semantics, do not implement inside it.
vLLM requires CUDA (user is on Metal), and the competitive-ratio method needs
offline Belady, which cannot be computed on a live server. Useful detail for the
writeup: **vLLM's automatic prefix caching evicts with LRU**, so the LRU baseline
is the real incumbent rather than a strawman. If GPU access appears, spend it
calibrating the cost model (especially `batch_decode_coeff`), not on a
reimplementation.

---
