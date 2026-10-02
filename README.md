# Adaptive LLM Inference Scheduling

A workflow-aware controller for LLM serving, and the multi-agent coding
assistant used to generate a realistic workload for it.

The serving question: an LLM server holds a KV cache it cannot fit everything
in. When a request finishes, do you **keep** its prefix in case the workflow
comes back to it, or **evict** it and recompute later? Keeping costs memory
that other requests need now; evicting costs prefill time later. That is
rent-or-buy, so the controller here is built on ski-rental:

1. **KV retention** — ski-rental, deterministic and learning-augmented, with
   rent derived from what an eviction actually costs its victim rather than
   from a tuned constant.
2. **Scheduling** — least-slack-first over the workflow DAG, with an
   output-length estimator, since a request's deadline depends on what is
   waiting downstream of it.

The workload is **MAARA**, a five-role repair pipeline (issue parser →
evaluator → retriever → coder → test runner/debugger) running HumanEval tasks
against `qwen2.5-coder:7b` and `llama3.2` on local Ollama. Agent workflows are
the interesting case because they have structure a general-purpose scheduler
cannot see: repeated prefixes, known successors, and gaps where a prefix sits
idle but will be needed again.

Every constant in the simulator carries a **MEASURED** or **PLACEHOLDER**
label, printed by `CostModel.provenance()`. That discipline is the point: the
first version of this work was built on constants that turned out to be wrong.

---

## Layout

The split is by **provenance** — what can be regenerated and what cannot.

| path | what it is |
|---|---|
| `maara/` | The agent pipeline. `orchestrator.run` is the entry point |
| `maara/agents/` | The seven role agents |
| `maara/sandbox.py` | The one definition of what the coding agent may see and touch |
| `maara/tracing.py` | One JSONL row per LLM call, plus a prompt sidecar |
| `simulator/` | Engine, workload model, scheduling and retention policies, trace replay |
| `probes/` | **The only code that calls a live model server.** Calibration and cache probes |
| `analysis/` | Reads traces and outcomes, writes numbers and figures. Calls nothing |
| `experiments/` | Runnable drivers: trace collection, the retention grid |
| `data/` | Committed measured inputs. **Not regenerable** |
| `results/` | Regenerable output. Gitignored except figures |
| `docs/` | Architecture, decisions, findings, open issues |
| `archive/` | Retired code, with reasons. Do not import |

---

## Running things

`make help` lists every task. Offline targets are safe to run anytime; the
three that call a live model require `CONFIRM=1`, because starting one by
accident costs hours.

```bash
make test          # full test suite (224 tests, ~77s)
make analyze       # trace stats + figures for both trace sets
make pass-at-1     # pass@1 per configuration from the committed outcomes
make grid          # retention x scheduler grid (quick mode)
make check-traces  # verify every committed trace stayed in its sandbox

make calibrate   CONFIRM=1   # re-measure rates and prompt sizes
make probe-cache CONFIRM=1   # probe Ollama's prompt cache
make collect     CONFIRM=1   # run HumanEval and collect a trace
```

Direct invocation works too, and every module takes `--help`:

```bash
./venv/bin/python -m simulator.engine --scheduler lsf --retention ski-rental --kv-mb 512
./venv/bin/python -m experiments.run_retention_grid --scales 1,8 --seeds 5
```

---

## Where the findings live

| Question | File |
|---|---|
| What is this made of, and what rules do the parts obey? | `docs/ARCHITECTURE.md` |
| Why is it built this way? What must not be silently revised? | `docs/DECISIONS.md` |
| What did the measurements actually show? | `docs/FINDINGS.md` |
| What is unvalidated, unknown, or known to be wrong? | `docs/OPEN_ISSUES.md` |
| How does one run flow end to end? | `docs/TOUR.md` |

Headline results, with detail and caveats in `docs/FINDINGS.md`:

- The reusable prefix at HumanEval scale is **small** — median 4 tokens, 13.7%
  of prompt tokens. A further ~24% of prompt tokens are repeated but
  unreachable by a prefix cache, because they recur out of order.
- Retention starts to matter at **2000–4000 token contexts**, not at the scale
  MAARA actually runs at. The mechanism is real; the workload has to be big
  enough for it to pay.
- Reordering prompts to put shared content first **backfired**: ordering by
  role coverage destroyed coder→coder reuse (7,626 → 703 tokens). The correct
  principle is to order by **volatility**, not by how many roles share a block.
- Ollama *does* do prefix caching, and `prompt_eval_count` hides it. An earlier
  "zero cache hits" conclusion was an artefact of a count-based detector.

---

## Setup

Requires Python 3.13+ and, for anything that calls a model,
[Ollama](https://ollama.com) running locally.

```bash
python3 -m venv venv
./venv/bin/pip install -e '.[sim,dev]'       # simulator + tests; no GPU needed
./venv/bin/pip install -e '.[agent]'         # adds the agent pipeline

ollama pull qwen2.5-coder:7b                 # coder, debugger  (4.7GB)
ollama pull llama3.2                         # issue parser, evaluator  (2.0GB)
ollama pull nomic-embed-text                 # RAG embeddings  (274MB)
```

Dependencies are pinned in `pyproject.toml`. The `sim` extra is stdlib plus
numpy and matplotlib and must stay that way — the simulator has to run on a
machine with no GPU and no model server, which is how the results above are
reproduced.

---

## Pipeline configurations

These are the four arms in the committed results. **The stored label
`single_agent` is a misnomer**: it is the full pipeline with both RAG and the
evaluator gate. The only ablation gates in the orchestrator are
`config != "hetero_no_rag"` and `config != "hetero_no_sentinel"`, so
`single_agent` trips neither. The stored strings are kept as written because
they appear in committed data; `simulator.trace_loader.config_label` maps them
to honest display names.

| stored | displayed | RAG | Sentinel | coder model | pass@1 (zero-shot) |
|---|---|---|---|---|---|
| `single_agent` | `full_pipeline` | yes | yes | qwen2.5-coder:7b | 86.7% |
| `hetero_no_rag` | `no_rag` | no | yes | qwen2.5-coder:7b | 93.3% |
| `hetero_no_sentinel` | `no_sentinel` | yes | no | llama3.2 | 60.0% |
| `homo_llama3` | `llama3_only` | yes | yes | llama3.2 | 46.7% |

`n=15` tasks per arm. `hetero` is a fifth, undocumented label: it is
`orchestrator.run`'s default and trips neither gate, so it behaves as
`full_pipeline`.

---

## Known issues

See `docs/OPEN_ISSUES.md` for the full list. The two that bite soonest:

- **`simulator/ollama_cache.py` is unvalidated.** It models the confirmed
  mechanism but reproduces only 56 of 89 measured cache hits on one trace and
  20 of 78 on the other. Do not use it as a baseline yet.
- **`llama3.2`'s `kv_bytes_per_token` is DERIVED, not measured.** qwen's was
  confirmed against the Ollama server log; llama3.2's still needs the same check.

---

## Authors

**Tabrez Mohamed** — tm3578@rit.edu
**Aparnaa Senthilnathan** — as1567@rit.edu
**Ashish Ubale** — au9332@rit.edu

Rochester Institute of Technology
