# archive/

Retired code, kept because it produced numbers that appear in the Milestone 1
writeup. Nothing here is imported by the live code, and nothing here is tested.
Do not import from this directory.

## legacy_case_harness/

The pre-HumanEval benchmark. Superseded by `experiments/collect_humaneval.py`
driving `maara.orchestrator.run`, which runs real HumanEval tasks, emits traces,
and enforces the sandbox guard this harness never had.

| Path | Reason retired |
|---|---|
| `runner.py` | Second pipeline (`run_case`, `run_benchmark`). Zero callers, zero tests, and it imported the agents without `maara.sandbox`, so its runs were unguarded. |
| `metrics.py` | `MetricsTracker`, used only by `runner.py`. |
| `setup_cases.py` | Generated `cases/`. Writes 40 files as an import side effect — no `__main__` guard. |
| `cases.json` | Case index for `runner.py`. |
| `cases/` | 20 synthetic `case_NN` bugs, replaced by 15 real HumanEval tasks. |
| `results/` | 7 runs from 2026-04-18, one case each, `total_tokens: 0` (the tracker predates token accounting). Not comparable to the HumanEval results in `data/outcomes/`. |

## prototype/

| Path | Reason retired |
|---|---|
| `agent.py` | The original single-file self-healing loop: called `ollama` directly, no tracing, no multi-agent roles, operated on the repo's own `target.py`. All five of its functions had zero callers. Kept because it is the starting point the multi-agent design was measured against. |

`maara/sandbox.py` still lists `agent.py` in `EXCLUDED_FILES`, and
`analysis/check_trace_clean.py` still lists the old directory names. Those
entries are deliberately left in place: the committed traces were collected when
these files were live, so the guards must still recognise them.
