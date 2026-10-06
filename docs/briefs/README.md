# Parallel work briefs

Three packages that can run in three separate sessions at the same time. They
were split so that **no two touch the same file**. Read your own brief and the
coordination rules below; do not read the others' files into your context.

## File ownership — do not cross these lines

| brief | owns | must not touch |
|---|---|---|
| `01-run-manifests.md` | `experiments/manifest.py` (new), `experiments/run_scheduler_table.py`, `experiments/run_retention_grid.py`, `tests/simulator/test_manifest.py` (new) | `simulator/`, `docs/FINDINGS.md` |
| `02-hardware-profiles.md` | `simulator/workload.py`, `tests/simulator/test_provenance.py` | `experiments/`, `docs/FINDINGS.md` |
| `03-findings-index.md` | `docs/FINDINGS.md` only | everything else |

Each brief appends to `docs/DECISIONS.md`, which conflicts. **Append at the very
end of the file and keep it to one section.** If you hit a conflict there, take
both sides — never drop someone else's section.

## Ground rules (these come from the project owner and are not negotiable)

1. **Never call the LLM.** No command that hits Ollama. If a step needs a real
   model run, write the exact command out and stop; the owner runs it.
2. **Commit after each step**, with the full test suite green. Baseline is
   `234 passed, 1 skipped` — run `./venv/bin/python -m pytest -q`.
3. **Keep MEASURED vs PLACEHOLDER labels** on every constant, and add new ones
   to `CostModel.provenance()`.
4. **Stop at the end of your brief and report.** Do not continue into adjacent
   work that looks obvious.
5. **Negative results are results.** If something does not work, report the
   number, do not soften it and do not tune until it looks better.
6. **Do not run two heavy jobs at once.** Simulator runs contend; a test suite
   alongside a grid cell once stretched 76s of CPU over 37 minutes of wall
   clock. Serialise.
7. **Unbuffered output or you are blind.** Use `./venv/bin/python -u` and never
   pipe a long-running job through `tail` — that re-buffers and you will see
   nothing until it exits. This has cost this project time twice.

## Required reading before you start (in this order)

- `docs/ROADMAP.md` — the four workstreams and where we are
- `docs/DECISIONS.md` §13, §21-24 — the rent model, per-block reuse, the Ollama
  cache mode, and common random numbers
- `docs/FINDINGS.md` §24 and §27 — why retention cannot move latency here, and
  why no scheduler beats FCFS. These two explain most of the current state.

Do not re-derive any of that. If your brief disagrees with what you find in the
code, the code is the truth and the brief is stale — say so in your report.

## Why these three exist

Three separate results have been retracted for one root cause: a measurement
taken outside the regime where a mechanism can act, or with calibration and
execution disagreeing about the configuration. Brief 01 makes that
structurally impossible. Brief 02 makes the hardware assumption explicit, since
every crossover measured so far is a Mac/Ollama result and a discrete GPU has a
very different prefill-to-decode ratio. Brief 03 makes the retraction history
legible so nobody quotes a dead number.
