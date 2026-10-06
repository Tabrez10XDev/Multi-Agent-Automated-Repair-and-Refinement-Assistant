# Brief 02 — Make the cost model a named hardware profile

**Owns:** `simulator/workload.py`, `tests/simulator/test_provenance.py`.
**Must not touch:** anything under `experiments/`, `docs/FINDINGS.md`.

Read `docs/briefs/README.md` first.

## Why this exists

Every crossover and regime boundary this project has measured is a
**Mac + Ollama + Metal** result, and nothing in the code says so. Workstream 4
will test a discrete GPU (RTX 4060, 8 GB), where the prefill-to-decode ratio is
very different — plausibly several times larger. That ratio is exactly what
decides whether either mechanism has room to act:

`docs/FINDINGS.md` §24 measured recompute at **0.8 to 1.1% of makespan**, with
decode at 55% and prefill at 25%. Retention can only ever win avoided prefill.
On hardware where prefill is a much larger share, the same workload could sit in
a completely different regime. Right now a reader cannot tell which machine any
number came from.

## What to build

### 1. Named profiles

Turn the cost model's constants into a named profile. `simulator/workload.py`
currently holds `CostModel` with a `rates` mapping of per-model
`prefill_rate` / `decode_rate` / `kv_bytes_per_token` (each already carrying a
`measured: bool`), plus `batch_decode_coeff`.

Introduce at least:

- `mac_ollama_measured` — exactly today's values, so nothing changes by default
- `gpu_4060` — a clearly-labelled **PLACEHOLDER** profile, every field marked
  unmeasured, present so workstream 4 has somewhere to put real numbers

Profile selection should be explicit and introspectable: a name on the
`CostModel`, reachable from `provenance()`.

### 2. The default must not change anything

This is the hard constraint. `mac_ollama_measured` must reproduce current
behaviour **exactly**. Verify with fixed-seed engine runs before and after:

```
./venv/bin/python -m simulator.engine --scheduler fcfs --retention lru --seed 7 --kv-mb 512
./venv/bin/python -m simulator.engine --scheduler lsf  --retention ski-rental --seed 7 --kv-mb 512
```

Capture those outputs before you start and diff them after. They must be
byte-identical apart from any new provenance line you deliberately add. Say so
explicitly in your report.

### 3. Provenance must name the profile

`CostModel.provenance()` already prints MEASURED / DERIVED / PLACEHOLDER per
field. Add the profile name and a one-line statement that the measured values
are Mac/Ollama/Metal. Keep every existing label; `batch_decode_coeff` stays
**PLACEHOLDER** (0.15) — a probe to measure it properly is queued separately
and is not your job.

Also still true and still worth carrying: `llama3.2`'s `kv_bytes_per_token`
is **DERIVED**, not measured (see `docs/OPEN_ISSUES.md` §23). Do not quietly
promote it.

### 4. Tests

Extend `tests/simulator/test_provenance.py`:

- the default profile is `mac_ollama_measured` and its measured fields are
  marked measured
- `gpu_4060` exists and **every** field is marked unmeasured
- `provenance()` names the active profile
- selecting an unknown profile fails with a clear error
- a guard that the default profile's numeric values match the current constants,
  so a future edit cannot silently shift the baseline

## Acceptance

- Full suite green (baseline 234 passed, 1 skipped)
- The two fixed-seed runs above are byte-identical to before, shown in your report
- One section appended to the very end of `docs/DECISIONS.md` recording the
  profile convention and that all current findings are Mac-profile results

## Do not

- Invent plausible-looking RTX 4060 numbers. Placeholders must be obviously
  placeholders and must be labelled unmeasured. A fabricated constant that
  looks measured is worse than a missing one.
- Change `batch_decode_coeff`, or promote any DERIVED value to MEASURED.
- Touch `experiments/` or `docs/FINDINGS.md`.
