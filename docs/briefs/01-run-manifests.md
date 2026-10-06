# Brief 01 — Run manifests, with a blocking calibration check

**Owns:** `experiments/manifest.py` (new), `experiments/run_scheduler_table.py`,
`experiments/run_retention_grid.py`, `tests/simulator/test_manifest.py` (new).
**Must not touch:** anything under `simulator/`, `docs/FINDINGS.md`.

Read `docs/briefs/README.md` first.

## Why this exists

`docs/FINDINGS.md` §23 had to be retracted. Its scheduler table was measured at
roughly **2.2x the intended load**, because `calibrate_rate` took no `batch`
argument at the time: it calibrated the arrival rate against `run_one`'s default
`batch=16`, producing 0.0753/s, and the table then ran at `batch=8`. Correctly
calibrated the rate is 0.0343/s. Everything downstream was wrong — p95 read 274
to 322s instead of about 133s, and the scheduler's decision-choice fraction read
34.7% when it is really 0.3%. A claimed 14.8% p95 improvement evaporated.

That class of error must become impossible rather than remembered.

## What to build

### 1. `experiments/manifest.py`

A `Manifest` that an experiment creates at startup, records into, and writes
beside its results.

It must capture:

- **git hash** and whether the working tree was dirty at run time
- **every parameter** the run used: trace set, scale, utilisation target, KV
  budget, max batch, cache mode, scheduler, retention, estimator, seeds, prefill
  chunk, sigma, slack factor
- **every calibrated value**, each stored *with the configuration it was
  calibrated under* — this is the whole point, not an extra

Suggested shape (adjust if the drivers want something else):

```python
@dataclass
class Calibration:
    name: str            # "arrival_rate", "slack_factor", ...
    value: float
    under: dict          # the config this was calibrated at
    target: str          # what it was calibrated to hit, in words
    achieved: float      # what it actually hit
```

### 2. The blocking check

`Manifest.assert_consistent(run_config)` raises a clear, named exception when
any **calibration-sensitive** key differs between a calibration's `under` and
the run's actual config. At minimum those keys are:

```
trace_set, scale, kv_mb, max_batch, cache_mode
```

The exception message must name the key, the calibrated value, and both
configurations, and say which one to change. A run that fails this check must
**not** produce results. Make the failure loud and early — before any engine
run, not at write time.

`util` belongs in `under` for the arrival-rate calibration specifically, since
that is what it targets.

### 3. Wire it into both drivers

`experiments/run_scheduler_table.py` and `experiments/run_retention_grid.py`
both already calibrate an arrival rate; the scheduler table also calibrates
`slack_factor`. Have each one:

- build a `Manifest` at startup
- record each calibration with the config it ran under
- call `assert_consistent` immediately before the measurement loop
- write the manifest into the existing results JSON under a `"manifest"` key,
  or alongside it as `<name>.manifest.json` — your call, but it must be
  machine-readable and sit next to the numbers it describes

### 4. Prove it catches the real bug

Add `tests/simulator/test_manifest.py` with at least:

- a calibration recorded at `max_batch=16` and a run at `max_batch=8` **raises**
- the same config passes
- a differing `trace_set`, `kv_mb` or `cache_mode` raises
- a manifest round-trips to JSON and back
- the git hash and dirty flag are populated

The first test is the regression guard for §23. Name it so that is obvious.

## Acceptance

- Full suite green: `./venv/bin/python -m pytest -q` (baseline 234 passed, 1 skipped)
- `./venv/bin/python -u -m experiments.run_scheduler_table --quick` still runs
  and now writes a manifest
- Deliberately mis-calibrating (e.g. pass a different `--batch` than the
  calibration used) **fails loudly**; show the error text in your report
- One section appended to the very end of `docs/DECISIONS.md` recording the
  convention and that FINDINGS entries must cite a manifest

## Do not

- Change any calibration *logic* or default. You are adding a guard, not
  retuning anything. If a default looks wrong, report it, do not change it.
- Touch `simulator/`. If the guard seems to need an engine change, stop and say
  so instead.
- Re-run the grid or the scheduler table for real results. `--quick` only.
