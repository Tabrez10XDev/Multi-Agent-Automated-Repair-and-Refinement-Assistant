# Decisions

Choices that are settled. Changing any of these needs a reason and a note here.

> Split out of the single `sim/HANDOFF.md` during the Milestone 2 cleanup.
> **Section numbers are the original HANDOFF numbers and are deliberately not
> renumbered**, so cross-references like "supersedes §3 decision 4" still
> resolve. They are not sequential within a file. The other parts live in
> `ARCHITECTURE.md`, `DECISIONS.md`, `FINDINGS.md` and `OPEN_ISSUES.md`.

---

## 3. Design decisions — locked by the user, do not silently revise

1. **Contention comes from many concurrent tasks.** Open Poisson arrivals of
   independent MAARA task instances sharing one inference server. A single task's
   DAG is almost fully sequential and can never fill a batch. Intra-task fan-out
   (`generate_fix` loops per file, `coder.py:95`) is a secondary source.
2. **One shared KV pool, per-model rates.** Requests carry a model tag;
   `CostModel.homogeneous=True` collapses everything onto the qwen rates as an
   ablation. The user asked whether to go homogeneous; the answer given was
   build hetero-capable and flip a flag, since `hetero` and `homo_llama3` are
   already arms in `data/outcomes/results.jsonl`.
3. **Active KV is pinned; only completed prefix entries are evictable.** No
   preemption of running requests. Consequence: the engine reserves prompt +
   full output at admission, because with no preemption a request that runs out
   of KV mid-decode has nowhere to go.
4. **Rent is emergent, never priced.** There is deliberately no per-token memory
   knob. **Do not add a memory-price config value** — it would let the retention
   policy read the answer it is supposed to infer.
   > **REVISED — see §13.** The *no-price* principle stands, but the mechanism
   > ("charge rent while a retained entry keeps a request waiting") has been
   > replaced by derived, eviction-pressure-scaled rent. Do not implement the
   > blocking-only version from this line.
5. **CPU stages consume wall-clock but hold no KV.** `navigate` (ChromaDB +
   nomic-embed-text) and `apply_and_test` (pytest) define the reuse gap.

---


## 13. Derived rent design (supersedes §3 decision 4, resolves §9.2)

Decided with the user before Milestone 1. This replaces blocking-only rent.

### The four rules

1. **Rent scales with eviction pressure.** The accrual rate is proportional to
   how fast the cache is actually being forced to evict, measured live from
   engine state.
2. **Rent for holding entry E = the recompute cost of the next victim that would
   be evicted instead of E** — the marginal victim, selected from current budget
   state, **excluding E itself**. If E were released, that victim would survive;
   the damage E causes by staying is precisely that victim's recompute.
3. **Rent = 0 when nothing else is evictable.** No victim means no pressure and
   no damage. This covers the case where the budget is full of pinned in-flight
   KV: retention had no lever, so charging it would blame the wrong mechanism.
   That time is reported separately as residual blocking.
4. **Rent accrues continuously over time**, throughout any interval where the
   budget is full and pending work would require an eviction — not only while a
   request is hard-blocked.

### Why not the obvious version

The first formulation set the rate equal to the marginal victim's recompute cost
directly. That gives the rate units of seconds per second, i.e. dimensionless,
so break-even lands at `E_recompute / victim_recompute` **time units** — and
which time unit is arbitrary. With measured numbers, two 7,710-token prefixes
(28.8s recompute each) break even at `t* = 1.0 second`, while the measured p50
reuse gap is ~21s. Ski-rental would evict almost everything immediately and lose
to LRU. Re-express the identical rule in milliseconds and it evicts 1000x
sooner. A rule whose behaviour depends on the unit you write it in has a missing
factor, and the missing factor is the eviction rate.

So: `rate = marginal_victim_recompute_seconds × evictions_per_second`. Accrued
rent then reads as "recompute-seconds actually being forced while I hold this",
which is directly comparable to E's own recompute cost — the ski-rental
comparison, in consistent units.

### Still no price knob

§3 decision 4's *principle* survives even though its mechanism is replaced:
there is no configurable memory price. Both terms in the rate are observed —
the victim's recompute comes from `CostModel` prefill physics (the same physics
the offline reference uses), and the eviction rate is counted from the engine.

One honest caveat: the eviction rate needs smoothing, so there is an EWMA time
constant. It is an **estimator** parameter, not a price — it changes how noisily
an observable is measured, not how much memory costs. It should still be listed
as a PLACEHOLDER and swept.

### RESOLVED in Phase 4 — retention unit and footprint scaling

Decided: **the retention unit is the block-quantised prefix head, plus a
separately-keyed tail**, following vLLM. KV is allocated in fixed blocks
(`EngineConfig.block_tokens`, default 16, PLACEHOLDER), and a finished call
leaves *all* its prompt blocks cached — not only the part a later call happens
to match. So each completed call leaves two things behind:

* a **head** entry: the shared prefix, rounded up to a whole block, genuinely
  reusable;
* a **tail** entry: every block past the divergence point, which costs exactly
  as much memory and can never be matched.

Online, a head and a tail are indistinguishable at creation time. That is the
decision retention actually has to get right, and modelling only the head would
have assumed the problem away.

Rent is charged on **held** bytes; benefit accrues on **useful** tokens:

```
rent_rate(E) = E.bytes x victim_recompute_s_per_byte x eviction_rate
break-even   = recompute_saved(useful tokens) / rent_rate(E)
```

The footprint question resolves cleanly, and the cancellation worry was real
but harmless. Rent and benefit are *both* linear in tokens, so entry size
cancels in every retention unit — break-even is the same for a 64-token and a
2,048-token entry. What does move it is `useful / held`:

| retention unit | useful/held | break-even at eviction rate r |
|---|---|---|
| hold only the shared prefix | 1.0 | `1 / r` |
| block-quantised head (chosen) | `shared / ceil(shared/16)*16` | `(useful/held) / r` |
| hold whole context | 0.138 (measured) | `0.138 / r` |

Measured on the committed trace, the median reusable prefix is **4 tokens**, so
the median head occupies a full 16-token block for a useful/held of **0.25**.
Block quantisation therefore quarters the break-even at this workload's median
and is negligible above one block. `results/figures/gap_histogram.png` draws
both bands: solid for the chosen unit, dotted for the exact-prefix reference.

### OPEN QUESTION as originally raised (kept for the record)

As written, rule 2 charges a 20,000-token entry exactly the same rent as a
500-token one, because the rate depends only on the *victim's* recompute cost
and the eviction rate. That is wrong in the direction that matters: a large
entry displaces proportionally more of the budget, so it should accrue rent
faster. Two candidate forms:

```
(a)  rent_rate(E) = (E.bytes / victim.bytes) × victim_recompute × eviction_rate
(b)  rent_rate(E) = per_byte_rent_price × E.bytes
     where per_byte_rent_price = (victim_recompute / victim.bytes) × eviction_rate
```

These are algebraically identical; (b) just names the intermediate quantity.
Note that under either form, if E and the victim are the same size the rule
reduces to rule 2 as originally written, so this is a generalisation rather than
a contradiction.

Unresolved consequence to check when implementing: E's break-even time becomes
`E_recompute / rent_rate(E)`. Both numerator and denominator scale with E's
size, so the scaling may cancel and leave break-even size-invariant — which
would defeat the purpose. Whether it cancels depends on whether recompute is
linear in tokens (it is, under the current CostModel) and whether bytes are
linear in tokens (they are). **Decide and justify at Phase 4, with a worked
example at two entry sizes.**

### Consequence for the competitive-ratio claim

Classic ski-rental assumes **constant** rent. Derived rent is not constant: it
moves with load and with which victim is currently marginal. The textbook
2-competitive guarantee therefore does **not** transfer, and must not be claimed.
Measure the realized ratio against the offline reference instead, and label that
baseline "offline reference" rather than "optimal" — exact OPT under continuous
batching with load-dependent rent is not proven here.

---


## 8. Bugs already fixed — do not reintroduce

1. **Double-counted KV.** Full output was reserved at admission *and* grown per
   decoded token, inflating usage ~2x. Growth removed; reservation is correct
   under the no-preemption decision.
2. **Iteration-averaged metrics.** `mean_batch` / `mean_kv_util` averaged per
   iteration, so a 99s prefill and a 0.035s decode step counted equally. Symptom
   was resident reading identically (13.95) across three very different configs.
   Now time-weighted; the KV split moved 407/810 → 532/726MB.
3. **`prefill_sample_tokens` read from the wrong response object** in
   `measure_rates`.
4. **ollama `num_ctx` truncation** (see §4).

Items 5-9 were fixed during Phase 4 and 5b and were never written into this
list; they were recovered from the session record during the Milestone 2
cleanup. Several are the kind that fail silently, so they are worth reading
before touching the engine.

5. **Eviction double-count, causing deadlock.** The eviction loop compared an
   accumulating `freed` total against `need - self.free_bytes`, but
   `free_bytes` already grows as entries are evicted, so the same space was
   counted twice and the loop concluded it had freed enough when it had not.
   Fixed to test live free space only.
6. **Pressure snapshot taken *after* admission.** `_try_admit` had already
   drained the evictable set by the time rent was assessed, so rent was charged
   on almost nothing — 2 of 7,284 iterations. The snapshot is now taken before
   admission. Symptom: retention policies scored identically for no visible
   reason.
7. **Rent did not conserve under block granularity.** Charging every retained
   entry the full victim rate inflated total rent by roughly 100x (observed
   ratios 131-723 against the hindsight denominator). Rent is now normalised by
   the evictable pool, so what the holders are charged sums to what the
   eviction actually cost.
8. **Self-blocking deadlock.** A request could be blocked by the very blocks it
   wanted to reuse, since those blocks were held and therefore not evictable.
   Fixed with a last-resort eviction of held blocks plus a worst-case
   feasibility guard that refuses to start a run whose largest request cannot
   fit the whole budget.
9. **`used_bytes` was the hot loop, not rent accrual.** Profiling, rather than
   the assumption, located it: 1.29 billion generator calls, 115s of a 167s
   run. Made incremental via `_entry_bytes` / `_evictable_bytes`; the 8x case
   went 44.3s to 18.7s. The lesson is in the ordering — the suspected cause was
   rent accrual, which was 8s of the same run.
10. **The evaluator crashed on a reply without a `SCORE:` line** (fixed
    2026-10-07, commit `d6d0f80`). `evaluate_plan` indexed the first `SCORE:`
    line before checking one existed, so the intended default of 5 was
    unreachable and a malformed llama3.2 reply killed the task before any fix
    attempt — 7 of 60 `shared_first` runs (FINDINGS §14b). Now
    `_parse_score`: any reply the old code accepted parses identically, and a
    missing or non-integer score falls back to `DEFAULT_SCORE`. Prompts are
    unchanged, so traces collected before and after stay comparable. Do not
    reintroduce an unguarded index into model output anywhere in `maara/`.

---

## 16. Test collection is scoped to `tests/`

`pyproject.toml` sets `testpaths = ["tests"]`. This is not cosmetic.

Before it, a bare `pytest` reported **240** tests. Sixteen of those lived in
`data/demo_target/test_target.py` — the fixture the orchestrator demo is
supposed to repair. They are the agent's target, not a check on this project,
and counting them conflated "the project works" with "the demo fixture happens
to be in a fixed state".

The hazard is the direction it fails in: a demo run that left `target.py`
broken, or an agent run that half-patched it, would have shown up as failures
in the project's own suite with no connection to the cause. The real count is
**224**.

Any future fixture must live under `data/`, never under `tests/`.

## 17. A prompt-cache restore copies the saved state, it does not consume it

`OllamaPromptCache.access` used to `pop()` the entry it restored from the host
RAM cache. That was a bug, and an invented constraint rather than a modelled
mechanism: loading a saved state into a slot gives the host no reason to delete
its own copy of it.

It mattered more than it looks. Each HumanEval task is replayed under all four
configs, so an `issue_parser` prompt recurs **byte-identically about four
times**. Consuming the entry on the first restore capped a 4x recurrence at a
single hit, and `issue_parser` is where almost the whole prediction gap lived.

Effect of the fix alone, with no other rule changed:

| trace | before | after |
|---|---|---|
| `original` | 36 of 89 hits | **68 of 89** (92.3% agreement, 0 false positives) |
| `shared_first` | 6 of 78 | 7 of 78 |

`shared_first` barely moves because there the in-place slot path fires on
almost every call, so a saved state is never looked up at all. Closing that
needs a best-match lookup, which is **not** adopted here -- see OPEN_ISSUES
section 15.

The general lesson is worth keeping: when a model underpredicts, check which of
its rules are measured mechanism and which are conveniences the author
invented. This one was never observed anywhere, in a log line or a probe.

## 18. SUPERSEDED by §19 — "the cache model's rule is frozen" (v1)

> Freeze v1 encoded the retracted "0.10 destroys the slot" rule. Commit
> `9911011` is left in place and history is not rewritten. Re-frozen as v2 in
> §19 below, on a rule read out of llama.cpp's source rather than inferred.


> **The frozen rule described below is superseded as of 2026-10-04.** It was
> built on the claim that overlap >= 0.10 destroys the displaced state, which
> the affinity rerun refuted (see FINDINGS §19). The freeze commit `9911011`
> is deliberately left in place and history is not rewritten — the point of a
> freeze is that it stays auditable, including when it turns out to have
> frozen the wrong rule.
>
> **No re-freeze until `--affinity-sweep` pins the restore-to-trim cutoff.**
> Freezing again before that would just repeat the mistake at a new value.
> `analysis/validate_cache_model.py` still works and its no-tuning discipline
> still applies; what it validates is simply known to be the wrong rule, so
> running it now would waste the held-out trace. Do not collect that trace yet.


About thirteen rule variants were scored on `original` and `shared_first` while
the Ollama cache mechanism was being worked out. That is enough selection
pressure that agreement on those two traces is no longer evidence about the
mechanism — it is partly evidence about how many things were tried.

So the rule and its parameters are frozen in `simulator/ollama_cache.py` under
a `FROZEN RULE` block, as of 2026-10-04, with the fit-set scores recorded
beside them (`original` 68/89, `shared_first` 7/78). The freeze is committed
**before** the held-out trace is collected, so the ordering is checkable in
git rather than asserted.

`analysis/validate_cache_model.py` runs the frozen rule against one trace and
has no tuning knobs. It refuses outright to run on either fit trace unless
overridden, and prints the fit-set scores alongside its result so the two
cannot be confused.

### The rule after seeing the held-out number

Write it down. Changing `SIM_THRESHOLD`, `DEFAULT_RAM_BYTES` or the rule in
response to that output turns the held-out trace into a third fitting set, and
there is no fourth chance at a reasonable price — a full collection run is
hours of GPU time.

Two variants are recorded as deliberately **not** adopted, with their better
fit-set scores, so nobody later mistakes them for untried ideas:

| variant | `original` | `shared_first` | why not adopted |
|---|---|---|---|
| best-match host-cache lookup | 84/89 | 76/78 | probe-consistent only under an untested size hypothesis |
| size gate on save-on-reuse | — | — | does no work on the traces at all; exists only to reconcile with the probe |

`probes/ollama_cache_probe.py --affinity-short` is the measurement that decides
both.

## 19. Cache rule re-frozen as v2, on cited source

The rule in `simulator/ollama_cache.py` is frozen again, this time on the
upstream constant rather than on a fit to two traces:

```c
if (f_keep < 0.5f) { update_cache = true; }     // llama.cpp server-context.cpp
```

`KEEP_GATE = 0.50`, `SIM_THRESHOLD = 0.10` (inert at `-np 1`),
`DEFAULT_RAM_BYTES = 8 GiB`, `DEFAULT_TOKEN_LIMIT = 8192`, `HIT_FRAC = 1/3`.
Scores at freeze: `original` 84/89 (97.8%), `shared_first` 76/78 (89.3%), with
exact-prompt repeats and partial prefix reuse reported separately
(FINDINGS §20).

### Why this freeze is on firmer ground than v1

v1 was chosen by scoring ~13 variants on two traces. v2's gate is a literal
constant in the shipped implementation, independently reproduced from
`server.log` (0.8% vs 64.8% search rate either side of 0.5), and consistent
with the affinity sweep. Three sources, none of them a fit.

### What is still not settled, and the held-out trace still matters

Agreement of 97.8% on `original` is **not** evidence the model generalises:

- Over half those hits (46 of 89) are exact-prompt repeats produced by the
  harness replaying 15 tasks under 4 configs (OPEN_ISSUES §20). Any
  prompt-keyed cache gets them.
- `shared_first`'s 21 false positives are mostly an artefact of its degraded
  segment (FINDINGS §18), so its 89.3% is not a clean failure either.
- The char-for-token approximation leaves 1.1% / 5.6% of calls near the gate
  where the model may pick the wrong branch.

So the held-out trace is still the test, and for the same reason as before:
these two traces measure the benchmark repeating itself. Do not tune
`KEEP_GATE` against them.

### `save_on_reuse` removed

It was a free parameter asking whether an in-place reuse destroys the displaced
state. The source answers it: the save is a consequence of `f_keep < 0.5`, not
an independent choice. A knob that encodes a settled question is a liability.

## 20. Repair tasks are stub bodies over canonical HumanEval, not mutations

The held-out task set is built by `experiments/build_tasks.py` from canonical
HumanEval:

- source `https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz`
- sha256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`
- 164 problems; keys `task_id`, `prompt`, `canonical_solution`, `entry_point`, `test`

Each task is `{task_id, issue, buggy_code, test_code}`, the schema already in
use. `buggy_code` is the problem's own `prompt` — imports, signature, full
docstring — plus a trivial wrong body: return the first argument, or a default
of the annotated return type. `issue` comes from the docstring's first sentence
through one fixed template. `test_code` is the problem's `check()` asserts
rewritten to `from target import f` / `def test_N()`.

### Why stub bodies rather than mutating the canonical solution

1. **A mutation's difficulty is unknown and unstable.** Flipping an operator
   can produce a bug that is trivial, impossible, or silently equivalent. A
   stub is uniformly "the body is missing", so difficulty comes from the
   *problem*, which HumanEval already characterises, rather than from how the
   mutation happened to land.
2. **Mutations can be semantically null.** Some produce code that still passes
   the tests, which the validator would catch but only after wasting the slot.
   A stub returning the wrong type cannot accidentally be correct.
3. **The failure mode matches the workflow.** MAARA is a repair pipeline: the
   issue parser reads a bug report, the coder writes a fix, tests run. A
   missing body gives the coder the real task — implement the docstring — and
   makes the retry loop meaningful. A one-character mutation is often found in
   a single attempt, which exercises neither the debugger nor the retry path
   that workstream 2 is about.
4. **It keeps the docstring in the prompt.** That matters here for a reason
   beyond realism: the docstring is most of the prompt's bulk, so prompt sizes
   and prefix structure stay comparable to the Milestone 1 tasks.

The cost is honesty about what the benchmark measures: these are
implement-from-spec tasks, not localise-a-subtle-defect tasks. Pass rates are
not comparable to defect-injection benchmarks and should not be reported as if
they were.

### Validation is a gate, not a check

Every task must have its tests **FAIL** on `buggy_code` and **PASS** on the
canonical solution. 55 were selected, **54 kept**, 1 dropped:

- `HumanEval/32` (`find_zero`) — canonical solution fails its own tests. Its
  check uses randomised coefficients and a tolerance the reference does not
  always meet.

### Selection

`HumanEval/1-15` are excluded: they are the fit set behind both committed
traces, so including them would not be held out. `HumanEval/16-30` are pinned
(already written for the harness, never run), then the remainder is sampled
round-robin across terciles of canonical-solution length, so the set is not
all one-liners. Final: **54 tasks, ids 16-158**, 371 tests, median
`buggy_code` 12 non-blank lines (range 6-32).

### Tasks moved out of code

`data/tasks/humaneval_original30.json` (the Milestone 1 tasks, extracted
verbatim) and `data/tasks/humaneval_new.json`. `MAARA_TASK_SET` selects:
`original` (default, `HumanEval/1-15` — reproduces Milestone 1 exactly and is
asserted byte-identical), `original30`, or `new`. The raw dataset is not
committed; the URL and hash above make it reproducible.

## 21. Reuse is annotated per block across the whole trace, not per prefix_key

`_annotate_measured` keys reuse edges on `Node.prefix_key`. On the trace path
that key is built as `f"{run_key}:{model}:{digest}"`
(`simulator/trace_loader.py:362`), so it **embeds the task-run index** and can
only ever describe reuse within one run. The engine, however, matches on
`Node.block_keys` — content hashes seeded only with the model
(`simulator/workload.py:400-413`) — which collide freely across runs.

So every block genuinely reused by a *different* task run carried
`true_gap_s = None`. `OfflineReferenceRetention.on_complete` returns `False` on
that and `on_tick` drops it immediately, which means **the offline reference
retained nothing at all on the trace path** — it behaved as `evict-now`:

| trace | buggy recompute | fixed | rent (buggy → fixed) | ratio (buggy) |
|---|---|---|---|---|
| `original` | 74.0s | **20.0s** | 0.0s → 5.7s | `nan` |
| `holdout_qwen` | 140.3s | **19.4s** | 0.0s → 8.3s | `nan` |

A "reference" that retains nothing is not an upper bound on anything. Any
trace-replay figure quoting `offline-ref`, `hindsight_cost_s` or
`realized_ratio` from before this fix should be discarded. The grid's shared
keep-all denominator is computed from `_key_rent`/`_key_recompute` rather than
`true_gap_s`, so that column survives — but the `offline-ref` row did not.

### The fix

`trace_loader._annotate_block_reuse(tasks)` runs once after all tasks are
built. It groups emissions by `(model, block_key)` on a **global** clock
(`task.arrival_time + within-task offset`), links consecutive users, and stores
the gap in a new `Node.block_reuse_gap` dict. The engine prefers it and falls
back to the node-level field:

```python
true_gap_s=node.block_reuse_gap.get(k, node.next_reuse_gap_s)
```

Scale of what was missing, from `LoadedTrace.validation`:

| trace | reused block keys | within-run edges | **cross-run edges** |
|---|---|---|---|
| `original` | 624 | 817 | **821** |
| `holdout_qwen` | 1154 | 2108 | **400** |

On `original` roughly half the reuse edges were invisible. Normalising arrival
times afterwards shifts every clock by one constant, so the gaps are unaffected.

### Why the synthetic path is untouched

The generator bakes `t{task_id}` into every segment ident
(`simulator/workload.py:462-465, 472-473`), so cross-task block collisions
cannot occur and `block_reuse_gap` stays empty — the node-level fallback then
applies. Verified: four fixed-seed synthetic runs are **byte-identical** across
this change, and a test asserts the map stays empty for `generate_trace()`.

`HintBuilder` still reads the node-level `next_reuse_gap_s`, so
`ski-rental-pred` gets a within-run-only hint. That is left as-is deliberately:
the hint is meant to be noisy and partial, and the oracle is the thing that had
to be correct.

## 22. Ollama is a cache MODE, not a retention policy

I first argued that `LRURetention` was a faithful stand-in for Ollama's prompt
cache, on the grounds that a block-level cache has no "slot" for the `f_keep`
gate to act on. That was wrong, and it erased the one behaviour that matters.

Ollama's defining behaviour (FINDINGS §20) is that **only one prompt is
resident per model**, and similarity decides whether the displaced one
survives: at `f_keep >= 0.5` the slot is reused in place and the rest of the
previous prompt is discarded with **no saved copy**. That is precisely what
loses a coder's state across a retry (FINDINGS §21). A retention policy chooses
what to keep *within a shared cache*; it cannot express "there is only one
slot". So this is `EngineConfig.cache_mode`:

- `"block"` — vLLM-style shared, content-keyed block cache (the default)
- `"ollama"` — one slot per model plus a host-RAM tier

Measured, `holdout_qwen` 1x, util 0.8, batch 4, KV 512MB, FCFS:

| mode / policy | recompute | rent | total | slot reuse | discarded | restores | host saves |
|---|---|---|---|---|---|---|---|
| block / keep-all | 13.4s | 146.0s | **159.4s** | — | — | — | — |
| block / ski-rental | 18.4s | 166.2s | 184.6s | — | — | — | — |
| **ollama / keep-all** | **122.1s** | 117.1s | **239.3s** | 1 | 3724 | 184 | 317 |

So today's server pays **9.1x the recompute** and **1.50x the total cost** of a
shared block cache. That gap is the thesis's opportunity, and it is now
measured rather than asserted.

### The host tier is free, deliberately

`ollama_host_mb` defaults to 8192, MEASURED from the server log
("prompt cache is enabled, size limit: 8192 MiB"). It is **host** memory, so it
does not consume `kv_budget_mb` — the Ollama baseline gets a second tier the
block-cache policies do not have. That is generous to the incumbent, which is
the right direction for a baseline, but it must be stated wherever the
comparison is reported or Ollama looks artificially strong.

### Three bugs this shook out, all from the same root

The matcher is called more than once per admission attempt, and the attempt can
still fail.

1. **Mutation in the matcher.** The first version updated the slot inside
   `_match_ollama`, so a failed admission left the slot naming blocks that were
   never allocated — `KeyError` two requests later. Split into a pure
   `_match_ollama` plus `_commit_ollama`, called only after admission succeeds.
2. **Prefill and residency were conflated.** A host restore skips prefill but
   the blocks are *not* in GPU KV; loading a saved state costs bytes. The
   resident match and the restored-block discount are now tracked separately
   (`Request.ollama_restored_blocks`).
3. **The slot is not proof of residency.** Eviction pressure can remove a slot
   block. `f_keep` is still computed against the whole slot, as llama.cpp does,
   but the *match* is capped by what is actually resident.

Block mode is unchanged: four fixed-seed synthetic runs are byte-identical.

## 23. OPEN QUESTION — the derived-rent model is policy-invariant in this regime

Not changed, recorded. Revisiting §13 mid-grid would invalidate everything
measured so far, so this is logged for a deliberate decision later.

Derived rent charges a holder in proportion to its bytes times the victim's
recompute-per-byte, normalised by the evictable pool. The consequence, measured
(FINDINGS §24): in the regime where a retention policy actually operates, total
rent barely moves with the policy — 146.0s under keep-all against 166.2s under
ski-rental — and only collapses to zero when almost nothing is held at all
(evict-now). Releasing **shifts** rent onto whoever still holds rather than
removing it.

So an objective of `recompute + rent` is dominated by a term the policy cannot
reduce, and it misranks: it rates `evict-now` 20% better than `keep-all` while
evict-now is 17% worse on p95 and 9.5x worse on GPU work.

If this is revisited, the question to answer first is what rent is supposed to
represent. As a *shared pressure* signal it is behaving correctly and simply
is not a policy-reducible cost. As a *private holding cost* it would need to
charge a holder only for pressure it personally caused, which is a different
model and would need its own validation.

Until then: report latency and GPU recompute, treat rent as a diagnostic, and
do not use `recompute + rent` to rank policies.

## 24. Paired comparisons use common random numbers

Retention and scheduler comparisons are paired: every policy sees the same
seeds, and a difference is reported as a paired difference with a confidence
interval. On this trace seed variance is large — plus or minus 20% on composite
cost (FINDINGS §27) — so unpaired comparisons cannot resolve the effects being
measured.

Verified, and one channel was broken:

| channel | status |
|---|---|
| task arrival times | **common** — identical for a fixed seed, different across seeds |
| estimator noise | **common** — already cached per `req_id`, not per call |
| hint noise | **was NOT common**; fixed |

`HintBuilder` drew from one shared RNG, so draws followed `hint_for` call
order, which follows the admission sequence, which differs by policy. The same
block therefore received a different predicted gap under each policy: of 12
blocks hinted under both `ski-rental-pred` and `pred-belady`, **0%** got the
same noised value. That turns a paired comparison into an unpaired one and
reintroduces precisely the variance pairing exists to remove.

Noise is now derived from `(seed, block key)`, so a prediction is a property of
the block rather than of when it was queried — which is also what a real
predictor would be. After the fix, 50 of 56 blocks hinted under both policies
agree exactly.

The remaining 6 are **not** a CRN violation. The multiplier is deterministic
per `(seed, key)`, so a difference means the *base* `true_gap` differed: the
same block key can be created by a different node under a different eviction
order, and each node carries its own per-block gap. That is a structural
property of replay, not randomness, and it is why the end-to-end figure is 89%
rather than 100% while the unit-level noise is exactly reproducible.

## 25. Every experiment writes a run manifest, and a calibration mismatch blocks the run

FINDINGS §23 was retracted because an arrival rate calibrated at `batch=16`
was used by a table run at `batch=8`, about 2.2x the intended load, and nothing
in the output said so. `experiments/manifest.py` makes that class of error
fail rather than be remembered.

**The convention.**

- Each driver builds a `Manifest` at startup: git hash, dirty flag and dirty
  paths, and every parameter the run used.
- Every calibrated value is recorded as a `Calibration` **with the
  configuration it was calibrated under**, read from the bound call with
  defaults applied (`bound_args`), not from what the driver believes it
  passed. The §23 rate came from a default nobody passed.
- The achieved value is stored next to the target (e.g. the busy fraction the
  calibrated rate actually produced), not only the requested one.
- Before any engine time is spent, and again immediately before the
  measurement loop, the run config is checked against each calibration on
  `trace_set, scale, kv_mb, max_batch, cache_mode`, plus `util` for the
  arrival rate and `arrival_rate` for anything calibrated at a given rate. A
  mismatch, or a key missing on either side, raises `CalibrationMismatch` and
  no results are written. It is deliberately not a `RuntimeError`, because
  both drivers swallow those around engine runs.
- The manifest is written into the results JSON under `"manifest"`.

Wired into `run_scheduler_table` (arrival rate, slack factor) and
`run_retention_grid` (per cell: arrival rate and the hindsight denominator,
which is equally a measured value valid only at its rate and config).

**FINDINGS entries must cite a manifest.** A number quoted in FINDINGS names
the results file it came from, and that file must carry a manifest whose
`verified_runs` covers the configuration being claimed. A number without one
is provisional.

**The guard found a live instance on its first run.** `run_scheduler_table`
calls `calibrate_rate` without `batch` or `trace_set`, so the rate is always
calibrated at `batch=8` on `holdout_qwen`, whatever the table runs at.
`--batch 16` and `--trace-set holdout_llama` now both fail before any engine
run. The calibration logic was deliberately not changed here; passing those
arguments through is a separate, owner-approved fix. Consequence for the
record: FINDINGS §27's batch-16 rows used a rate calibrated at batch 8, so its
statement that the rate was calibrated at the batch the table runs at holds
only for the batch-8 rows.

**Fixed and re-measured 2026-10-07.** The driver now passes `batch` and
`trace_set` to `calibrate_rate`. Recalibrated at batch 16 the rate is
0.034278564453125/s, bit-identical to the batch-8 rate, so the bisection
lands on the same grid point. The 20-seed batch-16 rerun
(`results/scheduler_table_holdout_qwen_b16.json`, manifest verified at
`max_batch=16`) gives FCFS p95 132.5s and paired p95 differences against FCFS
of lsf +0.04s [-0.06, +0.14], edf +0.04s [-0.05, +0.14], oracle +0.08s
[-0.78, +0.93], all not significant. **§27's batch-16 conclusion stands**; the
process was wrong, the number was not. The driver now also writes per-seed
p95 and these paired intervals itself, so §27-style tables are reproducible
from its output.

## 26. The `final` task set is sealed until the policies are frozen

Built 2026-10-07 (commit `7d4a96c`) as `data/tasks/humaneval_final.json`,
selected with `MAARA_TASK_SET=final`: **every remaining HumanEval problem**,
92 tasks after validation.

**Why it exists.** `holdout_qwen` was held out for the *cache model*, but the
retention and scheduler work has since been developed, calibrated and debugged
on it (FINDINGS §24-27, OPEN_ISSUES §26-27). For policy claims it is now a
fitting set. Final policy numbers need data none of that work has seen, by the
same reasoning that froze the cache rule before its held-out test (§18, §19).

**How it was built.** Same source, sha256, stub-body construction and
validation gate as §20. It excludes the 30-task Milestone 1 file and the 54
held-out tasks, **and HumanEval/32**, which was selected and dropped there:
it was looked at, so it is not untouched. 94 candidates, 2 dropped:
HumanEval/38 and /50, whose `check()` calls the `encode_*` helper that the
converted test does not import. All 164 problems are accounted for, and
`tests/maara/test_task_sets.py` asserts the sets stay disjoint.

**The rules.**

1. Collect it under the **same two configs, models and strategy** as the
   held-out traces (`single_agent` and `homo_llama3`, `zero_shot`), one
   process per config, Ollama restarted between, never resumed (OPEN_ISSUES
   §21).
2. The **cache-model** validation may run on it immediately: that rule is
   frozen (§19) and was never tuned on any held-out data.
3. **Retention and scheduler drivers must not run on it** until the policies
   and their parameters are frozen in a commit, with the freeze commit
   preceding the first run on `final` so the ordering is checkable in git.
   One run, reported as it comes out. Tuning in response to it turns it into
   another fitting set, and HumanEval has no problems left to replace it.
4. Pass rates on it are implement-from-spec rates and carry §20's caveat.

## 27. Cost constants live in a named hardware profile; every finding so far is a Mac result

Added 2026-10-07 (brief 02). `simulator/workload.py` now holds `PROFILES`, and
`CostModel` takes the machine-dependent constants (per-model rates, KV bytes
per token, `batch_decode_coeff`, `iteration_overhead_s`) from the one named in
`CostModel.profile`. `provenance()` prints the profile on its second line.

**Why.** The prefill-to-decode ratio decides whether retention has anything
to win (FINDINGS §24 put recompute at about 1% of makespan), and it differs by
several times between a Mac and a discrete GPU. Until now nothing in the code
or output said which machine a number described.

**The convention.**

1. `mac_ollama_measured` is the default and reproduces the previous behaviour
   exactly: the two fixed-seed engine runs in brief 02 are byte-identical
   apart from the new provenance line. **Every result in FINDINGS to date is a
   `mac_ollama_measured` result**, and should be quoted as one.
2. `gpu_4060` is a **PLACEHOLDER**: the Mac numbers copied so it runs, every
   field marked unmeasured. It is not an estimate of an RTX 4060. Workstream 4
   replaces its values with measurements, and flips each `measured` flag only
   for a value it actually measured.
3. Unchanged labels: `batch_decode_coeff` stays PLACEHOLDER at 0.15;
   llama3.2's KV bytes per token stays DERIVED (OPEN_ISSUES §23).
4. A constant passed explicitly (the decode-coefficient sweep) overrides the
   profile, and `provenance()` reports it as OVERRIDDEN rather than hiding it.
5. `data/calibration/measured.json` loads only under the Mac profile and only
   if its `backend` is `mac-metal`; otherwise construction fails. Calibrating
   another machine must add a profile, never overwrite the Mac's.
6. `test_mac_profile_values_are_pinned` pins the Mac numbers. Changing one is
   a deliberate act: update the test in the same commit and record why here.

Not done here, by file ownership: the engine CLI and the experiment drivers
still build `CostModel()` (the Mac default) and do not yet write
`hardware_profile` into their manifests.

## 28. `batch_decode_coeff` is measured on the Mac: 0.841, not 0.15

Measured 2026-10-07 by `probes/batch_decode_probe.py` (owner-run): Ollama
0.35.1 with `OLLAMA_NUM_PARALLEL=8`, qwen2.5-coder:7b, n = 1, 2, 4, 8, three
repeats of 256 decoded tokens each. Output `data/calibration/batch_decode.json`.
Every round overlapped fully (overlap = n), so the requests really ran
concurrently, not queued.

| n | per-token per request | aggregate decode |
|---|---|---|
| 1 | 34.2 ms | 28.5 tok/s |
| 2 | 49.7 to 68.7 ms | 28.4 to 38.7 tok/s |
| 4 | 108 to 118 ms | 33 to 36 tok/s |
| 8 | 237 to 240 ms | 33 tok/s |

Least-squares fit 0.841 (per n 0.82 / 0.79 / 0.85; max residual 0.16, mostly
the bimodal n=2 rounds). **Batching on this stack is nearly serial**: eight
concurrent streams decode only about 15% faster in total than one.

**The decision.** `mac_ollama_measured` now uses 0.841, marked MEASURED, and
`test_mac_profile_values_are_pinned` pins it (§27 item 6). This changes every
simulator number from here on; FINDINGS sections written before this commit
used 0.15 and should be quoted as such.

**Why the conclusions survive.** The sweep (OPEN_ISSUES §9.3) was rerun at
0.841: recompute 0.6% of makespan, ski-rental still significantly worse than
keep-all on recompute and tied on p95, scheduler choice 1.6%, LSF tied with
FCFS. Capacity at util 0.8 falls (rate 0.0323/s to 0.0265/s) and p95 roughly
doubles.

**Caveat to carry into the writeup.** 0.841 describes Ollama's parallel
slots on Metal, not the hardware's batching limit and not a vLLM-class
continuous batcher, which batches decode far more efficiently. The simulator
models vLLM semantics on Mac costs, so on this profile batching is costed as
Ollama does it. The GPU profile must measure its own value.

## 29. The GPU is one RTX 5060 Ti with 16 GB, and that is the ceiling

Set by the owner 2026-10-07. The expected GPU is an RTX 5060 Ti with 16 GB,
replacing the RTX 4060 with 8 GB assumed in brief 02 and ROADMAP §4. **No work
in this repo may need more than this card**: no second GPU, no larger VRAM,
no datacenter hardware. A step that does not fit is scaled down or dropped.

Consequences, in detail in `docs/GPU_TODO.md`:

1. The placeholder profile `gpu_4060` (§27 item 2) is renamed `gpu_5060ti`.
   It is still the Mac values copied, every field unmeasured.
2. vLLM runs one model per server, AWQ int4, `--max-model-len 16384`. bf16
   7B weights (about 15 GB) leave no room for KV and are out of scope.
3. Ollama on CUDA uses the same model tags as the Mac, so quantisation
   matches across profiles.
4. The regime map's synthetic long-context scales stay simulator-only.
5. Both probes take `--backend`. Anything other than `mac-metal` writes to
   `data/calibration/<backend>/`, so a GPU calibration cannot overwrite the
   Mac's files or be labelled as Metal.

## 30. llama3.2's KV bytes per token is measured: 114,688, as derived

Confirmed 2026-10-07 from `~/.ollama/logs/server-3.log`, a Mac server log
from the earlier collections (found by the brief-03 session, rechecked here).
Two independent lines agree:

- the KV buffer at load, for "Llama 3.2 3B Instruct":
  `llama_kv_cache: size = 448.00 MiB (4096 cells, 28 layers, 1/1 seqs), K (f16): 224.00 MiB, V (f16): 224.00 MiB`,
  i.e. 448 x 1048576 / 4096 = **114,688 B/token**, exactly the
  architecture-derived value;
- the prompt-cache size estimate, 8192 MiB for 74,845 tokens, about
  114,770 B/token (the same estimate gives 57,358 for qwen, matching its
  confirmed 57,344).

`mac_ollama_measured` now lists both models in `kv_measured`, which revises
§27 item 3. The value is unchanged, so no result moves. This closes
OPEN_ISSUES §23; `gpu_5060ti` still says DERIVED until checked on that card.

## 31. Policies frozen for the sealed `final` run; the headline is program-level ordering

Set by the owner 2026-10-15, the Milestone 2 deadline. **This commit is the
freeze commit that §26 rule 3 requires**: every run on `final_qwen` or
`final_llama` sets `MAARA_POLICIES_FROZEN` to it, and no policy, parameter or
driver default changes between it and those runs.

**Headline.** Scheduling under load, stated as **program-level ordering**:
finishing older workflows first (`program-fcfs`) is the supported claim
(FINDINGS §29–31). LSF is an analysed negative result against that baseline,
and retention is a cost mechanism and an analysed negative result at 1x
context (FINDINGS §24–26). Retention at 8x and above (OPEN_ISSUES §27) is run
on the fitting set only and reported as preliminary, whatever it shows.

**What is frozen.** The code at this commit: the schedulers, estimators and
retention policies in `simulator/policies.py` with their defaults, including
`program-fcfs-cap` with `EPS = 0.05`. That rule has been run only in a
3-seed `--quick` smoke check on `holdout_qwen` (batch 8, two loads), to see
that the driver runs it; nothing was tuned on it. Its first full map is the
sealed one, so it is a pre-registered rule, not a tuned one. The measured profile `mac_ollama_measured`, coefficient
0.841.

**The runs, in order, each once, reported as they come out.**

1. Scheduling, primary:
   `scheduling_map --trace-set final_qwen --slo-mix
   tight:1.0:0.3,standard:2.0:0.4,loose:4.0:0.3 --workers 6`
   with every other flag at its default (20 seeds, batches 2 4 8 16, KV
   1024 MB, LRU, `estimate` deadlines, `busy` load axis): the FINDINGS §30
   setup on unseen tasks.
2. Retention, primary:
   `run_retention_grid --trace-set final_qwen --scales 1 --headrooms
   0.25,0.5,1.0 --util 0.8 --batch 4 --seeds 20`: the 1x headroom grid.
3. Secondary, if time allows: run 1 on `final_llama`.

**Pre-registered comparisons** (paired 95% CIs, no multiple-comparison
correction, so about 1 in 20 cells is expected to separate by chance):

- S1: `program-fcfs` vs `fcfs` on p95. Expected: better at batch 2–4 from the
  operating load up.
- S2: `lsf` vs `program-fcfs` on p95. Expected: never significantly better.
- S3: `program-fcfs-cap` vs `program-fcfs` on p95. No expectation; first
  evaluation.
- R1: `ski-rental` and `ski-rental-pred` vs `keep-all` on recompute and p95 at
  1x. Expected: not better on either.

A result against expectation is reported as such, and no policy is changed in
response: HumanEval has no problems left for another test set (§26).

**Amendment, same day, before any policy result.** The first attempt of run 1
at `8a8b6c9` was stopped during batch 2's load calibration. It printed only
rate calibration lines, all at the search floor (0.0070/s, busy 1.00), and no
cell had run. Cause: one call in `final_qwen`, `issue_parser` on HumanEval/75,
generated **40,960 tokens over 1,589 s** (a runaway with no stop), and needs
2,253 MB of KV against the 1,024 MB budget, so every engine run raised, and
`calibrate_rate` counts an error as busy 1.0. The owner chose to **drop that
task** from the simulator runs on `final_qwen` (91 tasks), rather than cap its
output or raise the budget, which would change the setup from FINDINGS §30.
It is the only call over 1,000 output tokens in that trace. The exclusion is
in code (`TRACE_SET_EXCLUSIONS`), recorded in every manifest, and tested. The
runaway itself is a MAARA finding: `issue_parser` sets no output limit.
`final_llama` needs no exclusion (its largest call is 2,286 tokens). The
aborted log is kept as
`results/sealed/ABORTED_8a8b6c9_scheduling_map_final_qwen.log`. The commit
holding this amendment is the freeze commit for runs 1-3; nothing else
changed.
