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

---
