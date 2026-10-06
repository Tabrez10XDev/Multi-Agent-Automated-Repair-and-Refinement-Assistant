# The story so far

A first-principles walkthrough of the project, from the project description deck to 2026-10-06.

**How to read the citations.** "(FINDINGS §21)" means `docs/FINDINGS.md` section 21. "(DECISIONS §22)" and "(OPEN_ISSUES §27)" work the same way. Section numbers are the original HANDOFF numbers, so they skip around. "(commit 8342a6e)" points to a commit message. "(trace_stats)" means `results/trace_stats_original.json`. Anything marked **[not in records]** is a number I could not trace to a written record. Section 6 lists every one of these.

---

## 1. The 2-minute version

**The problem.** An agent workflow like MAARA calls an LLM server many times per task. Those calls follow a known graph and repeat a lot of text. A general-purpose server can't see any of that structure. We asked whether a server that *could* see it would do better at two things: deciding which cached work to keep in GPU memory (**retention**, using ski rental), and deciding which waiting request to run next (**scheduling**, using least slack first).

**What we built.** We added per-call tracing to MAARA. We built a trace-driven simulator of a batching LLM server with a vLLM-style cache. We implemented the scheduling and retention policies. We also wrote a model of Ollama's real cache, taking its rule from llama.cpp's source code, and validated that model on tasks it had never seen.

**The three findings that matter most.**
1. **At MAARA's scale, neither mechanism changes latency.** Recompute is about 1% of simulated runtime (FINDINGS §24). Every scheduler, including one given perfect information, is statistically tied with first-come-first-served (FINDINGS §27). Ski-rental retention costs 1.158x more than simply keeping everything (FINDINGS §24).
2. **The cache architecture matters more than the policy.** Ollama's single-slot cache does 9.1x the recompute of a shared block cache (DECISIONS §22). With a weaker coder model, it throws away reusable state on 19.5% of handoffs (FINDINGS §21). Plain keep-all already captures this whole gap.
3. **Our own numbers needed correcting.** The Milestone 1 claim that "1 call in 3 hits Ollama's cache" is really about 1 in 6 on fresh tasks (FINDINGS §22). We also retracted three findings, and each one had been measured in a different regime than we believed.

**Where we are now.** We're on the `milestone-2` branch. The results are negative or neutral at the operating point MAARA actually runs at. The records point toward stating every result per regime (concurrency, load, context size), but nobody has written that "regime map" down yet (see §5).

---

## 2. Concepts you need first

### 2.0 The cast
- **MAARA**: our multi-agent repair pipeline. `issue_parser` (llama3.2) → retrieval (RAG) → `evaluator`, also called the Sentinel (llama3.2) → `coder` (qwen2.5-coder:7b) → pytest → `debugger` (qwen) → back to the coder, for up to 3 attempts (TOUR).
- **Ollama**: the local LLM server MAARA runs on. Underneath, it uses llama.cpp's `llama-server`.
- **vLLM**: a production LLM server. Its cache design is the alternative we model.
- **Token**: about 4 characters of text (4.077 chars per token measured on code, FINDINGS §5).

### 2.1 Prefill vs decode, and why decode dominates
A call has two phases. **Prefill** reads the whole prompt in one parallel pass. **Decode** generates output one token at a time, and each token needs its own pass through the model. So decode is much slower per token. For qwen on the Mac, prefill runs at 267.9 tok/s and decode at 28.09 tok/s (FINDINGS §4).

Worked example, the median coder call (323 prompt tokens, 47 output tokens, from trace_stats):
```
prefill  323 / 267.9 = 1.2 s
decode    47 / 28.09 = 1.7 s     <- 7x fewer tokens, still the bigger cost
```
Across the whole `original` trace, decode is **79.2%** of LLM time (198.5s prefill vs 755.8s decode) **[not in records: I recomputed it from `data/traces/full_zs_4cfg*.jsonl.gz`]**. Here's why that matters: caching only saves *prefill*. It can never touch the 79%.

### 2.2 The KV cache and its memory cost
During prefill, each layer computes a key vector and a value vector for every token. Decode reads those back for every new token. Storing them is the **KV cache**. For qwen, the cost per token is:
```
2 (K and V) x 28 layers x 4 KV heads x 128 dims x 2 bytes = 57,344 B = 56 KiB / token
```
That was derived first, then confirmed from Ollama's log at 57,356 B/token (FINDINGS §4, commit fb4aff8). Some real sizes:
- a 3,791-token saved state is 207.4 MiB (commit fb4aff8)
- a 512 MB KV budget holds about 9,360 qwen tokens in total
- a realistic 7,710-token context is about 422 MiB, and rebuilding it costs 7,710 / 267.9 = **28.8 s** of prefill (DECISIONS §13)

llama3.2 is the smaller model, yet it costs **112 KiB/token**, twice as much. It uses less aggressive grouped-query attention (GQA). This figure is derived, not measured (OPEN_ISSUES §23).

### 2.3 Prefix caching: why it only matches from the start
A token's K/V depends on *every token before it*. So a cached prompt can only be reused up to the first character where a new prompt differs:
```
cached:  [You are a coder][issue][tests][code v1]
new:     [You are a coder][issue][tests][code v2]
          <---- reused (prefix) ---->  recompute from here
new2:    [You are a debugger][issue][tests]...
          ^ differs at char 10 -> nothing reused, even though issue+tests are identical
```
That second case is real. Every role opens with its own system message, so prompts diverge at character 10, on "You are a ". 37.8% of prompt tokens repeat earlier content, but only 13.8% sit in a shared prefix. The remaining **24.0% (87.1s of prefill) is out of reach** of any prefix cache (commit 22e3337).

### 2.4 Block caching (vLLM) vs Ollama's single slot
**Block caching.** KV lives in 16-token blocks. Each block is keyed by a hash of the model plus all earlier blocks (a hash chain). Any request can hit any cached block, and partial hits work. Under memory pressure, the least recently used blocks are evicted (DECISIONS §13, ARCHITECTURE §12, commit b593cf5).

**Ollama.** There's one slot per model holding one prompt's KV, plus an 8 GiB host-RAM "prompt cache" (FINDINGS §20, DECISIONS §22):
```
new prompt arrives; f_keep = shared prefix length / slot prompt length
  f_keep >= 0.5 : reuse slot in place, drop the rest. NO saved copy.
  f_keep <  0.5 : save the slot to host RAM, search host RAM for a better start
```
A coder retry differs from the previous coder prompt only by an appended diagnosis. So it clears 0.5 and overwrites the slot in place (FINDINGS §21).

### 2.5 Ski rental
Here's the textbook version, with toy numbers. Renting skis costs $50 a day, buying costs $500, and you don't know how many days you'll ski. The strategy is to rent until you've spent $500, then buy. In the worst case you quit right after buying, so you've paid about $1,000 when hindsight says $500 was enough. That's never worse than **2x optimal**, which is the 2-competitive bound.

In our setting, keeping a cache entry is renting. Releasing it, and paying recompute if it's needed later, is buying. The rule is to **release once accrued rent equals the entry's recompute cost** (the break-even point). Rent is *derived* from what holding costs other requests, never set by a knob (DECISIONS §13):
```
rent_rate(E) = E.bytes x (victim recompute s / byte) x eviction rate   (normalised by evictable pool, DECISIONS §8.7)
```
Worked example: holder and victim are both 7,710 tokens (28.8s recompute each), and the eviction rate is 0.1/s. Then the rent rate is 2.88 s/s, so break-even is 28.8 / 2.88 = **10 s**. At 1/s, break-even is 1 s, and at 0.01/s it's 100 s. Size cancels out, so t* = 1/eviction rate (trace_stats `break_even`).

**Why the 2x bound doesn't carry over.** The proof assumes rent is constant. Our rent moves with load and with which victim is marginal, so an adversary controls both the rate and the end date. We therefore don't claim the bound. We measure the actual cost against a hindsight reference instead (DECISIONS §13, `policies.py` docstring). A learning-augmented variant (Purohit et al. 2018) works from a predicted reuse time. If reuse is predicted to be far off, it releases at λ·b. If it's predicted to be near, it holds until b/λ (`policies.py`).

### 2.6 Slack, deadlines, and the schedulers
Each task gets a **deadline** = arrival + `slack_factor` × its estimated time on an idle server (commit d07ac62). **Slack** = deadline − now − estimated remaining work on the task's critical path, meaning the longest chain of dependent steps still to run (`policies.py`).
- **FCFS**: arrival order. This is the incumbent.
- **SJF**: least remaining work first.
- **EDF**: earliest deadline first.
- **LSF**: least slack first.
- **Static priority**: a fixed rank per stage.

Toy example. Tasks A and B both have deadlines 100s away. A has 10s of work left, B has 60s. EDF sees a tie. LSF computes slack A = 90 and slack B = 40, so it runs B first. That's the whole idea of LSF: it uses the workflow graph.

### 2.7 Mechanisms need a regime to act in
- **A scheduler needs a queue.** It only decides anything when more than one request is waiting. At a correctly calibrated load, that happens on just **0.3 to 0.4%** of admissions (FINDINGS §27).
- **Retention needs memory pressure.** Rent is zero when nothing has to be evicted (DECISIONS §13 rule 3). If memory is full of *in-flight* work instead, retention has no lever. In a toy setting, only 2 of 1,557 blocked iterations (0.1%) were fixable by retention (FINDINGS §7). The knob that creates pressure is **concurrency (max batch size)**, not KV budget (OPEN_ISSUES §27).

The engine now prints an "activity line" showing whether each mechanism was ever active (commit c097f99).

---

## 3. The journey

### Ch 0. Starting point (Apr to Sep 2026)
MAARA was built Apr to May 2026 (git log), with 120 HumanEval runs (OPEN_ISSUES §14). An earlier simulator was lost. It processed one request at a time, so holding cache never blocked anyone and rent was always zero (ARCHITECTURE §1). We rebuilt it to avoid exactly that.

### Ch 1. Measure, don't assume (calibration, ~2026-09-15)
- **Question:** are the assumed constants right?
- **Did:** timed live Ollama calls (`probes/calibrate.py`).
- **Found:** qwen prefill is 267.9 tok/s, not the assumed 123 (2.2x off). Decode is 28.09, which matched. The stable prompt prefix is 480 tokens, not the assumed 2,000. Building the RAG index takes 15.6s (FINDINGS §4).
- **Why it matters:** every simulated cost comes from these rates.
- **What went wrong:** Ollama's default `num_ctx` of 2048 silently truncates prompts, which made the rates nonsense until it was set explicitly (FINDINGS §4).

### Ch 2. The simulator and its cost model
- **Question:** under contention, can retention matter at all?
- **Did:** built a continuous-batching engine: many tasks arriving at random, one shared KV pool, in-flight KV pinned, no preemption, prompt plus full output reserved at admission (DECISIONS §3).
- **Found:** "dead at this repo's scale, alive at realistic scale." At 480 tokens, rent is 0.2s and recompute is 0.0s. At 8,000 tokens, rent is 41.3s and recompute is 352.4s. The crossover is around 2,000 to 4,000 tokens (FINDINGS §7). Real repos sit at 700 to 27,000 tokens (FINDINGS §5).
- **Caveat:** these numbers predate chunked prefill and derived rent. The records don't show whether they were rerun.

### Ch 3. Tracing the agents, and the RAG incident (2026-09-29)
- **Did:** `tracing.py` logs one row per LLM call, and a sidecar file stores the prompt text.
- **What went wrong:** a traced run's RAG retrieved `sim/workload.py` as "relevant code" for a HumanEval bug, rewrote it, and truncated the simulator and `tracing.py` (commit 1ce5571).
- **Fix:** each task now runs in a sandbox holding only `target.py` and `test_target.py`, with one shared exclusion list, a write guard, and a `check_trace_clean` gate. Forensics showed the 120 historical runs were uncontaminated, because the simulator didn't exist when they ran (OPEN_ISSUES §14).

### Ch 4. First trace analysis and schedulers (commit "Phases 2 to 4", 09-29/30)
- **Found:** the median reusable prefix is 4 tokens. The reusable share went 14.4% → 13.8% → **13.7%** as errors were fixed (commits 08a99ab, 8342a6e, FINDINGS §14a). Reuse gaps are short: p50 3.2s, p95 16.8s (trace_stats). The break-even lines on the gap histogram use *assumed* eviction rates (0.01, 0.1, and 1 per second, giving 100s, 10s, and 1s). The traced run had no memory pressure, so its true break-even was infinite (trace_stats caveat).
- **Schedulers built:** FCFS, SJF, EDF, LSF, and static priority, plus an output-length estimator. The early comparison reported "SJF best p50 −18%, LSF best p95 −12%" (commit 8342a6e). **That is superseded by Ch 12.**
- **Derived rent** replaced blocking-only rent (DECISIONS §13).
- **What went wrong:** an eviction double-count caused deadlock, and the pressure snapshot taken after admission meant rent was charged on only 2 of 7,284 iterations (DECISIONS §8).

### Ch 5. Hidden reuse, and the shared-first backfire (09-30)
- **Question:** can we recover the stranded 24% by reordering prompts?
- **Did:** built `shared_first`, a layout with a common system message and shared blocks first, ordered by how many roles use each block.
- **Found:** it made things worse. Reusable share fell 13.7% → 12.0%. Coder→coder reuse fell from **7,626 to 703 tokens**, because the code block, which changes on every attempt, now came first. The evaluator also crashed on 7 of 60 runs because its `SCORE:` parse is unguarded (FINDINGS §14b).
- **Lesson:** put the most stable content first and the most volatile last.
- **What went wrong:** prefix matching ignored which model a call used. Correcting it took shared_first from 20.2% to 12.0%. Two processes also shared one run ID, which fabricated 305 task boundaries instead of 60 (FINDINGS §14a, §14c). Both are fixed, and tracing now refuses to append to an existing file.

### Ch 6. Block caching and retention machinery (commit "Phase 5/5b", 09-30)
- **Did:** added vLLM-style block-hash keying, chunked prefill (512 tokens), arrival rates calibrated to a target utilisation, and the retention policies: evict-now, ski-rental, ski-rental-pred, and an offline reference.
- **Found:** block keying turned 5 shareable entries into 607 shared block keys. Retention now controlled 574 to 1,040 recompute misses instead of 5 (commit b593cf5).
- **What went wrong:** rent wasn't conserved and was inflated about 100x. A request could deadlock behind its own blocks. The suspected hot loop wasn't the real one: profiling found `used_bytes` taking 115s of a 167s run (DECISIONS §8.7 to 8.9).

### Ch 7. Discovering Ollama's hidden cache (09-30)
- **Question:** the trace showed zero cache hits. Was that real?
- **Found:** no. A cold call took 13.4s and an identical repeat took **0.040s (335x faster)**, yet `prompt_eval_count` never changed (FINDINGS §14d). We switched to a duration-based detector, which flags a call as a hit if it runs faster than 1.5x the uncached rate. That reclassified **89 of 272 calls (32.7%) as hits**, which became Milestone 1's "1 in 3." 45 of those 89 were `issue_parser` calls.
- **Side finding:** 54% of hits had no matching previous call, so we concluded "not a single slot." Ch 8 explains this through the host-RAM tier instead.

### Ch 8. Reading the slot mechanism (09-30 to 10-04)
- **Did:** ran probes. Capacity came out around 39 × 207 MiB ≈ 8 GiB. We also ran an affinity probe and modelled the cache.
- **What went wrong, three times:**
  1. The model consumed a saved state when it restored it. Fixing that took hits from 36 to 68 of 89 (DECISIONS §17).
  2. We claimed that overlap ≥ 0.10 destroys the slot (FINDINGS §17). We **retracted** this when a rerun with a full-length B prompt showed A surviving at 25% and 50% overlap (FINDINGS §19).
  3. Freeze v1 encoded that wrong rule (DECISIONS §18).
- **What really explains the hits:** exact prompt repeats. **46 of 46** repeats on `original` and **47 of 47** on `shared_first` were hits, at 14 to 15x speed. The harness replays 15 tasks under 4 configs, so prompts recur byte for byte (FINDINGS §19).
- **Read the source:** in llama.cpp, the gate is `f_keep < 0.5`, not 0.10. Server logs agree: the cache search rate was 0.8% above 0.5 and 64.8% below it (FINDINGS §20).
- **Re-froze the rule as v2** before seeing held-out data: 84/89 and 76/78 on the fit traces (DECISIONS §19).

### Ch 9. Held-out validation: 1 in 3 becomes 1 in 6 (10-04/05)
- **Did:** built 54 new tasks from canonical HumanEval, disjoint from the original 15. Each uses the problem's own prompt with a stub body (DECISIONS §20). We ran 2 configs in separate processes and restarted Ollama between them.
- **Found:** the frozen rule's recall was **81% and 80%**, against 94% and 97% on the fit traces. All of the gap is in `issue_parser` (FINDINGS §22). Use recall, not agreement: only about 17% of calls are hits, so "always miss" would already score 83%.
- **The correction:** the hit rate fell from **34.9% → 16.9% / 17.8%**, about **1 in 6**. Exact repeats fell from 46% of hits to 2 to 7%. Genuine partial-prefix reuse actually grew (48 → 50 and 65 hits) (FINDINGS §22).

### Ch 10. The server discards state in the retry loop
- **Found:** with the llama coder, **19.5%** of same-model handoffs within a task cross `f_keep ≥ 0.5` and lose state with no saved copy. 92% of those are coder→coder retries. With the qwen coder it's only **1.4%** (FINDINGS §21). The weaker coder retries more (103 vs 79 coder calls).
- **Why it matters:** this is the handoff most likely to want its predecessor's state, and today's server throws it away.
- **Caveat:** coder model and retry rate are confounded, so cause isn't proven.

### Ch 11. The offline reference was blind (10-05)
Reuse edges were keyed per task run, so the reference couldn't see the 821 of 1,638 edges that cross runs, and it retained nothing. On `original` it paid 74.0s of recompute instead of 20.0s. Every earlier offline-ref figure is void (DECISIONS §21).

### Ch 12. Scheduler table, retracted and redone
- **First result (FINDINGS §23):** LSF cut p95 by 14.8%, and a perfect estimator made it worse. **Retracted.** The arrival rate was calibrated at batch 16 (0.0753/s), but the table ran at batch 8, about **2.2x overload**.
- **Corrected (FINDINGS §27):** the rate is 0.0343/s and p95 is about 133s. Every scheduler, *and the oracle*, sits within paired 95% confidence intervals that include zero, at both batch 8 and batch 16. The reason is that the scheduler had a choice on only 0.3 to 0.4% of decisions. Only 6% of missed tasks were doomed from the start, so the "overload cascade" theory is unsupported.

### Ch 13. Ski-rental "never fires", then the live-rent band
- **First claim (OPEN_ISSUES §26):** zero voluntary releases across 24.2M entry-ticks. **Partly retracted.** That was measured at util 0.6, and at 0.8 it fires 4 to 1,315 times. We had also swept the wrong knob (KV instead of batch size).
- **Live-rent band (OPEN_ISSUES §27):** rent is active on 50 to 59% of ticks at batch 2, 14 to 16% at batch 4, 2.6 to 3.5% at batch 8, and about 1% at batch 16. KV budget barely moves it. Batch 16 and batch 32 are identical. Even inside the band, ski-rental/keep-all ran 1.107 to 1.408, **never below 1**.

### Ch 14. The Ollama cache mode (DECISIONS §22)
I first argued that LRU was a fair stand-in for Ollama. That was wrong, because it erases the one-slot behaviour. Ollama is now an engine **mode**, not a retention policy. At holdout_qwen 1x, util 0.8, batch 4, and 512 MB:

| mode / policy | recompute | total cost |
|---|---|---|
| block / keep-all | 13.4s | 159.4s |
| ollama / keep-all | 122.1s | 239.3s |

That's **9.1x the recompute and 1.50x the total cost**. Ollama's host tier is *free* (it doesn't count against the KV budget), so this comparison is generous to Ollama.

### Ch 15. The ski-rental diagnostic (FINDINGS §24, §25)
- **Recompute is 0.8 to 1.1% of makespan.** Decode is 55%, prefill 25%, idle 18%. p95 across all deployable policies spans 185.8 to 187.6s, so retention cannot move latency. It *can* move GPU work: keep-all recomputes 13.4s against evict-now's 127.4s.
- On a shared denominator, **ski-rental is 1.158x worse than keep-all**.
- **Why:** it holds 4.5 MB less but pays 20.2s more rent. Releasing doesn't remove rent, it shifts it onto the remaining holders. The rule picks the cheapest-to-rebuild entries to release first, which leaves an expensive residue that accrues rent faster. This isn't a bug: 90% of its 862 releases were never needed again.
- **Lesson:** rent is nearly the same under every policy (DECISIONS §23, still open). We now report latency and recompute, and keep rent as a diagnostic only.

### Ch 16. Belady loses to LRU (FINDINGS §26)
Belady's rule evicts whatever will be needed furthest in the future. Even with a verified oracle it recomputes **29.4s against LRU's 13.4s**. Belady is only optimal for same-size items evicted one at a time. Here qwen and llama blocks differ in size, and eviction is bulk, freeing room for a whole prompt at once. So `offline-belady` isn't a valid lower bound.

### Ch 17. Common random numbers (DECISIONS §24, 10-06)
Hint noise followed call order, so the same block got a different prediction under each policy (0 of 12 agreed). That quietly made paired comparisons unpaired. It now depends on (seed, block), and 50 of 56 agree.

### Ch 18. The three retractions and their shared root cause
| retraction | what we claimed | why it was wrong |
|---|---|---|
| FINDINGS §17 | 0.10 overlap destroys the slot | the probe's B prompt was too short to tell the cases apart, we misread 0.10 (it's a slot-*choice* threshold, inert at `-np 1`), and the correlation came from call ordering (§19, §20) |
| FINDINGS §23 | LSF wins p95 by 14.8% | rate calibrated at batch 16, run at batch 8, so 2.2x overload (§27) |
| OPEN_ISSUES §26 | ski-rental never fires | measured at util 0.6, wrong knob swept (§27) |

The records state the lesson for the last two: "calibrate every parameter at the configuration the measurement runs at, and check the activity line before believing any comparison" (FINDINGS §27). The records don't name one root cause covering all three. **My synthesis:** in every case, the measurement ran under different conditions than the claim assumed, and nothing in the output flagged it.

### Ch 19. Toward a regime map
The phrase "regime map" doesn't appear in the records. What they do say points that way. Scheduling only becomes measurable in genuine overload, and that regime "has to be chosen and labelled" (FINDINGS §27). Retention only acts at batch 2 to 4 (OPEN_ISSUES §27). It only matters above about 2,000 to 4,000 context tokens (FINDINGS §7). The planned grid covers context {1x, 8x, 16x} × util {0.6, 0.8} (ROADMAP §2), but the only saved grid is 1x, util 0.6, batch 8 (`results/retention_grid_holdout_qwen.json`), which is outside the live band.

---

## 4. What we know now

| Finding | Status | Evidence | Confidence |
|---|---|---|---|
| qwen prefill 267.9, decode 28.09 tok/s | current | FINDINGS §4 | high, Mac only |
| qwen KV 56 KiB/token; llama 112 | current | FINDINGS §4, OPEN §23 | qwen high (measured); llama medium (derived) |
| ~79% of LLM time is decode | current | recomputed from trace | medium [not in records] |
| 13.7% of prompt tokens reusable as prefix | current (was 14.4, 13.8) | FINDINGS §14a | high on fit trace |
| 24% of tokens repeated but out of reach | current | commit 22e3337 | medium (computed before the model-aware fix) |
| Retention only matters at 2k to 4k+ contexts | current | FINDINGS §7 | medium (older engine) |
| Shared-first backfires; order by volatility | current | FINDINGS §14b | high |
| Zero Ollama cache hits | retracted | FINDINGS §14d | n/a |
| 1 in 3 calls hit Ollama's cache | superseded by ~1 in 6 | FINDINGS §19, §22 | high |
| 0.10 overlap destroys the slot | retracted | FINDINGS §17 → §19, §20 | n/a |
| Ollama gate is f_keep < 0.5 | current | FINDINGS §20 | high (source + logs) |
| Cache model recall 81% / 80% held out | current | FINDINGS §22 | high, except issue_parser |
| Retry loop discards state 19.5% (llama) / 1.4% (qwen) | current | FINDINGS §21 | medium (confounded) |
| Pre-fix offline-ref figures | void | DECISIONS §21 | n/a |
| SJF −18% p50, LSF −12% p95 (Milestone 1) | superseded | commit 8342a6e → FINDINGS §27 | n/a |
| LSF −14.8% p95; oracle worse | retracted | FINDINGS §23 → §27 | n/a |
| All schedulers tied with FCFS | current | FINDINGS §27 | high at this load |
| Ski-rental never fires | partly retracted | OPEN §26 → §27 | n/a |
| Retention only active at batch 2 to 4 | current | OPEN §27 | medium |
| Retention can't move latency (recompute ~1%) | current | FINDINGS §24 | high at 1x |
| Ski-rental 1.158x worse than keep-all | current | FINDINGS §24, §25 | medium (one point; rent model open) |
| Ollama 9.1x recompute, 1.50x cost vs block | current | DECISIONS §22 | medium (simulated, generous host tier) |
| Belady loses to LRU | current | FINDINGS §26 | high (oracle verified) |
| 2-competitive bound not claimed | current | DECISIONS §13 | high |

**Scheduling.** At MAARA's natural load, no scheduler, oracle included, differs from FCFS (FINDINGS §27). That isn't a failure of LSF. The queue almost never holds two requests, so there's nothing to reorder. It's still a result: workflow-aware scheduling only pays in overload, and any claim has to name its load and its concurrency limit.

**Retention.** Ski-rental works as designed (it releases at break-even, and 90% of releases are correct) yet loses to keep-all. Holding memory here has no *private* cost to avoid. Rent is shared pressure that just moves between holders (FINDINGS §25, DECISIONS §23). And recompute is about 1% of runtime, so even perfect retention can't help latency (FINDINGS §24). The useful conclusion: hold-or-release is the wrong shape of mechanism for this workload.

**Ollama vs block caching.** This is the positive result. Today's server pays 9.1x the recompute of a shared block cache (DECISIONS §22), and it drops the coder's state on retries (FINDINGS §21). But it's an *architecture* gap, and plain keep-all captures all of it. So never credit it to ski-rental. Retention's own contribution has to be measured as block/ski-rental vs block/keep-all (FINDINGS §24).

---

## 5. What's still open

- **The regime map.** Results need to be stated per (batch, load, context) cell. Nobody has written a plan, and the 8x and 16x retention grid hasn't been run on real traces (see Ch 19).
- **Mac vs GPU.** All rates are Mac/Metal (OPEN §9.4). The hardware table in FINDINGS §7 is simulated. An RTX 4060 feasibility check hasn't started (ROADMAP §4). Decode/prefill balance and batching efficiency may differ on CUDA, and that could change every ratio.
- **Placeholder constants** (`engine.py`, `workload.py`):
  - `batch_decode_coeff = 0.15`: the most load-bearing, and never swept (OPEN §9.3)
  - `block_tokens = 16`
  - `prefill_chunk_tokens = 512`
  - `ewma_tau_s = 30`
  - `model_switch_seconds = 0`
  - the default `slack_factor`
  - replan rejection probability
  - the estimator noise term
  - llama3.2's KV per token
- **Also open, any of which could change a conclusion:**
  - the derived-rent model is policy-invariant, and if we redefine it the retention verdict could change (DECISIONS §23)
  - the issue_parser recall gap (OPEN §24)
  - the retry-rate vs coder-model confound (FINDINGS §21)
  - no size-aware minimum-recompute reference (FINDINGS §26)
  - an unexplained slow segment in shared_first (OPEN §22)
  - large seed variance, so single-seed results aren't usable (commit eb00d17)
  - real repos have 700 to 27,000-token contexts, and MAARA's test-file lookup finds nothing there (FINDINGS §5)

---

## 6. Map to the Milestone 1 slides

Note: the repo has a tag `m1-presented` at commit 4d6634a (2026-10-02). Everything from 2026-10-04 onward came after it.

| Slide | Sections to understand | Numbers that changed |
|---|---|---|
| 1 Title | 1 | none |
| 2 Reminder (DAG, API boundary, flat queue) | 2.0, 2.6, 2.7 | none. "API boundary" and "flat queue" aren't described in the records; the closest is FCFS as the incumbent (`policies.py`) |
| 3 Milestone status | Ch 1 to 8, OPEN §11 | no status record exists for Milestone 1 beyond OPEN §11 (items 1 to 5 done) |
| 4 Workload | 2.1 to 2.5, Ch 4, 7, 9 | 79% decode: unchanged but [not in records]. 14% reusable: 13.7% now. **Ollama hit rate 1 in 3 → ~1 in 6** (FINDINGS §22). Gap histogram break-even lines use assumed eviction rates |
| 5 Retention rule | 2.5, 2.7, Ch 13, 15 | the formula is unchanged. Add: rent is policy-invariant (DECISIONS §23) |
| 6 Results so far | 2.6, 2.7, Ch 12 to 16, §4 | **scheduler table superseded: all tied with FCFS** (FINDINGS §27). "Retention not yet measured" → **now measured: ski-rental 1.158x worse than keep-all; Ollama 1.50x worse than block** |
| 7 Lessons & next steps | Ch 18, 19, §5 | add the three retractions, the calibration rule, and the regime map |

---

## 7. Glossary

- **Activity line**: engine output saying whether the scheduler, rent, and voluntary release were ever active.
- **Agreement**: the share of calls where the cache model's prediction matches reality. Inflated by true negatives.
- **Batch / max_batch**: the number of requests running concurrently.
- **Belady**: evict whatever will be reused furthest in the future.
- **Block / block key**: a 16-token KV chunk, keyed by a hash chain over the model and all earlier blocks.
- **Break-even**: the point where accrued rent equals recompute cost.
- **CI**: confidence interval.
- **Common random numbers (CRN)**: every policy sees identical randomness, so differences can be compared paired.
- **Competitive ratio**: worst-case online cost divided by the hindsight-optimal cost.
- **Continuous batching**: requests join and leave the running batch at every step.
- **Critical path**: the longest chain of remaining dependent work in a task.
- **DAG**: directed acyclic graph, here the workflow's call graph.
- **Decode**: generating output one token at a time.
- **EDF**: earliest deadline first.
- **Evict-now**: never retain anything.
- **EWMA**: exponentially weighted moving average.
- **f_keep**: shared prefix ÷ slot prompt length. Ollama reuses in place at ≥ 0.5.
- **f_sim**: shared prefix ÷ incoming prompt length. Used to choose between slots.
- **FCFS**: first come, first served.
- **Fit set / held-out**: traces a model was tuned on, vs traces it never saw.
- **Freeze**: committing a rule before testing it on held-out data.
- **GQA**: grouped-query attention. Fewer KV heads, so less KV per token.
- **Hindsight denominator**: per block, min(rebuild cost, rent while held), measured under keep-all.
- **Host tier / prompt cache**: Ollama's 8 GiB RAM store of saved slot states.
- **HumanEval**: a benchmark of Python programming problems.
- **Keep-all**: never release voluntarily; LRU when forced to evict.
- **KV cache**: stored keys and values for processed tokens.
- **LCP**: longest common prefix.
- **llama.cpp / llama-server**: the inference engine under Ollama.
- **LRU**: least recently used eviction.
- **LSF**: least slack first.
- **MAARA**: our multi-agent repair pipeline.
- **Makespan**: total simulated time to finish the workload.
- **MAPE**: mean absolute percentage error of the estimator.
- **Metal**: Apple's GPU API.
- **-np 1**: llama-server running with one slot.
- **num_ctx**: Ollama's context window setting. The default of 2048 truncates silently.
- **Offline reference / offline-ref**: a hindsight policy that reads the true future.
- **Oracle**: given true future values.
- **p50 / p95**: median and 95th-percentile latency.
- **Paired comparison**: same seeds for every policy, compared by per-seed differences.
- **PLACEHOLDER / MEASURED**: provenance labels on every constant.
- **Pred-belady**: Belady using noisy predicted reuse times.
- **Prefill**: processing the prompt in one parallel pass.
- **Prefix caching**: reusing KV for a prompt's shared beginning.
- **prompt_eval_count**: Ollama's prompt token count. Doesn't reveal cache hits.
- **RAG**: retrieval-augmented generation (here, code retrieval).
- **Recall**: the share of real cache hits that the model predicts.
- **Recompute**: re-running prefill for evicted KV.
- **Rent**: derived cost of holding KV that others need.
- **Rent-active ticks**: the share of engine steps where rent was accruing.
- **Retention**: deciding what cached KV to keep.
- **Sandbox**: a per-task directory, the only thing the agent can see.
- **Sentinel / evaluator**: the llama3.2 plan-review gate.
- **shared_first / original**: the two prompt layouts.
- **SJF**: shortest job first.
- **Ski rental**: rent until rent paid equals the buy price, then buy.
- **Ski-rental-pred**: ski rental using a predicted reuse time and trust parameter λ.
- **Slack**: deadline − now − remaining critical-path work.
- **slack_factor**: deadline multiplier on a task's idle-server time.
- **Slot**: llama-server's per-model active KV context.
- **Static priority**: a fixed rank per stage.
- **Token**: a unit of text, about 4 characters.
- **Trace / sidecar**: per-call log rows, and the separate file of prompt text.
- **Utilisation**: the target GPU busy fraction used to set arrival rate.
- **vLLM**: a production LLM server with a block prefix cache.
- **Voluntary release**: ski-rental dropping an entry at break-even, without being forced.

---

## 8. Check yourself

1. Why can't a prefix cache reach the 24% of repeated tokens?
2. Why did shared_first reduce reuse?
3. What caused "1 in 3" to become "1 in 6"?
4. Why doesn't the 2-competitive bound apply to our ski rental?
5. Why are all schedulers tied with FCFS, and what would separate them?
6. Why does ski-rental pay more rent while holding less memory?
7. Why can't the 1.50x Ollama gap be credited to ski-rental?
8. What knob makes retention active, and why isn't it KV budget?

**Answers.**
1. KV depends on all earlier tokens. The roles' prompts diverge at character 10, so identical later text can't be matched.
2. It put the code block first, and that changes every attempt. Coder→coder reuse fell from 7,626 to 703 tokens.
3. The harness replayed 15 tasks under 4 configs, so prompts recurred exactly. On disjoint tasks, the rate is 16.9 to 17.8%.
4. Rent isn't constant. It moves with load and with which victim is marginal, so the adversary controls the rate too.
5. The queue almost never holds two requests (0.3 to 0.4% of decisions). Deliberately chosen overload would separate them.
6. Releasing shifts shared rent onto the remaining holders, and it releases the cheap-to-rebuild entries first, leaving an expensive residue.
7. Plain keep-all on a block cache captures the whole gap. It comes from architecture, not policy.
8. max_batch (concurrency). Tightening KV fills memory with pinned in-flight work, so nothing is left to evict.
