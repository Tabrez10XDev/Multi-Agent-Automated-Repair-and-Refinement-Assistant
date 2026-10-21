# Milestone 2 presentation: brief for the slide-building agent

As of 2026-10-21. Project: "Adaptive LLM Inference Scheduling" (Master's capstone, RIT). Repo branch `milestone-2`.

## 1. How to use this brief

- **Talk:** 10 minutes, for the mentor and committee: a technical audience that knows the project's goal. Aim for about 11 slides at about 55 seconds each, plus backup slides.
- **Numbers:** every number here is final and has a source. Use them as given. Don't add numbers, recompute them, or round them differently. Values with a `*` are statistically significant (paired 95% CI excludes zero).
- **GPU:** don't show any GPU performance numbers. None exist yet (see section 7).
- **Every result slide** carries the footer: *"Simulated; costs measured on Apple M3 Pro / Ollama / Metal; 20 paired seeds unless noted."*

## 2. The story in one line

For agentic LLM workflows, **finishing older workflows first** (`program-fcfs`) is what cuts tail latency, by up to 40%. The textbook mechanisms, least-slack-first (LSF) scheduling and ski-rental KV-cache retention, never beat simple baselines. It all holds on a **sealed test set** that was run once, after the policies were frozen.

## 3. Setup (slides 2–4)

| Item | Fact |
|---|---|
| Question | In multi-agent LLM pipelines, does smarter scheduling (LSF over the workflow graph) or smarter KV-cache retention (ski-rental) reduce latency? |
| Workload | **MAARA**: a multi-agent code-repair pipeline on HumanEval. 4 LLM roles (issue parser, evaluator, coder, debugger) on `qwen2.5-coder:7b` and `llama3.2`, plus RAG retrieval and a pytest stage |
| Method | Record real agent runs against local Ollama, then replay them in a **trace-driven continuous-batching simulator** with a shared KV-cache budget, under controlled load |
| Measured costs (Mac) | qwen prefill 267.9 tok/s, decode 28.09 tok/s. Batching coefficient 0.841: 8 concurrent streams decode only about 15% faster in total than one |
| Fitting set | `holdout_qwen`: 54 tasks, where all policy development happened |
| Sealed test set | `final`: the last 92 HumanEval problems, collected once and locked in code until the policies were frozen. 91 tasks for qwen; one runaway call was excluded before any result existed |
| Statistics | 20 paired seeds (common random numbers), paired 95% CIs. 40 cells per map: 4 batch sizes × 10 load levels |
| Integrity | Freeze commit `f4c889b` came before the sealed run, with 4 pre-registered comparisons. The sealed run's outcome was reported without changing any policy |

**Load** = arrival rate ÷ the rate that keeps the GPU 80% busy at that batch size. 1.00 is the operating point; above 1.0 is overload.

## 4. Results

### R-A. Program-level ordering wins (headline; slide 5)

Sealed set, both traces, 80 cells. `program-fcfs` beats request-level FCFS on p95 in **72 of 80 cells**, and is **never worse**. The largest cut is **−38%** (qwen) and **−40%** (llama).

At the operating load (1.00):

| Batch | final_qwen | final_llama |
|---|---|---|
| 2 | −28%* | −36%* |
| 4 | −17%* | −28%* |
| 8 | −8%* | −17%* |
| 16 | −1% (n.s.) | −4%* |

The gain is largest when batches are small and requests queue. At batch 16 it appears only in overload. Full tables are in section 9. Sources: `results/sealed/scheduling_map_final_qwen_full.json`, `results/sealed/scheduling_map_final_llama.json`.

### R-B. LSF never beats program-level ordering (slide 6)

| Policy vs `program-fcfs` (sealed, 80 cells) | Better | Worse | Tie |
|---|---|---|---|
| LSF (online length estimate) | **0** | **61** | 19 |
| LSF with oracle (true) lengths | 0 | 63 | 17 |
| EDF | 0 | 59 | 21 |
| Client priority (Dynamo-style) | 0 | 67 | 13 |

- **The gain is task order, not slack.** On the fitting set, EDF matches LSF in 39 of 40 cells, and even perfect length knowledge doesn't help LSF.
- **Against the usual baseline, LSF looks good.** It beats request-level FCFS, and that is where "LSF helps" claims come from. Against `program-fcfs` it never wins.

### R-C. Concurrency cap: a new lead (slide 7)

`program-fcfs-cap` is oldest-workflow-first plus a cap on concurrent requests, set from the hardware's batching law. On this nearly serial batcher the cap is 2 at batch 4 and 3 at batches 8–16. Its parameter was fixed before any run, and it was **first evaluated on the sealed set** as pre-registered comparison S3.

- **Batches 4–16, sealed, both traces (60 cells):** better than `program-fcfs` in **38**, worse in **15**, tie in **7**.
- **Batch 16:** −10% to −16% (qwen) and **−20% to −33%** (llama) at loads up to 1.5.
- **Batch 8:** −8% to −17% at low and moderate load.
- **Batch 4 in overload:** +2% to +6% **worse**. A fixed cap is too tight when the queue is deep, so the cap should become load-aware next.
- **Batch 2:** identical by construction, since the cap equals the batch size.
- **Fitting set:** the same pattern (`results/scheduling_map_slo_cap.json`), so the result is not a fluke of the sealed data.

### R-D. KV-cache retention (slide 8)

**At MAARA's real context size (1x), retention can't move latency.** Recompute is under 1% of makespan (FINDINGS §24). Sealed set, `final_qwen`, batch 4, load about 0.8, 20 seeds, with three cache budgets:

| vs keep-all | Δ p95 | Δ GPU recompute |
|---|---|---|
| ski-rental | +1.1, −2.8, +3.5 s (all ties) | **+15.9*, +16.6*, +9.9* s** |
| ski-rental with predictions | ties | +17.1*, +14.7*, +9.3* s |
| evict-now | +31.0*, +33.2*, +44.8* s | +114 to +139 s* |

**At 8x context, retention matters, but ski-rental still loses** (fitting set `holdout_qwen`, 10 seeds, batch 4, 80% busy). Not sealed.

| Cache budget | evict-now vs keep-all, p95 | ski-rental vs keep-all, p95 | ski-rental vs keep-all, recompute |
|---|---|---|---|
| Tight (2,545 MB) | +222 s (+10%)* | **+171 s (+8%)*** | +164 s* |
| Medium (2,864 MB) | +235 s* | **+96 s (+4%)*** | +118 s* |
| Ample (3,501 MB) | +241 s* | −2 s (tie) | +136 s* |

**Takeaway:** keep-all (equivalently LRU) is best everywhere tested. Ski-rental's release rule throws away state that is needed again. Sources: `results/sealed/retention_headroom_final_qwen.json`, `results/retention_headroom_8x.json`.

### R-E. Real-model measurements (slide 9)

- **Ollama prompt-cache model:** the rule is read from the llama.cpp source and was never tuned. Hit recall: 81% / 80% on the two holdout traces, 77% / 97% on the sealed traces, **86% pooled** over all unseen data. About **1 call in 8** hits Ollama's cache.
- **Task success (pass@1), sealed set:** 75.0% with the qwen coder against 48.9% with llama. Over 146 unseen problems the paired gap is **+24.0 points [+15.0, +33.0]**.
- **Pipeline defect found and fixed:** one `issue_parser` call generated **40,960 tokens over 26.5 minutes**, because no agent set an output limit. It is now capped at 4,608 tokens for future collections (DECISIONS §32).

## 5. Integrity (backup slide, or 20 seconds on slide 4)

| Earlier claim | Status |
|---|---|
| "LSF cuts p95 by 14.8%" | **Retracted:** measured at about 2.2× the intended load |
| Ollama cache-hit rate of 32.7% | **Superseded:** inflated by repeated prompts. About 1 in 8 on unseen problems |
| "Offline Belady is the minimum-recompute reference" | **Withdrawn:** it recomputes more than LRU here |

The sealed-set protocol was: freeze, then one run. A single exclusion (the runaway call) was recorded before any result existed.

## 6. Next steps: Milestone 3 (slide 10)

1. **GPU calibration** on the RTX 5060 Ti, then rerun the key cells under the measured GPU profile.
2. **The concurrency cap as the lead mechanism:** make it load-aware so it stops hurting at batch 4 in overload.
3. **Deadlines sized to each task's own length** (OPEN_ISSUES §29): LSF's fairest possible test.
4. **Related work:** Orca, vLLM/PagedAttention, Sarathi-Serve, Autellix (closest to the program-level result), and ski-rental with predictions.

## 7. GPU status (one line on slide 10; keep it short)

- **Say:** "GPU environment is up: RTX 5060 Ti 16 GB, CUDA 13.4, Ollama with the same model builds as our Mac runs. Both models load fully on the GPU (qwen about 4.8 GB). The full codebase is deployed there, with 381 tests passing, and the simulator already supports a GPU cost profile. Calibration runs are ongoing."
- **Don't show** throughput, latency, speedup, or any GPU-vs-Mac comparison. None has been measured.
- **Optional speaker note, if asked:** the card hit a hardware fault (Xid 79, "fell off the bus") on its first load. IT is resolving it.

## 8. Slide plan (about 10 minutes)

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | Title | Project, team, Milestone 2 | — |
| 2 | Problem | Agent pipelines create many dependent LLM calls; tail latency matters. Two mechanisms on trial: LSF scheduling, ski-rental KV retention | Simple workflow-DAG sketch |
| 3 | Setup | MAARA → recorded traces → simulator with measured costs | Pipeline diagram (section 3 table) |
| 4 | Method | Fitting set vs sealed set, freeze, 20 paired seeds, pre-registered comparisons | Timeline: develop → freeze → one sealed run |
| 5 | **Headline** | `program-fcfs` cuts p95 by up to 40%, in 72/80 sealed cells, never worse | Line chart: p95 change vs load, one line per batch size (section 9, first table) |
| 6 | LSF | 0 better / 61 worse of 80 against `program-fcfs`; the gain is task order | Bar chart of the R-B table |
| 7 | Concurrency cap | Up to −33% at batch 16; worse only at batch 4 in overload | Heatmap, batch × load (section 9 cap tables) |
| 8 | Retention | 1x: nothing to win. 8x: it matters, but ski-rental still loses | Two small tables or paired bars (R-D) |
| 9 | Real-model results | Cache model 86% recall; pass@1 75.0% vs 48.9%; runaway call found | Three stat callouts |
| 10 | GPU + next | One GPU status line (section 7), then the 4 next steps | — |
| 11 | Takeaways | (1) Order by workflow, not by slack. (2) A batching-aware cap is the next lever. (3) Keep the KV cache; ski-rental doesn't pay | — |
| B1 | Backup: integrity | Section 5 table | — |
| B2 | Backup: full cell tables | Section 9 | — |

## 9. Chart data (sealed unless noted)

Values are % change in p95 task latency. Negative is better. `*` means the paired 95% CI excludes zero. Headers are the load points at batches 4–16; batch 2's load points differ slightly and are listed under each table. `=` means identical by construction.

**program-fcfs vs FCFS, p95 change, final_qwen (sealed)**

| batch \ load | 0.78 | 0.85 | 1.00 | 1.10 | 1.15 | 1.25 | 1.50 | 1.75 | 2.20 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -14%* | -23%* | -28%* | -31%* | -33%* | -37%* | -38%* | -36%* | -31%* | -25%* |
| 4 | -10%* | -13%* | -17%* | -23%* | -25%* | -30%* | -34%* | -33%* | -28%* | -22%* |
| 8 | -2% | -3% | -8%* | -11%* | -12%* | -17%* | -27%* | -27%* | -24%* | -19%* |
| 16 | -0% | -0% | -1% | -1% | -3%* | -5%* | -12%* | -17%* | -16%* | -12%* |

Batch 2 loads: 0.71, 0.85, 1.00, 1.10, 1.15, 1.25, 1.50, 1.75, 2.20, 3.00.

**program-fcfs-cap vs program-fcfs, p95 change, final_qwen (sealed)**

| batch \ load | 0.78 | 0.85 | 1.00 | 1.10 | 1.15 | 1.25 | 1.50 | 1.75 | 2.20 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | = | = | = | = | = | = | = | = | = | = |
| 4 | -6%* | -5%* | -3% | +2% | +4%* | +4%* | +5%* | +6%* | +4%* | +3%* |
| 8 | -8%* | -9%* | -9%* | -8%* | -6%* | -4%* | +0% | +2%* | +2%* | +2%* |
| 16 | -10%* | -12%* | -15%* | -16%* | -16%* | -14%* | -13%* | -6%* | -3%* | -2%* |

Batch 2 loads: 0.71, 0.85, 1.00, 1.10, 1.15, 1.25, 1.50, 1.75, 2.20, 3.00.

**program-fcfs vs FCFS, p95 change, final_llama (sealed)**

| batch \ load | 0.74 | 0.83 | 1.00 | 1.10 | 1.17 | 1.25 | 1.50 | 1.75 | 2.20 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -27%* | -30%* | -36%* | -37%* | -40%* | -39%* | -37%* | -34%* | -29%* | -23%* |
| 4 | -16%* | -20%* | -28%* | -32%* | -33%* | -35%* | -36%* | -33%* | -28%* | -22%* |
| 8 | -7%* | -8%* | -17%* | -22%* | -26%* | -27%* | -30%* | -28%* | -24%* | -19%* |
| 16 | -1% | -0% | -4%* | -4%* | -8%* | -9%* | -12%* | -13%* | -11%* | -10%* |

Batch 2 loads: 0.74, 0.83, 1.00, 1.10, 1.25, 1.26, 1.50, 1.75, 2.20, 3.00.

**program-fcfs-cap vs program-fcfs, p95 change, final_llama (sealed)**

| batch \ load | 0.74 | 0.83 | 1.00 | 1.10 | 1.17 | 1.25 | 1.50 | 1.75 | 2.20 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | = | = | = | = | = | = | = | = | = | = |
| 4 | -5%* | -6%* | -1% | +2% | +3% | +3%* | +5%* | +5%* | +4%* | +4%* |
| 8 | -14%* | -17%* | -15%* | -12%* | -9%* | -8%* | -3%* | -1%* | -0% | +2%* |
| 16 | -20%* | -23%* | -30%* | -33%* | -32%* | -31%* | -26%* | -20%* | -14%* | -7%* |

Batch 2 loads: 0.74, 0.83, 1.00, 1.10, 1.25, 1.26, 1.50, 1.75, 2.20, 3.00.

**program-fcfs-cap vs program-fcfs, p95 change, holdout_qwen (in-sample)**

| batch \ load | 0.78 | 0.85 | 1.00 | 1.10 | 1.22 | 1.25 | 1.50 | 1.75 | 2.20 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | = | = | = | = | = | = | = | = | = | = |
| 4 | -1% | -0% | +4% | +3% | +6%* | +6%* | +10%* | +12%* | +10%* | +9%* |
| 8 | -8%* | -9%* | -8%* | -7%* | -3% | -4% | +1% | +2% | +6%* | +8%* |
| 16 | -8%* | -8%* | -11%* | -10%* | -11%* | -11%* | -9%* | -9%* | -4%* | +3%* |

Batch 2 loads: 0.78, 0.85, 1.00, 1.10, 1.22, 1.25, 1.50, 1.75, 2.20, 3.00.

## 10. Don'ts

- No GPU performance numbers, and no implication that GPU results exist.
- Don't claim LSF or ski-rental "work"; both are analysed negative results against the right baseline.
- Don't claim the 2-competitive ski-rental bound. The cost model here breaks its assumptions (DECISIONS §13).
- Don't call the 8x retention result sealed: it's the fitting set, 10 seeds.
- Don't drop the footer saying results are simulated with Mac-measured costs.
- p95 is per-task end-to-end latency across the whole workflow, not per request.
