# Milestone 3 report: source notes

Saved 2026-10-24. The report is not due yet. This file collects what the
report needs, so it can be written later without re-deriving anything. Every
number points to the FINDINGS or DECISIONS section it comes from; quote from
those sections, not from here, if they have since changed.

## 1. The claim, in one paragraph

For agentic LLM workflows, **program-level ordering** (`program-fcfs`, oldest
workflow first) is what cuts tail latency. Against request-level FCFS, which
is what vLLM and Ollama do by default, it cuts p95 task latency by 17–36% at
batch 2–4 at load 1.0, and by up to 40% in overload. It is never worse, on a
sealed test set. No other ordering rule tested beats it:
- SJF, EDF and the hint-priority rule.
- LSF, with online and with oracle output lengths.
- Autellix's PLAS.
- Deadlines sized to each task's own critical path.

The two textbook mechanisms the project set out to test are negative results:
- **Least-slack-first scheduling** never beats it.
- **Ski-rental KV retention** never beats keep-all.

## 2. Results to report, with sources

| Result | Numbers | Source |
|---|---|---|
| `program-fcfs` vs FCFS, sealed | 72 better / 0 worse / 8 tied of 80. At load 1.0, by batch 2/4/8/16: qwen −28/−17/−8/−1% (batch 16 n.s.), llama −36/−28/−17/−4%. Largest −38% (qwen), −40% (llama) | FINDINGS §33 |
| LSF / LSF-oracle / EDF / hint vs `program-fcfs`, sealed | 0/61/19, 0/63/17, 0/59/21, 0/67/13 | §33 |
| SJF vs `program-fcfs`, sealed | 0/68/12 (+5% to +34% worse at batch 2–4) | §33 addendum |
| PLAS (Autellix) vs `program-fcfs` | 0 better in 80 cells (S1 Mac, S2 GPU); +91/+67/+23/+1% at load 1.0 (Mac). Also worse than FCFS (P2 failed) | §40, DECISIONS §36 |
| Task-sized deadlines | LSF and EDF still 0 better of 40; fewer misses at batch 2 only, for a +5 to +22% worse tail | §40 |
| Retention at 1x, sealed | Recompute under 1% of makespan; ski-rental ties keep-all on p95 and recomputes 9.9–16.6 s more | §24, §33–35 |
| Retention at 8x, holdout | evict-now +222/+235/+241 s p95 (+10–11%); ski-rental +171*/+96*/−2 s and 118–164 s more recompute | §35 |
| Concurrency cap, sealed (S3) | batch 4–16: 38/15/7 of 60. Batch 16: 10/0/0 on both traces (−10 to −16% qwen, −20 to −33% llama). Batch 4 overload +3 to +6% worse | §34 |
| Cap on GPU costs | Never a net gain below batch 16 | §37 |
| Self-lifting cap (qcap) | Dominates the cap (0 worse / 25 better of 160) but failed its "safe" test Q1 (3/5/6 worse cells). Development result only | §39, DECISIONS §35 |
| Batching-cost sensitivity | H1–H4 hold at c = 0.03, 0.15, 0.3, 0.5, 0.841, on Mac and G1 GPU rates | §36–38 |
| Public-data GPU costs | c ≈ 0.3 for Ollama/llama.cpp (llama-batched-bench, RTX 3060/3090), ≈ 0.03 for vLLM (Qwen2.5-7B AWQ, L4) | DECISIONS §34 |
| Task quality, sealed | pass@1 75.0% vs 48.9% on 92 unseen problems; 78.1% vs 54.1% over all 146, paired gap +24.0 points [+15.0, +33.0]. A coder-model comparison | §28 |
| Workload | Reusable prefix median 4 tokens (13.7% of prompt tokens); retention starts to matter at 2,000–4,000-token contexts | README, FINDINGS early sections |

Hypothesis scorecard (all pre-registered and committed before the runs):
- **H1–H4** (DECISIONS §33/§34): held in every pass.
- **Q1** failed; **Q2** held; **Q3** held in 2 of 3 runs (DECISIONS §35).
- **P1** held; **P2** failed; **T1** and **T2** held (DECISIONS §36).

## 3. Method points the report must state

- **Simulator.** Trace-driven continuous-batching engine with a shared KV budget, replaying measured MAARA traces (`simulator/engine.py`, `docs/ARCHITECTURE.md`).
- **Provenance.** Every cost constant is tagged MEASURED / TRANSCRIBED / PUBLIC ESTIMATE / PLACEHOLDER (`CostModel.provenance()`).
- **Oracle discipline.** Online policies can't read future fields (`OracleLeakError`, `simulator/policies.py`).
- **Statistics.** Paired 95% CIs over 20 seeds with common random numbers (DECISIONS §24). Every run writes a manifest with git hash and dirty flag (§25).
- **Pre-registration.** DECISIONS §26/§31/§33–36: commit hypotheses, then run, then score as written. Failed hypotheses are reported as failed.
- **Sealed set.** `final_qwen`/`final_llama` were run once, after the policy freeze `f4c889b` (DECISIONS §26, §31). HumanEval is now used up.
- **Load axis.** Two axes; say which for every number. The sealed headline maps (§33, figure 1) are on the **busy** axis: load = arrival rate / the rate giving 80% GPU busy, so "load 1.0" there is 80% busy. The sweeps, qcap and PLAS runs (§36–40) are on the **capacity** axis: load = arrival rate / saturation throughput (OPEN_ISSUES §30).

## 4. Threats to validity (write this section)

1. **Seeds vary arrival times, not tasks.** The 20 seeds re-time a fixed set of traces; task content is the same.
2. **Arrivals are synthetic.** The real runs were sequential. Contention comes from re-timing arrivals as a Poisson process (`simulator/trace_loader.py:load_trace`).
3. **One workload family.** MAARA on HumanEval. Task sizes vary little (critical-path CV 0.61, §40), which is why PLAS loses. A heavy-tailed workload may rank the rules differently.
4. **Hidden scheduler.** The engine serves one model per iteration (`_choose_active_model`), and no policy controls it (OPEN_ISSUES §28).
5. **Batching law.** Decode time is 1 + (n−1)c. The GPU results are a sensitivity analysis inside that law, not GPU measurements. G2 was never measured, because the card failed with Xid 79 (DECISIONS §34).
6. **Correlated cells.** Cells share traces and seeds, and there is no multiple-comparison correction. Cell counts show consistency, not independent confirmations.
7. **CPU stages.** RAG and pytest timings are Mac-measured, even in GPU runs.
8. **Remaining open issues** (OPEN_ISSUES index): 9.5 model-switch cost assumed 0; 14 MAARA write-up caveats; 24 `issue_parser` cache recall gap; 27 no point where ski-rental wins (16x untested).

## 5. Corrections owed to the Milestone 2 brief

These are left unedited in the submitted brief; correct them in the Milestone 3 report.
- **"Up to 40%"** is the best overload cell. Lead with 17–36% at operating load, batch 2–4.
- **"The cap is the next lever / make it load-aware"** was tried (§39). It isn't safe, and it never helps on GPU costs.
- **GPU wording.** It's simulated, not measured, apart from the G1 rates.
- **pass@1.** Quote 75.0% vs 48.9% as the 92-problem figure, and +24.0 points as the 146-problem paired gap (§28). The brief mixed the two.

## 6. Related work to cover (verify each citation before use)

- **Autellix** (Luo et al., 2025): program-level scheduling, PLAS/ATLAS. Closest work; §40 compares against PLAS.
- **vLLM / PagedAttention** (Kwon et al., SOSP 2023): request-level FCFS baseline and KV paging.
- **Orca** (Yu et al., OSDI 2022): iteration-level (continuous) batching.
- **SGLang / RadixAttention** (Zheng et al., 2024): prefix caching.
- **Parrot** (Lin et al., OSDI 2024): application-level information for LLM serving.
- **NVIDIA Dynamo agent hints**: static client priorities (the `hint-priority` baseline).
- **Learning-augmented ski-rental** (Purohit, Svitkina, Kumar, NeurIPS 2018): the retention policy's basis.
- **Size-based vs age-based scheduling theory** (LAS / SRPT / FCFS under low- vs high-variance sizes): explains §40.

## 7. Figures the report needs (input to the visualization refactor)

Built 2026-10-24: `make figures-m3` writes figures 1–6 below to `results/figures/m3/` (`experiments/plot_m3.py`; sources in that folder's README). Older figures live in `results/figures/regime/` (`experiments/plot_regime.py`, busy-axis maps from §29–31) and `results/figures/original/`. None yet cover the capacity-axis, sealed, sweep, qcap or PLAS results. Needed:
1. **Headline:** p95 change vs FCFS against load, one line per batch, sealed qwen and llama (§33).
2. **Baseline league table:** cells better/worse/tied vs `program-fcfs` for every rule, including SJF and PLAS (§33, §40).
3. **Sensitivity:** headline effect vs batching coefficient, Mac and GPU rates (§36–38).
4. **Retention:** recompute and p95 at 1x vs 8x (§24, §35).
5. **Cap / qcap heatmap:** batch × load, Mac vs GPU (§34, §37, §39).
6. **Why PLAS loses:** task-size distribution (CV 0.61) next to the p95 gap (§40).
