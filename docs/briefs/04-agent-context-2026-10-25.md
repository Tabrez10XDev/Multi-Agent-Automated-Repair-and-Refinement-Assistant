# Agent context: what the project found since Milestone 1, and where it stands

Snapshot at 2026-10-25 20:30 EDT. Branch `milestone-2`, HEAD `5deff32`. The
owner will give you a task after this. Read this file first; then read the
source section named next to any number before you quote or change it.

Citation style: "FINDINGS §33" = `docs/FINDINGS.md` section 33. DECISIONS and
OPEN_ISSUES work the same way. Section numbers are non-sequential on purpose;
never renumber them.

---

## 1. The project in five lines

- **Title:** "Adaptive LLM Inference Scheduling", a Master's capstone (RIT).
- **Question:** in multi-agent LLM pipelines, does workflow-aware scheduling
  (least-slack-first over the task DAG) or workflow-aware KV-cache retention
  (ski-rental) cut latency?
- **Workload:** MAARA, a multi-agent code-repair pipeline on HumanEval: issue
  parser and evaluator on `llama3.2`, coder and debugger on
  `qwen2.5-coder:7b`, plus RAG and pytest stages. It was run against local
  Ollama on an Apple M3 Pro, and every LLM call was traced.
- **Method:** replay the traces in a trace-driven continuous-batching
  simulator with a shared KV budget (`simulator/engine.py`), under controlled
  Poisson arrivals. All results are simulated; costs are measured on the Mac
  or taken from public GPU data. Nothing has been measured on a real GPU.
- **Statistics:** 20 paired seeds with common random numbers, paired 95% CIs.
  A policy is "better" when `hi_s < 0` and "worse" when `lo_s > 0`. There is
  no multiple-comparison correction.

## 2. Milestone 1 (presented 2026-10-02, git tag `m1-presented`)

What M1 claimed, and what happened to each claim afterwards:

| M1 claim | Now |
|---|---|
| "1 call in 3 hits Ollama's cache" | **Corrected to about 1 in 6** on held-out problems (16.9%, 17.8%), about 1 in 8 over all unseen traces. The 1-in-3 came from exact prompt repeats in the harness (FINDINGS §19, §22, §28). |
| LSF cuts p95 by 14.8% | **Retracted.** It was measured at about 2.2x overload. At the correct load, every scheduler ties FCFS (FINDINGS §23 → §27). |
| Ski-rental "never fires" | **Retracted.** It was measured at util 0.6 with the wrong knob swept (OPEN_ISSUES §26 → §27). |
| The 0.10 overlap threshold destroys Ollama's slot | **Retracted.** The real gate is `f_keep < 0.5`, read out of llama.cpp's source (FINDINGS §17 → §19–20). |
| Retention's crossover is at 2,000–4,000 tokens of context | Still current, but as a GPU-work result, not a latency one (FINDINGS §7, §24). |

`docs/STORY_SO_FAR.md` tells the story up to 2026-10-06 chapter by chapter,
including the three retractions.

## 3. Milestone 2 results (MAARA; all pre-registered, run on a sealed set)

**The sealed test set.** `final_qwen` and `final_llama` are the last 92
HumanEval problems. They were collected once and locked in code until the
policy freeze `f4c889b` (DECISIONS §26, §31). They were run once. HumanEval
has no problems left, so there is no fresh MAARA test set.

**The headline: program-level ordering.** `program-fcfs` serves the oldest
workflow first. It beats request-level FCFS (the vLLM and Ollama default) on
p95 task latency in **72 of 80** sealed cells, and is **never worse**
(FINDINGS §33). At load 1.0, batch 2/4/8/16: qwen −28/−17/−8/−1% (batch 16
n.s.), llama −36/−28/−17/−4%. The largest cut is −38% (qwen) and −40%
(llama), in overload. Lead with 17–36% at batch 2–4, not "up to 40%".

**Nothing beats `program-fcfs`.** Counts are better / worse / tied, sealed,
80 cells.

| Rule vs `program-fcfs` | Result | Source |
|---|---|---|
| LSF (online estimate) | 0 / 61 / 19 | FINDINGS §33 |
| LSF-oracle (true lengths) | 0 / 63 / 17 | §33 |
| EDF | 0 / 59 / 21 | §33 |
| Dynamo-style client priority | 0 / 67 / 13 | §33 |
| SJF | 0 / 68 / 12 | §33 addendum |
| Autellix PLAS | 0 better in 80 cells, worse in 59. It is also worse than FCFS, so P2 failed. | §40, DECISIONS §36 |
| LSF and EDF with task-sized deadlines | Still 0 better of 40 | §40 |

Why PLAS loses: MAARA's task sizes vary little (critical-path CV 0.61), and
age-based ordering beats size-based ordering at low size variance (§40).

**The two textbook mechanisms are negative results.**
- **LSF scheduling** never beats `program-fcfs`. Its gains against FCFS come
  from task order, not slack: EDF matches LSF in 39 of 40 cells.
- **Ski-rental retention** never beats keep-all. At 1x, recompute is under 1%
  of makespan, so retention cannot move latency (FINDINGS §24). At 8x context
  (holdout), retention matters (evict-now is +10–11% on p95), but ski-rental
  still does not beat keep-all (§35). Both numbers were corrected after a
  bug fix; see §5.

**Other M2 results.**
- **Concurrency cap** (`program-fcfs-cap`, sealed S3): at batch 16 it is 10
  better / 0 worse / 0 tied on both traces (−10 to −16% qwen, −20 to −33%
  llama). At batch 4 in overload it is 3–6% worse (§34). On GPU costs it is
  never a net gain below batch 16 (§37).
- **Self-lifting cap** (`qcap`): it dominates the cap (0 worse / 25 better of
  160) but failed its "safe" test Q1. It is a development result only (§39,
  DECISIONS §35).
- **Batching-cost sensitivity:** the conclusions (H1–H4) hold at c = 0.03,
  0.15, 0.3, 0.5 and 0.841, on Mac and G1 GPU rates (§36–38).
- **Simulated GPU:** profile `gpu_5060ti_public`, built from public data:
  c ≈ 0.3 for Ollama/llama.cpp, ≈ 0.03 for vLLM (DECISIONS §34). The real
  RTX 5060 Ti 16 GB failed with Xid 79; only its G1 rates were measured. The
  GPU is closed (owner, 2026-10-23), and 16 GB is the ceiling for any plan.
- **Task quality, sealed:** pass@1 is 75.0% (qwen coder) vs 48.9% (llama
  coder) on 92 problems. The paired gap over 146 problems is +24.0 points
  [+15.0, +33.0] (FINDINGS §28).
- **Ollama cache model:** the rule was frozen as v2 and validated on
  held-out data (recall 80–81%). Ollama's single-slot cache does 9.1x the
  recompute of a shared block cache (DECISIONS §19, §22; FINDINGS §20–22).

**Hypothesis scorecard** (each pre-registered and committed before its run):
- H1–H4 held in every pass (DECISIONS §33/§34).
- Q1 failed, Q2 held, Q3 held in 2 of 3 runs (§35).
- P1 held, P2 failed, T1 and T2 held (§36).
- R1 (ski-rental not better than keep-all at 1x): held pre-fix, now **fails
  in 1 cell** on the fixed engine (see §5).

`docs/MILESTONE3_REPORT_NOTES.md` is the source pack for the M3 report:
claims, a results table, method points, eight threats to validity,
corrections owed to the M2 brief, related work, and the figure list. **The M3
report is not due yet. Do not start writing it unless the owner asks.**

## 4. The current phase (since 2026-10-24): generalization

The motive is threat 3 in the notes: there is only one self-made workload,
with low size variance. Will an examiner believe the result generalizes?

Done:
1. **Seam refactor** (`ffd15e4`). `simulator/workload_sources.py` is a
   workload-source registry (`register_source`, `resolve_workload`, sealed
   sources gated by `MAARA_POLICIES_FROZEN`). Shared driver code moved to
   `experiments/common.py`. A new **p95 slowdown** metric divides latency by
   the task's lone replay time on an idle engine
   (`Engine._isolated_service_s`). Golden check: S1 reproduced exactly.
2. **Figures 1–6** (`8fc617d`). `experiments/plot_m3.py` writes them to
   `results/figures/m3/` via `make figures-m3`. Figure 4 (retention) still
   shows **pre-fix** numbers and needs a rebuild.
3. **Public workload: TraceLab v0.0.2** (uw-syfi/TraceLab, CC BY 4.0), real
   Claude Code and Codex sessions, pre-registered as DECISIONS §37 with
   hypotheses G1–G4.
   - **Mapping:** one user turn is a program, a chain of LLM rounds plus CPU
     tool nodes. All rounds run at qwen2.5-coder:7b rates. Context is capped
     at 32,768 tokens. Users are split by hash into dev and sealed, with 80
     turns sampled per split.
   - **Code and data:** `simulator/adapters/tracelab.py`,
     `experiments/build_tracelab.py`, and the derived data in
     `data/external/tracelab/`. The raw download lives outside the repo, in
     `~/maara_external/`, and is treated as untrusted data.
   - **Known threat:** KV reuse across a human's turns is not modelled.
4. **Two engine bugs, found and fixed:**
   - **KV leak in admission** (DECISIONS §8 item 11, `4049013`). Surviving
     blocks were re-created, which leaked `_entry_bytes`. Scheduling verdicts
     were unchanged (1,440 of 1,440); retention numbers moved (§5).
   - **Lone-replay memo filled memory** (§8 item 12, `87f952b`). The cache
     was keyed on full node reprs, which reached about 5 GB per worker and
     filled swap. It is now keyed by SHA-256 digest; results were unaffected.
5. **D1, the TraceLab dev map** (5 seeds, c = 0.3, development only, not
   scored). Output is `results/tracelab/D1_dev_c0.3.json`.
   - `program-fcfs` vs FCFS: 4 better / 0 worse / 36 tied on p95 latency,
     and 1 / 3 / 36 on p95 slowdown.
   - On p95 slowdown, PLAS was better than `program-fcfs` in 3 cells, and
     LSF and EDF in 6 each.
   - Batch 2 shows large cuts (−15 to −66% at load ≥ 0.8) with very wide
     CIs. Batches 4–8 show −2 to −9%.
   - **Batch 8 and batch 16 were identical**: the 8,192 MB KV budget caps
     concurrency at 6.
   - **Early read:** the headline looks weaker on TraceLab than on MAARA.
6. **Scope cuts** (owner's decisions, recorded in DECISIONS §37 before the
   freeze):
   - D2 and E2 (the c = 0.03 public maps) were dropped.
   - E1 runs batches 2/4/8 only, with the scored policies only (`fcfs`,
     `program-fcfs`, `plas`, `lsf`, `edf`), via the new `--policies` flag.
   - G1 is rescaled to: better in at least 15 of 30 cells and worse in at
     most 2, on both p95 latency and p95 slowdown.
7. **TraceLab policy freeze: `9297ff0`.** G2's precondition was measured
   before any policy run: the sealed lone-replay CV is **1.53**, which is
   at least 1.0, so **G2 applies** (`5deff32`).

**Pre-registered TraceLab hypotheses, as amended** (DECISIONS §37):
- **G1:** `program-fcfs` beats FCFS in at least 15 of 30 cells and is worse
  in at most 2, on p95 latency **and** p95 slowdown.
- **G2:** PLAS beats `program-fcfs` on p95 slowdown in at least 3 cells.
- **G3:** LSF and EDF are each better than `program-fcfs` in at most 2 cells
  on p95 latency.
- **G4:** in E3, keep-all's recompute is over 1% of makespan at util 0.8.

## 5. Corrections from the KV-leak fix (report both versions)

- **FINDINGS §35, 8x holdout, ski-rental vs keep-all:**
  - p95: +171.2* / +95.7* / −2.4 became **+42.7 / −0.1 / −28.1**, none
    significant. The ski-rental p95 penalty is gone.
  - Recompute: +164.1* / +118.4* / +136.1* became +56.2* / +22.6 / +43.3.
  - Evict-now's penalty holds.
- **FINDINGS §24, ski-rental extra recompute:** +10.2 s became +4.4 s, still
  significant.
- **Sealed 1x retention rerun** (owner-approved correction, same freeze
  `f4c889b`): `results/sealed/retention_headroom_final_qwen_fixed.json`. The
  pre-fix file is kept.

  | Headroom | KV MB | Ski-rental Δp95, pre → fixed | Ski-rental Δrecompute, pre → fixed | Ski-rental-pred Δp95, fixed |
  |---|---|---|---|---|
  | 0.25 | 418 → 452 | +1.1 → −6.7 | +15.9* → +4.1* | **−11.2* [−18.2, −4.2]** |
  | 0.50 | 466 → 500 | −2.8 → +0.3 | +16.6* → +6.4* | −2.6 |
  | 1.00 | 563 → 597 | +3.5 → −2.5 | +9.9* → +6.7* | −5.5 |

  - **R1 now fails on p95 in 1 of 6 cells** (ski-rental-pred, headroom
    0.25), about what chance would give. It must be reported as a failure.
  - Recompute still holds: no variant beats keep-all.
  - The budgets differ because headroom mode sizes them from the fixed
    engine's peak.
  - **FINDINGS §33–35, OPEN_ISSUES §31, the notes' §2 table and figure 4 are
    not yet updated with this.**

## 6. Running right now (do not disturb)

- **What:** E1 (sealed TraceLab scheduling map, c = 0.3) and E3 (sealed
  TraceLab retention grid, batch 8, utils 0.6 and 0.8, six retention
  policies, 20 seeds). They run in parallel.
- **Since:** 2026-10-25 20:00, at HEAD `5deff32`, clean tree.
- **ETA:** about 07:00–09:00 on 2026-10-26, to be re-checked after E1's
  first batch.
- **Runner and logs:**
  - Runner: `~/maara_runs/run_sealed_tl.sh`. Status: `~/maara_runs/run_sealed_tl.out`.
  - E1 log: `results/tracelab/E1_sealed_c0.3.log`.
  - E3 log: `results/tracelab/E3_sealed_retention.log`.
- **How it runs:** detached with `nohup … & disown` under `caffeinate -i`.
  Sleep pauses it; shutdown or reboot kills it, and there is no checkpoint.
- **While it runs:**
  - **Do not edit tracked files in this checkout.** Run manifests record
    `git_dirty`; untracked files are fine. Use a separate worktree
    (`git worktree add ../docs-m3 milestone-2`) for doc edits, and merge
    after.
  - **Do not start another CPU-heavy job.** Plotting and reading are fine.
- **After it finishes:**
  1. Check that every manifest has `git_dirty: false` and a `git_hash` after
     `9297ff0`.
  2. Score G1–G4 exactly as written into a new FINDINGS §41, using `hi_s < 0`
     and `lo_s > 0`.
  3. Update OPEN_ISSUES and threat 3 in the notes.
  4. Add figure 7 (TraceLab vs MAARA).
  5. Force-add the results (`git add -f`), back them up to
     `~/maara_backups/`, and push.

## 7. Still to do (not started unless the owner says so)

1. Fold the §5 corrections into FINDINGS §33–35, close OPEN_ISSUES §31,
   update the notes' §2 table, and rebuild figure 4.
2. Write D1 and both engine bugs into FINDINGS and OPEN_ISSUES.
3. Score E1 and E3 (§6 above).
4. Optional: a Weka kv-cache-tester robustness check, which needs its own
   pre-registration (a future DECISIONS §38).
5. The M3 report itself, only when the owner asks.

**Open issues** (OPEN_ISSUES index): 9.5 model-switch cost assumed to be 0;
14 MAARA write-up caveats; 24 `issue_parser` cache-recall gap; 27 no
operating point where ski-rental beats keep-all (16x untested); 31 KV leak,
fixed but its docs are pending.

## 8. Rules you must follow

**Data**
- `data/` holds measured inputs and is not regenerable. Never overwrite or
  regenerate it.
- `results/` is gitignored; add results with `git add -f`.
- `archive/` must not be imported.

**Pre-registration**
- Write every run into `docs/DECISIONS.md` and commit it before running it.
- Score results exactly as written. A failed hypothesis is reported as
  failed, and no policy is changed in response.

**Sealed sets**
- The sealed sets are `final_*` and `tracelab_sealed`. Use them only under
  `MAARA_POLICIES_FROZEN=<freeze commit>`.
- Never set that variable just to get past the error.
- No reruns without the owner's decision.

**Locked text**
- DECISIONS §3 is locked by the owner.
- §8 lists fixed bugs that must not be reintroduced.

**Engine and policies**
- Online policies must not read the oracle fields (`next_reuse_*`,
  `critical_path_remaining`).
- The engine must stay concurrent.
- The `sim` extra is stdlib + numpy + matplotlib only.

**Commands**
- Run everything through `./venv/bin/python`.
- Commands that call a live model need `CONFIRM=1`; never pass it unless
  asked.
- Wrap long jobs in `caffeinate -i`, and detach them with `nohup … &
  disown`. Run one CPU-heavy job at a time; the Mac has 5 performance cores
  and 18 GB of RAM.

**Git**
- Use `git add <paths>`, never `-A`.
- `CLAUDE.md` and `STATUS_REPORT.md` stay untracked.
- End commits with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Push with `git -c credential.helper='!gh auth git-credential' push -q
  origin milestone-2`.

**Hardware**
- The RTX 5060 Ti 16 GB is the ceiling. The GPU box is closed: don't plan
  GPU work, and never `sudo reboot` it.
- Don't invent GPU numbers.

**Owner preferences**
- Bottom line first, few bullets, end with the decision needed.
- When the owner asks "can you…", answer the question; don't act on it.
- **Before asking for approval, give a total wall-clock estimate for the
  whole job.** Per-step numbers have twice been 2.5x or more too low.

**Labels**
- The stored config label `single_agent` means the full pipeline. Map it for
  display with `simulator.trace_loader.config_label`.
- Every load figure must name its axis:
  - **Busy axis** (sealed §33 maps, figure 1): load 1.0 = 80% GPU busy.
  - **Capacity axis** (§36–40 and TraceLab): load = arrival rate ÷
    saturation throughput.

## 9. Where to look

| Need | File |
|---|---|
| Measured results, with corrections in place | `docs/FINDINGS.md` (status index at the top) |
| Every design decision and pre-registration | `docs/DECISIONS.md` (§26 sealed set, §31–37 pre-registrations) |
| What is unvalidated or wrong | `docs/OPEN_ISSUES.md` (index at the top) |
| M3 report source pack | `docs/MILESTONE3_REPORT_NOTES.md` |
| Narrative up to 2026-10-06 | `docs/STORY_SO_FAR.md` |
| Engine and simulator design | `docs/ARCHITECTURE.md`, `docs/TOUR.md` |
| Milestone 2 talk content | `docs/MILESTONE2_PRESENTATION_BRIEF.md` |
| Commands | `CLAUDE.md` (untracked), `Makefile`, every module's `--help` |
| Results | `results/sealed/`, `results/tracelab/`, `results/bugfix_check/`, `results/figures/m3/` |
