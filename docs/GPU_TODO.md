# GPU TODO: what to do once the RTX 5060 Ti arrives

> **TENTATIVE (2026-10-07).** The GPU is not confirmed, so no time goes into
> preparing for it. The unchecked prep items below are built only once GPU
> time is booked (about half a day). The minimum useful session, G0 to G3,
> needs no new code: the probes already take `--backend`.

**Target machine: one NVIDIA RTX 5060 Ti, 16 GB.** This is the ceiling for this
repo (DECISIONS §29). Nothing here, and no future work, may need more VRAM, a
second GPU, or a datacenter card. If a step does not fit in 16 GB, it gets
scaled down or dropped, never moved to bigger hardware.

The GPU **improves** the project; it does not unblock it. The prototype is
complete on the Mac. The GPU answers one question the Mac cannot: with a
different prefill-to-decode ratio and a real continuous batcher, does
retention or scheduling start to matter?

Every number measured here lands in the `gpu_5060ti` hardware profile
(`simulator/workload.py`), never in `mac_ollama_measured`.

---

## Will it fit? (16 GB budget)

Sizes are approximate; check the real numbers at setup (G0) and record them.

| What | Approx. VRAM | Fits? |
|---|---|---|
| qwen2.5-coder:7b, Ollama default (Q4_K_M, the same as on the Mac) | ~4.7 GB | yes |
| llama3.2 3B, Ollama default (Q4_K_M) | ~2.0 GB | yes |
| Both Ollama models resident at once + KV | ~7 GB + KV | yes |
| qwen2.5-coder-7b in vLLM, **AWQ int4** | ~5.5 GB + KV | yes, leaves ~8 GB of KV |
| qwen2.5-coder-7b in vLLM, FP8 | ~8 GB + KV | yes, leaves ~5 GB of KV |
| qwen2.5-coder-7b in vLLM, **bf16** | ~15.2 GB weights alone | **no**, do not use |
| KV per 16k-token sequence (qwen, 56 KiB/token) | ~0.9 GB | yes |
| The simulator's KV budgets (512 to 2048 MB) | under 2 GB | yes |

Rules that follow from this:

- **vLLM uses one model per server, AWQ int4**, `--max-model-len 16384`,
  the largest `num_ctx` the collected traces ran under (FINDINGS §20 records
  4096, 8192 and 16384). Run the two models in separate passes, not side by side.
- **Ollama on CUDA uses the same model tags as the Mac**, so the quantisation
  matches and the profiles are comparable.
- The regime map's 32x and 50x context scales are **simulator-only**. They are
  synthetic and exceed qwen's real context window, so they are never run on
  the GPU. Real GPU runs stay at 1x context.

---

## Prep work (build only once GPU time is booked)

- [x] **Probes write per-backend files.** `--backend cuda-5060ti` writes to
      `data/calibration/cuda-5060ti/`, never over the Mac's files
      (`tests/simulator/test_calibration_paths.py`).
- [x] **Placeholder profile retargeted** from `gpu_4060` (8 GB) to
      `gpu_5060ti` (16 GB), every field still marked unmeasured.
- [x] **Load the GPU calibration into `gpu_5060ti`.** Read
      `data/calibration/cuda-5060ti/measured.json` and `batch_decode.json`
      when present; flip `measured` only for a field that file actually
      contains; refuse a file whose `backend` is not `cuda-5060ti`. Tests
      offline, as for the Mac profile.
- [x] **Write `probes/vllm_batch_probe.py`**: the same fit as
      `batch_decode_probe.py` (reuse `fit_coeff` and `overlap`), against
      vLLM's OpenAI-compatible endpoint, reading per-token times from
      streaming timestamps. Offline tests for the parsing, like the Ollama one.
- [ ] **Write `probes/vllm_prefix_replay.py`** for G7: replay a trace's
      prompts from its sidecar in order, read vLLM's prefix-cache hit counters
      from `/metrics`, and output a hit rate comparable to the simulator's
      block cache on the same trace.
- [x] **Have the drivers record `hardware_profile` in the manifest** and take
      `--profile`, so a GPU rerun is one flag (still open from brief 02).

---

## On the GPU, in order

All of these call a model, so **the owner runs them**. Log machine state
(`nvidia-smi`, temperature, clocks) during each run, per OPEN_ISSUES §22.

### G0. Setup and sanity (about 1 hour)
- [ ] Linux or WSL2 (vLLM does not run on native Windows).
- [ ] Driver and CUDA 12.8 or newer: the 50-series (Blackwell, sm_120) needs
      it. PyTorch and vLLM builds must support sm_120; check versions at
      setup time rather than trusting this file.
- [ ] Ollama with CUDA. Pull `qwen2.5-coder:7b` and `llama3.2`, and confirm
      with `ollama show` that the quantisation matches the Mac.
- [ ] Record GPU, driver, CUDA, Ollama and vLLM versions in
      `data/calibration/cuda-5060ti/machine.json`.

### G1. Ollama single-stream rates (about 15 minutes)
```bash
./venv/bin/python -u -m probes.calibrate --backend cuda-5060ti
```
Gives prefill and decode for both models, under the same method as the Mac.

### G2. Ollama batching coefficient (about 10 minutes)
```bash
OLLAMA_NUM_PARALLEL=8 ollama serve      # terminal 1
./venv/bin/python -u -m probes.batch_decode_probe --backend cuda-5060ti
```
Restart Ollama normally afterwards.

### G3. KV bytes per token, both models (5 minutes)
- [ ] One server-log line per model ("N tokens, X MiB"). This also gives the
      first measured llama3.2 value (OPEN_ISSUES §23).

### G4. vLLM batching coefficient (about 30 minutes; only if time allows)
```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching
./venv/bin/python -u -m probes.vllm_batch_probe --backend cuda-5060ti
```
This is the coefficient for the batcher the simulator actually models. Expect
it to be far below the Mac's 0.841 (DECISIONS §28). Record that it is AWQ,
not Q4_K_M.

### G5. Fill the profile (Claude, no model calls)
- [ ] Load G1 to G4 into `gpu_5060ti`. Record in DECISIONS which coefficient
      the profile uses, Ollama's (G2) or vLLM's (G4), and why.
- [ ] Rerun `experiments.sweep_decode_coeff` at the GPU value.

### G6. Rerun under the GPU profile (Claude, CPU only)
- [ ] Scheduling map and retention map: **boundary cells only**, under
      `--profile gpu_5060ti`. The simulator itself needs no GPU; only the
      constants change.
- [ ] Headline policy table on `final`, but only if the policy freeze rules
      allow it (DECISIONS §26: one run, no tuning).

### G7. Optional, only if time allows: validate the block cache against real vLLM (about 1 hour)
```bash
./venv/bin/python -u -m probes.vllm_prefix_replay --trace holdout_qwen --backend cuda-5060ti
```
Compare vLLM's measured prefix-cache hit rate with the simulator's block cache
on the same trace, as §19 did for the Ollama model. qwen traces only, one
model per server.

---

## Out of scope: needs more than this GPU

Do not plan or build any of these:

- bf16 or fp16 7B weights in vLLM (about 15 GB of weights, no room for KV)
- larger models, tensor parallelism, or more than one GPU
- running both models in vLLM side by side at useful KV budgets
- real long-context runs beyond 16k tokens, including the regime map's
  synthetic 32x and 50x scales
- datacenter-only features (H100-class FP8 kernels, NVLink, MIG)
