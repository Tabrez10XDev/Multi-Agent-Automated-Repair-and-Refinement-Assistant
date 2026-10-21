# GPU work: plan for separate agents

This plan splits the GPU work in `docs/GPU_TODO.md` into five agent-sized tasks. Each task says who may run what, what it needs first, and what it hands back. Read `CLAUDE.md` and `docs/GPU_TODO.md` before starting any of them.

## The machines

| Machine | Role | Reach |
|---|---|---|
| Mac (this repo) | Simulator, code, analysis. The **only** place commits are made. | Local |
| GPU box `tm3578@gcis-tm3578-rl.gccis.rit.edu` | RTX 5060 Ti 16 GB. Runs probes and model servers only. | SSH, **RIT VPN required** (port 22 times out off-campus) |

Two prerequisites, done by the owner before agents 1, 2 and 4 start:

1. **Connect to the RIT VPN.**
2. **Install the key once:** `ssh-copy-id -i ~/.ssh/id_rsa.pub tm3578@gcis-tm3578-rl.gccis.rit.edu`. Agents can't type passwords, so they connect with `ssh -o BatchMode=yes`.

## Rules every agent follows

- **Model calls need a yes from the owner.** Probes, `ollama pull`, `vllm serve` and any download over 1 GB are confirmed with the owner before they run. Read-only checks need no confirmation.
- **The GPU box doesn't commit.** It writes only to `data/calibration/cuda-5060ti/` in its checkout. Files come back to the Mac by `scp`, and get committed there.
- **Never overwrite Mac data.** Every probe runs with `--backend cuda-5060ti`; without it, the probe writes over the Mac's calibration files. Data in `data/` is measured input and is never regenerated.
- **Stay within 16 GB.** No bf16 7B model in vLLM, and nothing beyond a 16k context (`docs/GPU_TODO.md`, "Out of scope").
- **Keep long jobs alive.** Run them under `tmux` or `nohup` on the GPU box, and log `nvidia-smi` state every minute while they run.
- **Leave the Mac's background runs alone.** The sealed runs (from `../sealed-f4c889b`, writing to `results/sealed/*_full.*` and `*_final_llama.*`) and the 8x retention grid (`results/retention_headroom_8x.*`) are running. Don't touch those paths or that worktree. Keep the Mac plugged in with the lid open.

## Order

```
Agent 1 (GPU box audit + setup) ──► Agent 2 (Ollama probes G1–G3) ──► Agent 5 (fill profile, reruns)
Agent 3 (Mac code prep) ─────────────────────────► Agent 4 (vLLM, G4) ──┘
```

Agents 1 and 3 can start at once. Agent 2 needs Agent 1. Agent 4 needs Agents 1 and 3. Agent 5 needs Agent 2, and Agent 4 if it runs.

---

## Agent 1: audit and set up the GPU box (G0)

**Goal:** a GPU box that can run the repo's probes against Ollama on CUDA, with versions recorded.

**Step 1, read-only. Report back before step 2.**
- `nvidia-smi`: confirm an RTX 5060 Ti with 16 GB, and read the driver version.
- CUDA version (`nvidia-smi` header, `nvcc --version` if present). Blackwell (sm_120) needs CUDA 12.8 or newer.
- OS: `/etc/os-release` and `uname -a`. Native Linux or WSL2? vLLM won't run on native Windows.
- `sudo -n true` (do we have sudo?), `python3 --version`, and whether `uv` or `pyenv` exists. The repo needs Python 3.13 or newer.
- `ollama --version`, and whether it is already running.
- Free disk (`df -h ~`, about 25 GB needed) and RAM (`free -h`).
- Who else uses the box (`who`, and the processes in `nvidia-smi`): it's a lab machine.

**Step 2, setup. Each install confirmed with the owner first.**
- Ollama with CUDA. Use the official install if we have sudo. Otherwise use the user-space tarball, run as `ollama serve` under tmux.
- Pull `qwen2.5-coder:7b` and `llama3.2`. Confirm with `ollama show` that both are Q4_K_M, the same as on the Mac.
- Get the repo onto the box at branch `milestone-2`. Either `git clone` (the GitHub repo may need a token), or create a git bundle on the Mac and `scp` it over.
- `python -m venv venv && ./venv/bin/pip install -e '.[sim,dev,agent]'`, then run `make test` there as a smoke check.
- Write `data/calibration/cuda-5060ti/machine.json` with the GPU, driver, CUDA, OS, Ollama and Python versions.

**Hands back:** the audit report, plus `machine.json` copied to the Mac.

**Stop and ask if:** there's no sudo and no way to install a user-space Ollama, the box is native Windows, someone else is using the GPU, or CUDA is below 12.8.

---

## Agent 2: Ollama probes on CUDA (G1–G3)

**Needs:** Agent 1 done, and owner approval for each probe. Each one calls the model.

1. **G1:** `./venv/bin/python -u -m probes.calibrate --backend cuda-5060ti` (about 15 min) → `data/calibration/cuda-5060ti/measured.json`.
2. **G2:** restart Ollama as `OLLAMA_NUM_PARALLEL=8 ollama serve`, then `./venv/bin/python -u -m probes.batch_decode_probe --backend cuda-5060ti` (about 10 min) → `batch_decode.json`. If the probe aborts on low overlap, the server is queueing: check `OLLAMA_NUM_PARALLEL`. Restart Ollama normally afterwards.
3. **G3:** from the Ollama server log, copy one KV-cache line per model ("… tokens, … MiB" or `llama_kv_cache: size = …`) into `data/calibration/cuda-5060ti/kv_log_lines.txt`. Expected values: qwen 57,344 and llama3.2 114,688 bytes per token.
4. Run `nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used --format=csv -l 60` alongside each probe, saved next to its output.
5. `scp` the whole `cuda-5060ti/` directory to the Mac.

**Sanity checks before handing back:** `backend` is `cuda-5060ti` in both JSONs. Rates should be well above the Mac's (qwen prefill 268 tok/s, decode 28 tok/s). The coefficient's per-n values should agree within about 0.1.

---

## Agent 3: code prep on the Mac (no model calls)

**Needs:** nothing. Can start now.

1. **Commit the GPU profile loader that's already written.** In `simulator/workload.py`, `gpu_rates_from` and `gpu_coeff_from` load `data/calibration/cuda-5060ti/` into `gpu_5060ti` and refuse files from another backend. Tests are in `tests/simulator/test_gpu_profile.py` (8 passing). Both are uncommitted. Run `make test`, then commit.
2. **Add a `--profile` flag** to `experiments/scheduling_map.py`, `run_retention_grid.py`, `run_scheduler_table.py` and `sweep_decode_coeff.py`. Each builds its cost with `CostModel.from_profile(args.profile)` (`simulator/workload.py`), defaulting to `mac_ollama_measured`. The places that build costs today:
   - `scheduling_map.py:204`;
   - `run_retention_grid.py:217, 231, 414`;
   - `run_scheduler_table.py:150` (its manifest string at :168 needs updating too);
   - `sweep_decode_coeff.py:99`.

   In `run_retention_grid.py`, the cost model is also rebuilt inside helpers, so thread the profile through those. Manifests already record `hardware_profile`. Add a test that `--profile gpu_5060ti` reaches the manifest, and check that `scheduling_map --quick --profile gpu_5060ti` runs end to end.
3. **Write `probes/vllm_batch_probe.py`** for G4. It does the same measurement as `probes/batch_decode_probe.py`, reusing its `fit_coeff` and `overlap`, against vLLM's OpenAI-compatible `/v1/completions` with `stream: true`. Per-token time comes from the gaps between streamed chunks. It needs `--url` (default `http://localhost:8000`), `--backend` via `add_backend_arg`, and `--model`.
   - Write to `vllm_batch_decode.json` (not `batch_decode.json`), with `"server": "vllm"` and the quantisation recorded.
   - Add offline tests for stream parsing, like `tests/simulator/test_batch_decode_probe.py`, and a `make probe-vllm-batch` target gated by `CONFIRM=1`.
4. Don't build `vllm_prefix_replay.py` (G7) yet.

**Must not:** edit `docs/DECISIONS.md` §3, touch `data/`, or touch the sealed worktree or the running results paths.

---

## Agent 4: vLLM on the GPU box (G4, optional)

**Needs:** Agents 1 and 3 done (pull `milestone-2` on the box for the new probe). Owner approval for the vLLM install and the model download.

1. Install vLLM in its own venv. Its PyTorch and vLLM builds must support sm_120, so check the versions at install time rather than assuming.
2. `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching`, under tmux.
3. `./venv/bin/python -u -m probes.vllm_batch_probe --backend cuda-5060ti --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`, then `scp` the output to the Mac.

**Stop and ask if:** the AWQ model plus KV doesn't fit in 16 GB at 16k context, or the vLLM build doesn't support sm_120.

---

## Agent 5: use the GPU numbers (Mac, CPU only)

**Needs:** Agent 2's files on the Mac, and Agent 3's code committed.

1. Commit `data/calibration/cuda-5060ti/`. Check `CostModel.from_profile("gpu_5060ti").provenance()` shows MEASURED for the loaded fields. If G3 confirmed KV, add both models to the profile's `kv_measured`.
2. Write a new DECISIONS section (next free number) recording which batching coefficient `gpu_5060ti` uses, Ollama's (G2) or vLLM's (G4), and why. If it is vLLM's, point `gpu_coeff_from` at `vllm_batch_decode.json`.
3. Run `experiments.sweep_decode_coeff` at the GPU value.
4. Rerun the scheduling-map and retention **boundary cells** on `holdout_qwen` with `--profile gpu_5060ti`. Don't run the GPU profile on the sealed `final` set without a DECISIONS ruling first (§26: one run per frozen policy set).
5. Add FINDINGS sections for the GPU calibration and the reruns. Update OPEN_ISSUES §9.4 and tick off `docs/GPU_TODO.md`.

---

## When it's done

- `data/calibration/cuda-5060ti/` has `machine.json`, `measured.json` and `batch_decode.json` (plus `vllm_batch_decode.json` if G4 ran), all committed on the Mac.
- `make test` passes, and `--profile gpu_5060ti` works on every driver.
- FINDINGS states, with numbers, whether the scheduling and retention conclusions hold on the GPU's costs.
