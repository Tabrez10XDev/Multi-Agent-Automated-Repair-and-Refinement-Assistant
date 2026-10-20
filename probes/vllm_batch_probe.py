"""
Measures batch_decode_coeff on a running vLLM server (G4 of docs/GPU_TODO.md).

    vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --max-model-len 16384 \\
        --gpu-memory-utilization 0.90 --enable-prefix-caching      # elsewhere
    ./venv/bin/python -u -m probes.vllm_batch_probe --backend cuda-5060ti \\
        [--model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ] [--url http://localhost:8000]

**Calls the model.** Writes data/calibration[/<backend>]/vllm_batch_decode.json,
never batch_decode.json (the Ollama fit). Nothing loads its output: which
batcher's coefficient the gpu_5060ti profile uses is a DECISIONS call.

WHAT IS MEASURED
    The same quantity, model and fit as probes/batch_decode_probe.py (its
    fit_coeff and overlap are reused, not copied): n identical-shape requests
    at once for n in --ns, and least squares through the origin of
        t_n / t_1 - 1 = c * (n - 1).
    vLLM has no per-request eval_duration, so a request's per-token time is
    read from its own stream: the mean gap between the chunks that carried
    tokens, from the first to the last. The first chunk's arrival includes
    queueing and prefill and is excluded; it is reported as ttft_s.

THE TRAPS THIS GUARDS AGAINST
    A queueing server: if vLLM runs fewer sequences than n, the extras wait,
    each one decodes in a smaller batch, and c reads low. Measured overlap,
    sum(request wall) / round wall, must be close to n, as in the Ollama probe.

    Coalesced chunks: a chunk gap is one decode step only if each chunk holds
    one token. If the server packs several tokens into a chunk, gaps measure
    something else. The final usage chunk gives completion_tokens; the probe
    aborts if tokens per chunk is not about 1.

    Prompts get a random nonce at the front so --enable-prefix-caching cannot
    serve them, and ignore_eos makes every request decode exactly
    --num-predict tokens. Timestamps are taken in Python threads; at n=8 the
    gaps are tens of milliseconds, well above the GIL's switching interval.
"""

import argparse
import json
import os
import random
import re
import statistics as st
import sys
import threading
import time
import urllib.error
import urllib.request

from probes.batch_decode_probe import MIN_OVERLAP_FRACTION, fit_coeff, overlap
from probes.calibrate import add_backend_arg, calibration_path

VLLM = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
OUT_NAME = "vllm_batch_decode.json"
DONE = "[DONE]"
# How far mean tokens per chunk may sit from 1 before gaps stop meaning
# per-token time.
MAX_TOKENS_PER_CHUNK_DRIFT = 0.05


# --------------------------------------------------------------------------
# Pure analysis (tested offline in tests/simulator/test_vllm_batch_probe.py)
# --------------------------------------------------------------------------

def default_out(backend: str) -> str:
    return calibration_path(backend, OUT_NAME)


def infer_quantization(model: str):
    """'awq', 'gptq' or 'fp8' from the model name, else None. The server does
    not report it, so --quantization overrides this."""
    m = re.search(r"(awq|gptq|fp8)", model, re.IGNORECASE)
    return m.group(1).lower() if m else None


def request_body(model: str, num_predict: int, nonce: int) -> dict:
    return {
        "model": model,
        # Raw completion, a counting task, nonce first so no prefix is cached.
        "prompt": (f"# run {nonce}\n"
                   "# Print the numbers from 1 to 1000, one per line.\n"
                   "1\n2\n3\n"),
        "max_tokens": num_predict, "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }


def parse_sse_line(raw):
    """One line of the server-sent-event stream: the chunk dict, DONE at the
    end of the stream, or None for blank lines and keep-alive comments."""
    line = (raw.decode() if isinstance(raw, bytes) else raw).strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == DONE:
        return DONE
    return json.loads(payload)


def stream_timing(t_start: float, events: list) -> dict:
    """
    `events` is [(arrival seconds, chunk)] in arrival order. Chunks with a
    non-empty `choices` carried tokens; the usage-only chunk at the end does
    not, and supplies completion_tokens.
    """
    token_times = [t for t, c in events if c.get("choices")]
    if len(token_times) < 2:
        raise ValueError("need at least two token chunks to time decode")
    usage = next((c["usage"] for _, c in reversed(events) if c.get("usage")), None)
    completion = usage.get("completion_tokens") if usage else None
    k = len(token_times)
    return {"per_token_s": (token_times[-1] - token_times[0]) / (k - 1),
            "ttft_s": token_times[0] - t_start,
            "chunks": k, "completion_tokens": completion,
            "tokens_per_chunk": completion / k if completion else None}


# --------------------------------------------------------------------------
# Live measurement
# --------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _stream(url: str, model: str, num_predict: int, timeout: int = 600) -> dict:
    body = json.dumps(request_body(model, num_predict,
                                   random.randint(10**9, 10**10))).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    events = []
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            chunk = parse_sse_line(raw)
            if chunk == DONE:
                break
            if chunk is not None:
                events.append((time.monotonic(), chunk))
    return stream_timing(t0, events)


def one_round(url: str, model: str, n: int, num_predict: int) -> dict:
    """Fire n streams at once; return per-token seconds and measured overlap."""
    results, walls, errors = [None] * n, [0.0] * n, []

    def worker(i):
        t0 = time.monotonic()
        try:
            results[i] = _stream(url, model, num_predict)
        except Exception as e:  # noqa: BLE001 -- reported, then the round fails
            errors.append(repr(e))
        walls[i] = time.monotonic() - t0

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    round_wall = time.monotonic() - t0
    if errors:
        raise RuntimeError(f"n={n}: {len(errors)} request(s) failed: {errors[0]}")
    tpc = [r["tokens_per_chunk"] for r in results if r["tokens_per_chunk"]]
    return {"n": n, "per_token_s": st.fmean(r["per_token_s"] for r in results),
            "ttft_s": st.fmean(r["ttft_s"] for r in results),
            "chunks": [r["chunks"] for r in results],
            "completion_tokens": [r["completion_tokens"] for r in results],
            "tokens_per_chunk": st.fmean(tpc) if tpc else None,
            "overlap": overlap(walls, round_wall), "round_wall_s": round_wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=VLLM)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quantization", default=None,
                    help="recorded in the output; default: inferred from --model")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--out", default=None,
                    help=f"default: data/calibration[/<backend>]/{OUT_NAME}")
    add_backend_arg(ap)
    args = ap.parse_args()
    args.out = args.out or default_out(args.backend)
    args.url = args.url.rstrip("/")
    if 1 not in args.ns:
        args.ns = [1] + args.ns
    quant = args.quantization or infer_quantization(args.model)

    try:
        version = _get_json(f"{args.url}/version").get("version")
        served = _get_json(f"{args.url}/v1/models").get("data", [])
    except (urllib.error.URLError, OSError, ValueError):
        print(f"vLLM is not reachable at {args.url}; start it and re-run.")
        return 1
    entry = next((m for m in served if m.get("id") == args.model), None)
    if entry is None:
        print(f"{args.url} does not serve {args.model!r}; it serves "
              f"{[m.get('id') for m in served]}.")
        return 1

    print(f"vllm {version}, model {args.model} ({quant}), ns {args.ns}, "
          f"{args.repeats} repeats, {args.num_predict} tokens each", flush=True)
    print("warming up...", flush=True)
    one_round(args.url, args.model, 1, 16)

    rounds, t_by_n = [], {}
    for n in args.ns:
        reps = [one_round(args.url, args.model, n, args.num_predict)
                for _ in range(args.repeats)]
        rounds.extend(reps)
        worst = min(r["overlap"] for r in reps)
        tpc = [r["tokens_per_chunk"] for r in reps if r["tokens_per_chunk"]]
        t_by_n[n] = st.fmean(r["per_token_s"] for r in reps)
        print(f"  n={n:<2} per-token {t_by_n[n]*1000:6.1f} ms  "
              f"overlap {worst:.2f} (min over repeats)  "
              f"tokens/chunk {st.fmean(tpc) if tpc else float('nan'):.2f}", flush=True)
        if n > 1 and worst < MIN_OVERLAP_FRACTION * n:
            print(f"\nABORT: at n={n} only {worst:.2f} requests overlapped. The "
                  f"server is queueing, so the fit would read low. Raise vLLM's "
                  f"--max-num-seqs to at least {max(args.ns)} and re-run.")
            return 2
        if not tpc or any(abs(x - 1) > MAX_TOKENS_PER_CHUNK_DRIFT for x in tpc):
            print(f"\nABORT: at n={n} tokens per chunk is {tpc or 'unreported'}, "
                  f"not 1, so chunk gaps are not per-token times. Check that the "
                  f"server streams one token per chunk and reports usage.")
            return 2

    fit = fit_coeff(t_by_n)
    print(f"\nbatch_decode_coeff = {fit['batch_decode_coeff']:.3f}  "
          f"(per n: " + ", ".join(f"{n}: {c:.3f}" for n, c in
                                   fit["implied_coeff_by_n"].items()) + ")")
    print(f"single-stream decode {fit['single_stream_decode_rate']:.1f} tok/s")

    out = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": args.backend,
           "server": "vllm", "vllm_version": version, "url": args.url,
           "model": args.model, "quantization": quant,
           "max_model_len": entry.get("max_model_len"),
           "num_predict": args.num_predict, "repeats": args.repeats,
           "timing": "mean gap between streamed token chunks, first chunk excluded",
           "fit": fit, "rounds": rounds}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
