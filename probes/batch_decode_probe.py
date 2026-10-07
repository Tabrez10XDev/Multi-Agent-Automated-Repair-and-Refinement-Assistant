"""
Measures CostModel.batch_decode_coeff on the running Ollama server.

    OLLAMA_NUM_PARALLEL=8 ollama serve        # in another terminal, fresh start
    ./venv/bin/python -u -m probes.batch_decode_probe [--model qwen2.5-coder:7b]

**Calls the model.** Writes data/calibration/batch_decode.json; it never
touches measured.json, and nothing loads its output automatically. Adopting a
value means editing the mac_ollama_measured profile in simulator/workload.py
in a commit that says so (DECISIONS §27, item 6).

WHAT IS MEASURED
    The simulator prices one decode step for a batch of n as
        (1 / decode_rate) * (1 + (n - 1) * batch_decode_coeff)
    so the per-token time a single stream sees grows linearly in n with slope
    batch_decode_coeff. The probe fires n identical-shape requests at once for
    n in --ns, reads each request's own eval_duration / eval_count from
    Ollama's counters, and fits
        t_n / t_1 - 1 = c * (n - 1)
    by least squares through the origin. The per-n implied c is reported too:
    if it drifts a lot with n, the linear model itself is wrong, and that is a
    finding, not noise to average away.

THE TRAP THIS GUARDS AGAINST
    If the server runs fewer slots than n, Ollama queues the extra requests.
    Each one then decodes alone, its eval_duration looks single-stream, and
    the fit returns c close to 0: a silently wrong answer that would make
    batching look free. Every round therefore checks measured overlap,
    sum(request wall time) / round wall time, and the probe aborts if it is
    well below n.

    Prompts are short and decodes long, so a request's prefill interleaving
    with another's decode steps barely inflates eval_duration. Each prompt
    gets a random nonce so Ollama's own prompt cache cannot serve it.
"""

import argparse
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO, "data", "calibration", "batch_decode.json")

# Must be identical on every request: a different num_ctx makes Ollama reload
# the model, and with OLLAMA_NUM_PARALLEL=n it allocates n of these.
NUM_CTX = 2048
MIN_OVERLAP_FRACTION = 0.8


# --------------------------------------------------------------------------
# Pure analysis (tested offline in tests/simulator/test_batch_decode_probe.py)
# --------------------------------------------------------------------------

def per_token_seconds(resp: dict) -> float:
    """One request's single-stream view of decode speed, from Ollama's counters."""
    if not resp.get("eval_count"):
        raise ValueError("response has no decoded tokens; cannot time decode")
    return resp["eval_duration"] / 1e9 / resp["eval_count"]


def overlap(request_walls: list, round_wall: float) -> float:
    """How many requests were in flight on average. About n if truly parallel,
    about 1 if the server queued them."""
    return sum(request_walls) / round_wall


def fit_coeff(t_by_n: dict) -> dict:
    """
    Least squares through the origin of  t_n / t_1 - 1  on  n - 1.
    `t_by_n` maps n to the mean per-token seconds at that n; it must include 1.
    """
    if 1 not in t_by_n:
        raise ValueError("need the n=1 baseline to normalise against")
    t1 = t_by_n[1]
    xs, ys, implied = [], [], {}
    for n, t in sorted(t_by_n.items()):
        if n == 1:
            continue
        x, y = n - 1, t / t1 - 1
        xs.append(x)
        ys.append(y)
        implied[n] = y / x
    if not xs:
        raise ValueError("need at least one n > 1")
    c = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    resid = [y - c * x for x, y in zip(xs, ys)]
    return {"batch_decode_coeff": c,
            "implied_coeff_by_n": implied,
            "max_abs_residual": max(abs(r) for r in resid),
            "single_stream_decode_rate": 1.0 / t1}


# --------------------------------------------------------------------------
# Live measurement
# --------------------------------------------------------------------------

def _prompt() -> str:
    # Raw mode, a counting task: it decodes steadily and rarely stops early.
    return (f"# run {random.randint(10**9, 10**10)}\n"
            "# Print the numbers from 1 to 1000, one per line.\n"
            "1\n2\n3\n")


def _generate(model: str, num_predict: int, timeout: int = 600) -> dict:
    body = json.dumps({
        "model": model, "prompt": _prompt(), "stream": False, "raw": True,
        "options": {"num_predict": num_predict, "temperature": 0, "num_ctx": NUM_CTX},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one_round(model: str, n: int, num_predict: int) -> dict:
    """Fire n requests at once; return per-token seconds and measured overlap."""
    results, walls, errors = [None] * n, [0.0] * n, []

    def worker(i):
        t0 = time.monotonic()
        try:
            results[i] = _generate(model, num_predict)
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
    per_tok = [per_token_seconds(r) for r in results]
    return {"n": n, "per_token_s": st.fmean(per_tok),
            "eval_counts": [r["eval_count"] for r in results],
            "overlap": overlap(walls, round_wall), "round_wall_s": round_wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    if 1 not in args.ns:
        args.ns = [1] + args.ns

    try:
        version = json.loads(urllib.request.urlopen(f"{OLLAMA}/api/version",
                                                    timeout=5).read()).get("version")
    except (urllib.error.URLError, OSError):
        print("ollama is not reachable at localhost:11434; start it and re-run.")
        return 1

    print(f"ollama {version}, model {args.model}, ns {args.ns}, "
          f"{args.repeats} repeats, {args.num_predict} tokens each", flush=True)
    print("warming up...", flush=True)
    _generate(args.model, 8)

    rounds, t_by_n = [], {}
    for n in args.ns:
        reps = [one_round(args.model, n, args.num_predict) for _ in range(args.repeats)]
        rounds.extend(reps)
        worst = min(r["overlap"] for r in reps)
        t_by_n[n] = st.fmean(r["per_token_s"] for r in reps)
        print(f"  n={n:<2} per-token {t_by_n[n]*1000:6.1f} ms  "
              f"overlap {worst:.2f} (min over repeats)", flush=True)
        if n > 1 and worst < MIN_OVERLAP_FRACTION * n:
            print(f"\nABORT: at n={n} only {worst:.2f} requests overlapped. The "
                  f"server is queueing, so the fit would read near 0. Restart it "
                  f"with OLLAMA_NUM_PARALLEL>={max(args.ns)} and re-run.")
            return 2

    fit = fit_coeff(t_by_n)
    print(f"\nbatch_decode_coeff = {fit['batch_decode_coeff']:.3f}  "
          f"(per n: " + ", ".join(f"{n}: {c:.3f}" for n, c in
                                   fit["implied_coeff_by_n"].items()) + ")")
    print(f"single-stream decode {fit['single_stream_decode_rate']:.1f} tok/s "
          f"(measured.json has the calibrated rate; compare, do not replace)")

    out = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": "mac-metal",
           "ollama_version": version, "model": args.model, "num_ctx": NUM_CTX,
           "num_predict": args.num_predict, "repeats": args.repeats,
           "fit": fit, "rounds": rounds}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
