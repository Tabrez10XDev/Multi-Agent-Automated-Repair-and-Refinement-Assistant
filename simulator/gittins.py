"""
Program-level Gittins index over requests completed (DECISIONS 38).

A task's state is k, the number of its requests that have finished. The prior
is a frozen distribution of R, requests per task, with expected GPU work w_j
for request j. The Gittins index of state k is the best achievable ratio of
"chance of finishing" to "expected work spent", over every stopping point m:

    index(k) = max over m > k of (S[k] - S[m]) / (C[m] - C[k])

where S[j] = #(R > j) and C[j] = sum_{i<j} S[i] * w_i. Serving the highest
index first minimises mean response time in the single-server model it comes
from (Aalto, Ayesta & Righter 2009). It reduces to oldest-first when R has an
increasing hazard and to fewest-completed-first when the hazard decreases,
which is the switch the two workloads need (FINDINGS 41).

weight="slowdown" replaces S in the numerator by G[j] = sum_{r>j} n_r / size_r,
so finishing a small task counts for more. A heuristic for mean slowdown, not
proven optimal. size_r = sum_{j<r} w_j + r * tool_s, the lone-replay size.

Tool and CPU time between requests is not service: it is in the size (for the
slowdown weight and the guard's E[S]) but never in C.

Priors are built offline from development data, never from the run's tasks:
`tracelab_prior` from the frozen dev constants, `maara_prior` from the fit
traces. They read recorded token counts of FIT tasks, which is the
experimenter sizing the policy, as dev_constants.json does for TraceLab.
"""

import math

import numpy as np

from simulator.workload import CODER_MODEL


class GittinsTable:
    """Index by state k, from counts {R: n} and per-request work w_j (the last
    entry of `work` repeats for every later request). States whose survivor
    count S[k] is below `min_count` share the index of the last state above it,
    so a handful of very long tasks cannot set the index on their own."""

    def __init__(self, r_counts, work, tool_s=0.0, weight="latency", min_count=30):
        if weight not in ("latency", "slowdown"):
            raise ValueError(f"weight must be 'latency' or 'slowdown', not {weight!r}")
        r_max = max(r for r, n in r_counts.items() if n > 0)
        n = np.zeros(r_max + 1)
        for r, c in r_counts.items():
            if r >= 1 and c > 0:
                n[r] = c
        w = np.array([work[min(j, len(work) - 1)] for j in range(r_max)], dtype=float)
        # Zero work makes C flat, and the index inf or NaN: a NaN sort key
        # gives an undefined order, so refuse it here.
        if n.sum() <= 0 or not np.all(w > 0) or tool_s < 0:
            raise ValueError("GittinsTable needs tasks with R >= 1, positive work "
                             "and non-negative tool time")
        # S[j] = tasks with more than j requests; S[r_max] = 0.
        self.S = n[::-1].cumsum()[::-1] - n
        self.C = np.concatenate(([0.0], np.cumsum(self.S[:-1] * w)))
        cum_w = np.concatenate(([0.0], np.cumsum(w)))
        self.size = cum_w + np.arange(r_max + 1) * tool_s     # size[r], r requests
        if weight == "latency":
            self.A = self.S
        else:
            per = np.zeros(r_max + 1)
            per[1:] = n[1:] / self.size[1:]
            self.A = per[::-1].cumsum()[::-1] - per
        self.mean_size = float((n * self.size).sum() / n.sum())
        above = np.nonzero(self.S >= min_count)[0]
        self.k_tail = int(above[-1]) if len(above) else 0
        self.r_max = r_max
        self._memo = {}

    def index(self, k: int) -> float:
        k = min(max(int(k), 0), self.k_tail)
        if k not in self._memo:
            if k >= self.r_max:
                self._memo[k] = 0.0
            else:
                m = np.arange(k + 1, self.r_max + 1)
                self._memo[k] = float(np.max((self.A[k] - self.A[m])
                                             / (self.C[m] - self.C[k])))
        return self._memo[k]


def pooling_threshold(n_tasks: int) -> int:
    """30, or 20% of the prior's tasks when it has fewer than 150 (DECISIONS 38
    amendment): MAARA's fit prior has 15 tasks."""
    return min(30, max(1, math.ceil(0.2 * n_tasks)))


def tracelab_prior(cost, constants, model=CODER_MODEL):
    """(r_counts, work, tool_s) from the frozen dev constants (DECISIONS 38): a
    cold first round (mean prefix + append prefill, mean decode) and cached
    later rounds (append prefill only), batch 1."""
    dec = constants["mean_output_tokens"] * cost.decode_iteration_seconds(model, 1)
    first = cost.prefill_seconds(model, constants["mean_prefix_tokens"]
                                 + constants["mean_append_tokens"]) + dec
    later = cost.prefill_seconds(model, constants["mean_append_tokens"]) + dec
    r_counts = {int(r): c for r, c in constants["rounds_per_turn_counts"].items()}
    return r_counts, [first, later], constants["mean_tool_s_per_round"]


_MAARA_PRIOR = {}


def maara_prior(cost, fit_set="original", config="single_agent"):
    """(r_counts, work, tool_s) from the MAARA fit traces, restricted to the
    configuration holdout_qwen was collected under (DECISIONS 38): R is the
    number of LLM calls per task, w_j the mean batch-1 cost of each task's j-th
    call in node order, tool_s the mean CPU seconds per call."""
    from simulator.trace_loader import load_trace, resolve_trace_set
    from simulator.workload import Kind
    key = (fit_set, config, repr(cost))        # every cost field, overrides included
    if key not in _MAARA_PRIOR:
        r_counts, by_j, cpu_s, calls = {}, {}, 0.0, 0
        for path in resolve_trace_set(fit_set):
            lt = load_trace(path, arrival_rate=0.01, seed=1)
            for task, cfg in zip(lt.tasks, lt.configs):
                if cfg != config:
                    continue
                llm = [x for x in sorted(task.nodes, key=lambda x: x.node_id)
                       if x.kind is not Kind.CPU]
                cpu_s += sum(x.cpu_seconds for x in task.nodes if x.kind is Kind.CPU)
                calls += len(llm)
                r_counts[len(llm)] = r_counts.get(len(llm), 0) + 1
                for j, x in enumerate(llm):
                    s = (cost.prefill_seconds(x.model, x.prompt_tokens)
                         + x.output_tokens * cost.decode_iteration_seconds(x.model, 1))
                    by_j.setdefault(j, []).append(s)
        work = [sum(v) / len(v) for _, v in sorted(by_j.items())]
        _MAARA_PRIOR[key] = (r_counts, work, cpu_s / calls if calls else 0.0)
    return _MAARA_PRIOR[key]
