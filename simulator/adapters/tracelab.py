"""
TraceLab adapter (DECISIONS 37): a user turn = one Task, a chain DAG.

Reads the DERIVED files written by experiments/build_tracelab.py
(data/external/tracelab/{dev,sealed,sealed2}.jsonl.gz), never the raw release.
Both sealed names are registered sealed=True, so the registry's
MAARA_POLICIES_FROZEN gate runs before this module opens the file.

DAG, in round order: an LLM node per round (stage "agent_round", model
CODER_MODEL, tokens from the clipped values), then a CPU node (stage "tool",
cpu_seconds) when the round had tools; each node depends on the one before it.
Node.attempt_index is the round's 0-based position in its turn: it is
observable (the scheduler sees how far along a turn is), and it is how the
online estimator knows how many rounds have completed.

BLOCK KEYS. Blocks of the engine's block size (EngineConfig.block_tokens, 16)
cover the round's clipped input (prefix + append). Key i is a hash chain seeded
by (model, session_id, epoch), exactly workload.build_block_chain with one
segment, so the model is folded into the seed as on MAARA and two rounds of the
same session and epoch share their leading blocks, up to the shorter prompt. A
new epoch (a compaction or reset, found by the builder) changes the seed, so a
reset context cannot falsely hit.

ORACLE FIELDS (expected_seconds, critical_path_remaining, critical_path_total)
are computed under CostModel.from_profile("gpu_5060ti_public") at batch 1,
cold (no cache hits). All public runs use that profile (DECISIONS 37), and at
batch 1 the batching coefficient c does not enter, so one cost model serves
every c the maps sweep. Reuse gaps come from trace_loader's annotation helpers.

ARRIVALS. As load_trace: seeded Poisson re-timing, random.Random(seed), the
first task at 0 and clock += expovariate(rate) after each. arrival_rate None
(which load_trace reads as "real timestamps") is defined here as SEQUENTIAL:
task i arrives after task i-1's batch-1 critical path x 1.25 + 1 s, so turns do
not overlap; the plumbing check uses it.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
from typing import Optional

from simulator.trace_loader import _annotate_block_reuse, _annotate_measured
from simulator.workload import (CODER_MODEL, CostModel, Kind, Node, Task,
                                build_block_chain)
from simulator.workload_sources import register_source

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO, "data", "external", "tracelab")
PROFILE = "gpu_5060ti_public"
STAGE_LLM, STAGE_TOOL = "agent_round", "tool"
SEQUENTIAL_MARGIN = (1.25, 1.0)        # x critical path, + seconds

SOURCES = {"tracelab_dev": "dev.jsonl.gz", "tracelab_sealed": "sealed.jsonl.gz",
           "tracelab_sealed2": "sealed2.jsonl.gz"}      # DECISIONS 38


def read_derived(path):
    """-> (header dict, [turn rows]) from a derived .jsonl.gz."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())["header"]
        rows = [json.loads(line) for line in f if line.strip()]
    return header, rows


def _block_tokens():
    from simulator.engine import EngineConfig
    return EngineConfig().block_tokens


def turn_to_task(row, task_id, cost, block_tokens=16) -> Task:
    nodes = []

    def add(**kw):
        n = Node(node_id=len(nodes), task_id=task_id,
                 deps=[len(nodes) - 1] if nodes else [], **kw)
        nodes.append(n)
        return n

    for j in range(len(row["epoch"])):
        pre, app = row["prefix"][j], row["append"][j]
        if pre + app <= 0:
            app = 1                      # an empty prompt has no blocks to run on
        ident = f"{row['session_id']}|e{row['epoch'][j]}"
        n = add(stage=STAGE_LLM, kind=Kind.LLM, model=CODER_MODEL,
                prefix_key=ident if pre > 0 else None,
                prefix_tokens=pre, suffix_tokens=app,
                output_tokens=max(1, row["output"][j]),
                attempt_index=j)
        n.block_keys = build_block_chain(
            CODER_MODEL, [(ident, math.ceil((pre + app) / block_tokens))])
        n.expected_seconds = (cost.prefill_seconds(n.model, n.prompt_tokens)
                              + n.output_tokens * cost.decode_iteration_seconds(n.model, 1))
        if row["tool_s"][j] is not None:
            t = add(stage=STAGE_TOOL, kind=Kind.CPU, cpu_seconds=float(row["tool_s"][j]),
                    attempt_index=j)
            t.expected_seconds = t.cpu_seconds
    return Task(task_id=task_id, arrival_time=0.0, nodes=nodes, n_replans=0,
                n_attempts=0, n_files=1, succeeded=True)


def build_tasks(rows, seed=1, arrival_rate: Optional[float] = None, cost=None,
                block_tokens=16) -> list:
    cost = cost or CostModel.from_profile(PROFILE)
    tasks = [turn_to_task(r, i, cost, block_tokens) for i, r in enumerate(rows)]
    for t in tasks:
        _annotate_measured(t)            # critical paths + within-task reuse edges
    rng = random.Random(seed)
    clock = 0.0
    for t in tasks:
        t.arrival_time = clock
        if arrival_rate:
            clock += rng.expovariate(arrival_rate)
        else:
            clock += SEQUENTIAL_MARGIN[0] * t.critical_path_total + SEQUENTIAL_MARGIN[1]
    _annotate_block_reuse(tasks)         # cross-turn gaps need the arrival clock
    return tasks


def load_tracelab(name, seed, arrival_rate):
    """Registry loader. The sealed gate has already run in resolve_workload."""
    _, rows = read_derived(os.path.join(DATA_DIR, SOURCES[name]))
    return build_tasks(rows, seed, arrival_rate, block_tokens=_block_tokens())


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def load_dev_constants(path=None) -> dict:
    with open(path or os.path.join(DATA_DIR, "dev_constants.json")) as f:
        return json.load(f)


class ChainEstimator:
    """
    Online estimator for chain-of-rounds turns (DECISIONS 37).

    Knows only the dev-split constants (frozen with the policies and reused
    unchanged on sealed) and what it has seen complete:
      * remaining work of a turn that has completed k rounds is the mean
        residual life E[R - k | R > k] rounds (R = rounds per turn, empirical
        dev distribution; 1 if no dev turn has R > k) times the mean round cost
        (prefill + decode at the dev means, under the run's cost model, batch 1)
        plus the mean tool time per round;
      * output length is a running mean of observed completions, cold-started
        from the dev mean.

    CONVENTIONS. When the request for round j is queued, k = j rounds are
    complete and R > j is known, so E[R-j | R>j] counts this round too.
    est_remaining_s (which, as on MAARA, EXCLUDES the request's own service) is
    therefore the residual rounds beyond this one, E[R-j | R>j] - 1, each
    costing round + tool, plus this round's own tool time (the mean; whether
    this round has tools is not known online). estimate_task_total is E[R]
    rounds of the same cost; it exists because the engine asks for it at
    start-up, and the "estimate" deadline basis is unsupported on this source.
    Never reads Node oracle fields.
    """

    is_oracle = False
    name = "online"
    attempt_weights = None            # MAARA retry law; printed by provenance only

    def __init__(self, cost: CostModel, tok=None, cpu=None, constants=None,
                 noise_sigma: float = 0.0, seed: int = 0):
        c = constants or load_dev_constants()
        self._constants = c
        self.cost, self.tok, self.cpu = cost, tok, cpu
        self.noise_sigma = noise_sigma
        self._rng = random.Random(seed)
        self._noise = {}
        self.r_counts = {int(k): v for k, v in c["rounds_per_turn_counts"].items()}
        self.mean_prefix = c["mean_prefix_tokens"]
        self.mean_append = c["mean_append_tokens"]
        self.mean_output = c["mean_output_tokens"]
        self.mean_tool = c["mean_tool_s_per_round"]
        self._out_sum = 0.0
        self._out_n = 0

    # -- learning ---------------------------------------------------------

    def observe(self, stage: str, attempt_index: int, output_tokens: int) -> None:
        self._out_sum += output_tokens
        self._out_n += 1

    def predict_output(self, stage: str, attempt_index: int = None,
                       true_output: int = None) -> float:
        """`true_output` is accepted and ignored; only the oracle uses it."""
        return self._out_sum / self._out_n if self._out_n else float(self.mean_output)

    def noise_factor(self, req_id) -> float:
        if self.noise_sigma <= 0 or req_id is None:
            return 1.0
        if req_id not in self._noise:
            self._noise[req_id] = math.exp(self._rng.gauss(0.0, self.noise_sigma))
        return self._noise[req_id]

    # -- derived times ----------------------------------------------------

    def mean_residual_rounds(self, k: int) -> float:
        """E[R - k | R > k] over the dev distribution; 1.0 if no dev turn has R > k."""
        num = den = 0
        for r, n in self.r_counts.items():
            if r > k:
                num += (r - k) * n
                den += n
        return num / den if den else 1.0

    def mean_round_cost(self, model: str = CODER_MODEL) -> float:
        """Prefill + decode at the dev means, batch 1, cold."""
        return (self.cost.prefill_seconds(model, self.mean_prefix + self.mean_append)
                + self.mean_output * self.cost.decode_iteration_seconds(model, 1))

    def service_s(self, model: str, prompt_tokens: int, stage: str,
                  attempt_index: int = None, true_output: int = None,
                  req_id=None) -> float:
        out = self.predict_output(stage, attempt_index, true_output)
        base = (self.cost.prefill_seconds(model, prompt_tokens)
                + out * self.cost.decode_iteration_seconds(model, 1))
        return base * self.noise_factor(req_id)

    def remaining_s(self, stage: str, attempt_index: int, model: str,
                    req_id=None, node=None) -> float:
        j = attempt_index or 0
        further = max(0.0, self.mean_residual_rounds(j) - 1.0)
        est = self.mean_tool + further * (self.mean_round_cost(model) + self.mean_tool)
        return est * self.noise_factor(req_id)

    def gittins_prior(self, cost):
        """The Gittins prior from the frozen dev constants (DECISIONS 38)."""
        from simulator.gittins import tracelab_prior
        return tracelab_prior(cost, self._constants)

    def estimate_task_total(self, model: str = CODER_MODEL) -> float:
        n = sum(self.r_counts.values())
        mean_r = sum(r * c for r, c in self.r_counts.items()) / n
        return mean_r * (self.mean_round_cost(model) + self.mean_tool)

    def _typical_prompt(self, stage: str) -> int:
        return int(self.mean_prefix + self.mean_append)


class ChainOracleEstimator(ChainEstimator):
    """
    Upper-bound reference only (LSF-oracle). Reads the recorded rounds left in
    the turn through the node's critical_path_remaining, the counterpart of the
    MAARA oracle's true remaining path, and true output lengths. Not deployable.
    """

    is_oracle = True
    name = "oracle"

    def predict_output(self, stage: str, attempt_index: int = None,
                       true_output: int = None) -> float:
        if true_output is None:
            return super().predict_output(stage, attempt_index)
        return float(true_output)

    def remaining_s(self, stage: str, attempt_index: int, model: str,
                    req_id=None, node=None) -> float:
        if node is None:
            return super().remaining_s(stage, attempt_index, model, req_id=req_id)
        return max(0.0, node.critical_path_remaining - node.expected_seconds)


def _online(cost, tok, cpu):
    return ChainEstimator(cost, tok, cpu)


def _oracle(cost, tok, cpu):
    return ChainOracleEstimator(cost, tok, cpu)


def register():
    for name in SOURCES:
        register_source(name, load_tracelab, sealed=name.startswith("tracelab_sealed"),
                        estimator_factory=_online, oracle_factory=_oracle,
                        supports_deadline_basis=("task",),
                        excluded_policies=("hint-priority",))


register()
