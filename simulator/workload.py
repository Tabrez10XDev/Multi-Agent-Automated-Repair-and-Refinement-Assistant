"""
Cost model + workflow-DAG trace generator.

This module answers two questions for the engine:
  1. How long does a given piece of work take on the GPU, and how much KV does
     it occupy?  -> CostModel
  2. What requests arrive, when, in what dependency order, and which of them
     share a reusable prefix?  -> generate_trace()

The DAG mirrors agents/ + orchestrator.py in this repo. It is not a generic
synthetic trace; stage-by-stage provenance is noted inline.

CALIBRATION
-----------
Numbers marked PLACEHOLDER are guesses and must be replaced. Run
    python -m probes.calibrate
to produce simulator/measured.json; this module loads it automatically and overrides
the placeholders with real measurements. `CostModel.provenance()` reports which
values came from measurement and which are still assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEASURED_PATH = os.path.join(_REPO, "data", "calibration", "measured.json")


def _load_measured() -> dict:
    try:
        with open(_MEASURED_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


MEASURED = _load_measured()


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

CODER_MODEL = "qwen2.5-coder:7b"
SENTINEL_MODEL = "llama3.2"


@dataclass(frozen=True)
class ModelRates:
    """
    Per-model serving constants.

    prefill_rate / decode_rate are single-stream tokens/sec. They set the
    ski-rental break-even: prefill is the "buy" price (what you pay to recompute
    a prefix you dropped), decode is the clock the rent runs on (what a blocked
    request is losing while it waits).

    kv_bytes_per_token for qwen2.5-coder:7b is now CONFIRMED against Ollama's
    own server log. Ollama 0.33.3 reports each saved prompt-cache state as
    "3791 tokens, 207.364 MiB", i.e. 207.364 x 1048576 / 3791 = 57,356 B/token
    against the 57,344 derived below -- a 0.02% difference, which is rounding
    in the logged MiB figure. llama3.2's value remains derived; it needs the
    same check against a log line for one of its states.

    Derived from published architecture configs:
        qwen2.5-coder:7b  28 layers, 4 KV heads (GQA), head_dim 128, fp16
                          -> 2 * 28 * 4 * 128 * 2 = 57344 B/token
        llama3.2:3b       28 layers, 8 KV heads, head_dim 128, fp16
                          -> 2 * 28 * 8 * 128 * 2 = 114688 B/token
    The 3B model costs MORE per token than the 7B one because qwen2.5-coder uses
    a more aggressive GQA ratio. That is not a typo, and it is exactly why the
    heterogeneous case is interesting: the cheap model is the memory hog.
    """
    name: str
    prefill_rate: float
    decode_rate: float
    kv_bytes_per_token: int
    measured: bool = False


_DEFAULT_RATES = {
    # MAC / METAL PLACEHOLDERS - measured by the author on Ollama + Metal.
    # Replace with CUDA measurements before publishing any absolute numbers.
    CODER_MODEL: ModelRates(CODER_MODEL, prefill_rate=123.0, decode_rate=29.0,
                            kv_bytes_per_token=57344, measured=False),
    # PLACEHOLDER - never measured. Scaled from the qwen numbers by rough
    # parameter-count ratio. probes/calibrate.py fills these in for real.
    SENTINEL_MODEL: ModelRates(SENTINEL_MODEL, prefill_rate=260.0, decode_rate=62.0,
                               kv_bytes_per_token=114688, measured=False),
}


def _rates_from_measured() -> dict:
    rates = dict(_DEFAULT_RATES)
    for name, m in MEASURED.get("rates", {}).items():
        key = name if name in rates else (SENTINEL_MODEL if name.startswith("llama") else name)
        if key in rates:
            rates[key] = replace(
                rates[key],
                prefill_rate=m["prefill_rate"],
                decode_rate=m["decode_rate"],
                measured=True,
            )
    return rates


@dataclass
class CostModel:
    """
    Single swappable object holding every GPU-side constant.

    Deliberately has NO memory-price knob. Per the design, rent is emergent: the
    engine charges it by observing that a retained entry kept a request waiting.
    Adding a price here would let the retention policy cheat by reading it.
    """
    rates: dict = field(default_factory=_rates_from_measured)
    homogeneous: bool = False

    # Batching efficiency. decode_rate is a single-stream number; a real engine
    # decodes a batch of n in much less than n times that. Iteration time is
    #     (1/decode_rate) * (1 + (n-1) * batch_decode_coeff)
    # so 0.0 = perfectly free batching, 1.0 = no batching benefit at all.
    # PLACEHOLDER: 0.15 is a plausible memory-bound-decode value, NOT measured.
    # This is the single most load-bearing unmeasured constant in the sim,
    # because it decides how much throughput a blocked request actually forgoes.
    # ablate.py should sweep it.
    batch_decode_coeff: float = 0.15

    # Fixed per-iteration engine overhead (scheduling, sampling), seconds.
    iteration_overhead_s: float = 0.001

    def rates_for(self, model: str) -> ModelRates:
        if self.homogeneous:
            return self.rates[CODER_MODEL]
        return self.rates[model]

    def prefill_seconds(self, model: str, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        return tokens / self.rates_for(model).prefill_rate

    def decode_iteration_seconds(self, model: str, batch_size: int) -> float:
        """Wall-clock for one decode step that advances `batch_size` requests by one token."""
        if batch_size <= 0:
            return 0.0
        r = self.rates_for(model)
        return (1.0 / r.decode_rate) * (1.0 + (batch_size - 1) * self.batch_decode_coeff)

    def kv_bytes(self, model: str, tokens: int) -> int:
        return tokens * self.rates_for(model).kv_bytes_per_token

    def recompute_seconds(self, model: str, tokens: int) -> float:
        """
        The ski-rental "buy" price: what it costs to rebuild a prefix you evicted.
        The engine compares accrued rent against this; it does not get a price.
        """
        return self.prefill_seconds(model, tokens)

    def provenance(self) -> str:
        lines = ["CostModel provenance:"]
        for name, r in self.rates.items():
            tag = "MEASURED" if r.measured else "PLACEHOLDER"
            # KV bytes/token has its own provenance: qwen's is confirmed
            # against an Ollama 0.33.3 prompt-cache log line ("3791 tokens,
            # 207.364 MiB" -> 57,356 B/token vs 57,344 derived, 0.02% apart).
            # llama3.2's is still derived from published config only.
            kv_tag = "MEASURED from ollama log" if name == CODER_MODEL else "DERIVED"
            lines.append(f"  {name:<20} prefill={r.prefill_rate:>7.1f} tok/s  "
                         f"decode={r.decode_rate:>6.1f} tok/s  "
                         f"kv={r.kv_bytes_per_token/1024:>5.0f} KiB/tok  "
                         f"[rates {tag}; kv {kv_tag}]")
        lines.append(f"  batch_decode_coeff   {self.batch_decode_coeff}  [PLACEHOLDER - sweep this]")
        lines.append(f"  homogeneous          {self.homogeneous}")
        # Applies whenever a trace-derived workload is used (simulator/trace_loader.py).
        # Ollama templates prompts server-side, so a prefix recovered from logged
        # message strings is an estimate, not a measurement. See tracing.py,
        # "PROMPT SIDECAR", for the three sources of error.
        lines.append("  trace prefix sizes   longest-common-prefix on raw prompts, "
                     "chars->tokens by per-call ratio  [APPROXIMATE]")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage token sizes
# --------------------------------------------------------------------------

@dataclass
class StageTokens:
    """
    Token sizes per pipeline stage, in one place so they are easy to sweep.

    stable_prefix_tokens is THE number that sets the scale of the whole
    memory-vs-recompute tradeoff. Everything else is secondary.
    """
    # generate_fix - from maara/agents/coder.py:113-131
    stable_prefix_tokens: int = 2000        # PLACEHOLDER until measured
    fix_suffix_tokens: int = 350            # PLACEHOLDER: source block, first attempt
    fix_suffix_retry_tokens: int = 700      # PLACEHOLDER: source + prev fix + diagnosis
    fix_output_tokens: int = 400            # PLACEHOLDER: coder returns the whole file

    # parse_issue - from maara/agents/issue_parser.py:16-29
    # The replan call is parse_issue(issue_text + feedback), so the preamble and
    # the original issue text are a genuine shared prefix across replans.
    issue_prefix_tokens: int = 320          # PLACEHOLDER
    issue_suffix_tokens: int = 140          # PLACEHOLDER: file list + JSON instructions
    issue_output_tokens: int = 180          # PLACEHOLDER

    # evaluate_plan - from maara/agents/evaluator.py:24-44. Different model, so its KV
    # can never be shared with the coder's even though the text overlaps.
    eval_prompt_tokens: int = 1800          # PLACEHOLDER
    eval_output_tokens: int = 200           # PLACEHOLDER

    # diagnose - from maara/agents/debugger.py:22-44. original_code is re-read from
    # disk AFTER apply_fix overwrote it, so it differs every attempt: no
    # meaningful reusable prefix here.
    diagnose_prompt_tokens: int = 1100      # PLACEHOLDER
    diagnose_output_tokens: int = 250       # PLACEHOLDER

    measured_fields: tuple = ()

    @classmethod
    def from_measured(cls) -> "StageTokens":
        st = cls()
        ps = MEASURED.get("prompt_sizes")
        if not ps:
            return st
        st.stable_prefix_tokens = ps["stable_prefix_tokens"]
        st.fix_suffix_tokens = ps["variable_suffix_tokens"]
        st.fix_suffix_retry_tokens = ps["variable_suffix_retry_tokens"]
        st.measured_fields = ("stable_prefix_tokens", "fix_suffix_tokens",
                              "fix_suffix_retry_tokens")
        return st


@dataclass
class CpuStageSeconds:
    """Non-GPU stages. They hold no KV but define the reuse gap."""
    pytest_seconds: float = 2.0             # PLACEHOLDER
    rag_index_seconds: float = 3.0          # PLACEHOLDER
    measured: bool = False

    @classmethod
    def from_measured(cls) -> "CpuStageSeconds":
        c = MEASURED.get("cpu_stages")
        if not c:
            return cls()
        return cls(
            pytest_seconds=c.get("pytest_seconds") or 2.0,
            rag_index_seconds=c.get("rag_index_seconds") or 3.0,
            measured=True,
        )


# --------------------------------------------------------------------------
# Trace structures
# --------------------------------------------------------------------------

class Kind(Enum):
    LLM = "llm"
    CPU = "cpu"


# tracing.py labels calls by agent module name; the simulator labels nodes by
# pipeline stage. Phase 2's loader uses this to translate a real trace into
# Nodes. Kept here so the two vocabularies are reconciled in exactly one place.
ROLE_TO_STAGE = {
    "issue_parser": "parse_issue",
    "retriever": "navigate",
    "evaluator": "evaluate_plan",
    "coder": "generate_fix",
    "debugger": "diagnose",
}
STAGE_TO_ROLE = {v: k for k, v in ROLE_TO_STAGE.items()}


@dataclass
class Node:
    """One unit of work in a task's realized DAG."""
    node_id: int
    task_id: int
    stage: str
    kind: Kind
    deps: list

    # LLM fields
    model: str = ""
    prefix_key: Optional[str] = None   # None = nothing reusable
    prefix_tokens: int = 0
    suffix_tokens: int = 0
    output_tokens: int = 0

    # CPU fields
    cpu_seconds: float = 0.0

    # Position in the two loops. Names match tracing.py's row schema so the
    # Phase 2 loader can copy them across without translation.
    #
    # Semantics mirror tracing.py exactly, including one quirk worth knowing:
    # set_replan() is never reset, so once the Sentinel loop has run, every
    # later node in the task carries the FINAL replan number rather than
    # reverting to 0. A synthetic trace that zeroed it would not match a real
    # one, and Phase 2 would see a spurious difference.
    attempt_index: int = 0
    replan_index: int = 0

    # vLLM-style prefix cache keys: one per KV block, each hashing the chain of
    # all blocks up to and including it. A lookup matches the longest run of
    # leading blocks already cached, so a hit is PARTIAL -- matched blocks skip
    # prefill, the rest are computed. Model is folded into the chain seed
    # because a KV cache belongs to one model instance.
    block_keys: list = field(default_factory=list)

    # Filled in by the generator, consumed by Phase-2 schedulers.
    # ORACLE FIELDS - an online policy must not read these.
    expected_seconds: float = 0.0
    critical_path_remaining: float = 0.0
    next_reuse_node: Optional[int] = None
    next_reuse_gap_s: Optional[float] = None

    @property
    def prompt_tokens(self) -> int:
        return self.prefix_tokens + self.suffix_tokens


@dataclass
class Task:
    task_id: int
    arrival_time: float
    nodes: list
    n_replans: int
    n_attempts: int
    n_files: int
    succeeded: bool
    critical_path_total: float = 0.0


@dataclass
class WorkloadConfig:
    n_tasks: int = 40
    arrival_rate: float = 0.35          # tasks/sec, Poisson
    seed: int = 7

    # Empirical, from data/outcomes/results.jsonl (121 runs): attempts {1:74, 2:13, 3:33}
    attempt_weights: tuple = (74, 13, 33)
    max_attempts: int = 3

    # MAX_REPLAN in orchestrator.py:71. Rejection probability is a PLACEHOLDER;
    # results.jsonl records evaluator_approved_on_attempt but it is null for the
    # single_agent / no_sentinel arms, so it is not cleanly recoverable yet.
    max_replan: int = 2
    replan_reject_prob: float = 0.30

    # generate_fix loops over relevant_code (agents/coder.py:95), so one attempt
    # can be several concurrent requests. This is the intra-task parallelism.
    files_per_task_weights: tuple = (70, 20, 10)   # 1, 2, or 3 files

    # Per-task size jitter: real issues vary a lot in context size.
    size_jitter_sigma: float = 0.35

    use_sentinel: bool = True
    use_rag: bool = True


# --------------------------------------------------------------------------
# Trace generation
# --------------------------------------------------------------------------

DEFAULT_BLOCK_TOKENS = 16


def build_block_chain(model: str, segments) -> list:
    """
    Hash chain over KV blocks.

    `segments` is [(identity, n_blocks), ...]. Two prompts sharing an identity
    for their leading segments produce identical leading keys, and diverge for
    good at the first differing block -- which is what a real prefix cache does.

    Only WHOLE shared blocks are given the shared identity. A block straddling
    the boundary between shared and private content holds both, so it cannot
    match, and flooring here reproduces that loss rather than hiding it.
    """
    h = hashlib.sha1(model.encode()).hexdigest()[:12]
    chain = []
    for ident, n in segments:
        for i in range(int(n)):
            h = hashlib.sha1(f"{h}|{ident}|{i}".encode()).hexdigest()[:12]
            chain.append(h)
    return chain


def block_chain_from_text(model: str, text: str, chars_per_block: int) -> list:
    """
    Chain built from prompt text, for traces where only the text is known.

    Chunk width is fixed rather than derived per call: two calls with slightly
    different chars-per-token would otherwise cut at different offsets and
    diverge on block 1 even with an identical prefix.
    """
    h = hashlib.sha1(model.encode()).hexdigest()[:12]
    chain = []
    for i in range(0, len(text), max(1, chars_per_block)):
        h = hashlib.sha1((h + text[i:i + chars_per_block]).encode()).hexdigest()[:12]
        chain.append(h)
    return chain


def _weighted_choice(rng: random.Random, weights) -> int:
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return i
    return len(weights) - 1


def _build_task(task_id: int, arrival: float, rng: random.Random,
                cfg: WorkloadConfig, tok: StageTokens, cpu: CpuStageSeconds) -> Task:
    """
    Unroll one realized execution of orchestrator.run() into a DAG.

    The loop trip counts are data-dependent in the real system, so we sample them
    up front and emit the realized path. That makes the trace replayable and lets
    Phase 2 compute Belady/OPT offline -- but it also means the trace object
    literally contains the future. Online policies must only ever see a node once
    it has been released by the engine.
    """
    nodes = []
    scale = math.exp(rng.gauss(0, cfg.size_jitter_sigma))

    def sz(base: int) -> int:
        return max(16, int(base * scale))

    n_files = _weighted_choice(rng, cfg.files_per_task_weights) + 1
    n_attempts = _weighted_choice(rng, cfg.attempt_weights) + 1
    succeeded = n_attempts < cfg.max_attempts or rng.random() < 0.5

    # Loop position, stamped onto every node as it is created. Mirrors how
    # tracing.py's contextvars behave: both are "current position", and neither
    # is reset when a loop exits.
    ctx = {"attempt": 0, "replan": 0}

    def add(stage, kind, deps, **kw) -> int:
        nid = len(nodes)
        n = Node(node_id=nid, task_id=task_id, stage=stage, kind=kind,
                 deps=list(deps), attempt_index=ctx["attempt"],
                 replan_index=ctx["replan"], **kw)
        if n.kind is Kind.LLM:
            bt = DEFAULT_BLOCK_TOKENS
            shared_blocks = n.prefix_tokens // bt          # whole blocks only
            total_blocks = -(-n.prompt_tokens // bt)
            ident = n.prefix_key or f"t{task_id}:n{nid}:head"
            n.block_keys = build_block_chain(n.model, [
                (ident, shared_blocks),
                (f"t{task_id}:n{nid}:tail", max(0, total_blocks - shared_blocks)),
            ])
        nodes.append(n)
        return nid

    # Model is part of the key: a KV cache belongs to one model instance, so
    # two calls can only share an entry if they ran on the same model.
    issue_key = f"t{task_id}:{CODER_MODEL}:issue"
    stable_key = f"t{task_id}:{CODER_MODEL}:stable_ctx"

    # Step 1: parse_issue
    prev = add("parse_issue", Kind.LLM, [], model=CODER_MODEL,
               prefix_key=issue_key, prefix_tokens=sz(tok.issue_prefix_tokens),
               suffix_tokens=sz(tok.issue_suffix_tokens),
               output_tokens=sz(tok.issue_output_tokens))

    # Step 2: navigate (CPU - ChromaDB + nomic-embed-text, no generation)
    if cfg.use_rag:
        prev = add("navigate", Kind.CPU, [prev], cpu_seconds=cpu.rag_index_seconds)

    # Step 2.5: Sentinel gate, with the replan loop from orchestrator.py:71-83
    n_replans = 0
    if cfg.use_sentinel:
        for _replan_i in range(cfg.max_replan):
            # orchestrator.py calls tracing.set_replan(replan + 1) here, so the
            # first Sentinel evaluation is replan_index 1, not 0.
            ctx["replan"] = _replan_i + 1
            prev = add("evaluate_plan", Kind.LLM, [prev], model=SENTINEL_MODEL,
                       prefix_key=None, suffix_tokens=sz(tok.eval_prompt_tokens),
                       output_tokens=sz(tok.eval_output_tokens))
            if rng.random() >= cfg.replan_reject_prob:
                break
            n_replans += 1
            # Re-parse with feedback appended. Shares the issue_text prefix.
            prev = add("parse_issue", Kind.LLM, [prev], model=CODER_MODEL,
                       prefix_key=issue_key, prefix_tokens=sz(tok.issue_prefix_tokens),
                       suffix_tokens=sz(tok.issue_suffix_tokens) + 60,
                       output_tokens=sz(tok.issue_output_tokens))

    # Retry loop: generate_fix (fan-out over files) -> apply_and_test -> diagnose
    for attempt in range(1, n_attempts + 1):
        ctx["attempt"] = attempt
        first = attempt == 1
        suffix = tok.fix_suffix_tokens if first else tok.fix_suffix_retry_tokens
        fix_nodes = [
            add("generate_fix", Kind.LLM, [prev], model=CODER_MODEL,
                prefix_key=stable_key, prefix_tokens=sz(tok.stable_prefix_tokens),
                suffix_tokens=sz(suffix), output_tokens=sz(tok.fix_output_tokens))
            for _ in range(n_files)
        ]
        prev = add("apply_and_test", Kind.CPU, fix_nodes, cpu_seconds=cpu.pytest_seconds)

        last = attempt == n_attempts
        if last and succeeded:
            break
        # orchestrator.py:142-161 runs diagnose even on the final failing attempt.
        prev = add("diagnose", Kind.LLM, [prev], model=CODER_MODEL,
                   prefix_key=None, suffix_tokens=sz(tok.diagnose_prompt_tokens),
                   output_tokens=sz(tok.diagnose_output_tokens))

    return Task(task_id=task_id, arrival_time=arrival, nodes=nodes,
                n_replans=n_replans, n_attempts=n_attempts, n_files=n_files,
                succeeded=succeeded)


def _annotate(task: Task, cm: CostModel) -> None:
    """Expected durations, critical path to sink, and reuse edges."""
    for n in task.nodes:
        if n.kind is Kind.CPU:
            n.expected_seconds = n.cpu_seconds
        else:
            n.expected_seconds = (cm.prefill_seconds(n.model, n.prompt_tokens)
                                  + n.output_tokens * cm.decode_iteration_seconds(n.model, 1))

    # Longest path to sink, computed backwards. Nodes are in topological order
    # by construction, so a reverse sweep is sufficient.
    succs = {n.node_id: [] for n in task.nodes}
    for n in task.nodes:
        for d in n.deps:
            succs[d].append(n.node_id)

    by_id = {n.node_id: n for n in task.nodes}
    for n in reversed(task.nodes):
        downstream = [by_id[s].critical_path_remaining for s in succs[n.node_id]]
        n.critical_path_remaining = n.expected_seconds + (max(downstream) if downstream else 0.0)

    task.critical_path_total = max(n.critical_path_remaining for n in task.nodes)

    # Reuse edges: for each prefix key, link each node to the next node that
    # wants the same prefix, and record the gap in expected time. This is the
    # ORACLE the learning-augmented ski-rental variant gets a noisy version of.
    earliest = {}
    acc = 0.0
    for n in task.nodes:
        earliest[n.node_id] = acc
        acc += n.expected_seconds

    last_for_key = {}
    for n in task.nodes:
        if n.prefix_key is None:
            continue
        if n.prefix_key in last_for_key:
            prior = last_for_key[n.prefix_key]
            prior.next_reuse_node = n.node_id
            prior.next_reuse_gap_s = earliest[n.node_id] - earliest[prior.node_id]
        last_for_key[n.prefix_key] = n


def generate_trace(cfg: WorkloadConfig = None, cm: CostModel = None,
                   tok: StageTokens = None, cpu: CpuStageSeconds = None) -> list:
    """Produce a list of Tasks with Poisson arrivals and fully annotated DAGs."""
    cfg = cfg or WorkloadConfig()
    cm = cm or CostModel()
    tok = tok or StageTokens.from_measured()
    cpu = cpu or CpuStageSeconds.from_measured()

    rng = random.Random(cfg.seed)
    tasks = []
    t = 0.0
    for i in range(cfg.n_tasks):
        t += rng.expovariate(cfg.arrival_rate)
        task = _build_task(i, t, rng, cfg, tok, cpu)
        _annotate(task, cm)
        tasks.append(task)
    return tasks


def describe_trace(tasks: list) -> str:
    llm = [n for t in tasks for n in t.nodes if n.kind is Kind.LLM]
    cpu_nodes = [n for t in tasks for n in t.nodes if n.kind is Kind.CPU]
    reuse = [n for n in llm if n.next_reuse_gap_s is not None]
    gaps = sorted(n.next_reuse_gap_s for n in reuse)

    stages = {}
    for n in llm:
        stages[n.stage] = stages.get(n.stage, 0) + 1

    out = [
        f"Trace: {len(tasks)} tasks, {len(llm)} LLM requests, {len(cpu_nodes)} CPU stages",
        f"  span: {tasks[-1].arrival_time:.1f}s of arrivals",
        f"  stages: {stages}",
        f"  attempts: {dict(sorted((t.n_attempts, sum(1 for x in tasks if x.n_attempts == t.n_attempts)) for t in tasks))}",
        f"  reuse edges: {len(reuse)}",
    ]
    if gaps:
        out.append(f"  reuse gap p50={gaps[len(gaps)//2]:.1f}s  "
                   f"min={gaps[0]:.1f}s  max={gaps[-1]:.1f}s")
    return "\n".join(out)


if __name__ == "__main__":
    cm = CostModel()
    print(cm.provenance())
    print()
    tok = StageTokens.from_measured()
    print(f"stable_prefix_tokens = {tok.stable_prefix_tokens} "
          f"({'MEASURED' if tok.measured_fields else 'PLACEHOLDER - run probes.calibrate'})")
    cpu = CpuStageSeconds.from_measured()
    print(f"pytest={cpu.pytest_seconds}s rag={cpu.rag_index_seconds}s "
          f"({'MEASURED' if cpu.measured else 'PLACEHOLDER'})")
    print()
    print(describe_trace(generate_trace(cm=cm, tok=tok, cpu=cpu)))
