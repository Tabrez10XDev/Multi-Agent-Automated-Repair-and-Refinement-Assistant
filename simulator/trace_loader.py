"""
Load a real MAARA trace into the simulator's Node DAG.

The output is the same shape generate_trace() produces -- a list of
workload.Task, each holding workload.Node objects -- so every policy and the
engine itself run unchanged on real or synthetic input.

TASK-RUN SEGMENTATION
---------------------
tracing.py does not record which experiment configuration a call belongs to.
It is recoverable anyway, in two steps.

1. Segment. The harness runs one task at a time in a single process, so one
   task run's calls are contiguous in `seq`. A new block starts wherever
   task_id changes. Verified on the committed trace: 55 blocks, zero parent
   edges crossing a boundary, zero dangling parents, seq contiguous 1..366.

2. Fingerprint. orchestrator.py gates mechanisms by config name, so each arm
   leaves a distinct signature:

       config              coder model        retriever  evaluator
       single_agent        qwen2.5-coder:7b   yes        yes
       homo_llama3         llama3.2           yes        yes
       hetero_no_rag       qwen2.5-coder:7b   NO         yes
       hetero_no_sentinel  llama3.2           yes        NO

   The four are uniquely separated by (coder model, retriever present,
   evaluator present). Every block in the committed trace has a coder call, so
   the fingerprint always resolves.

This is inference, not recorded fact. add_task_run_id() in tracing.py would
make it explicit -- see PROPOSAL at the bottom of this module.

PREFIX RECOVERY IS APPROXIMATE
------------------------------
Shared prefix size is estimated, not measured. See tracing.py "PROMPT SIDECAR"
for the three error sources: Ollama templates prompts server-side, chars->tokens
is a per-call linear ratio, and a character-level common prefix can end
mid-token. CostModel.provenance() carries the APPROXIMATE label.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Optional

from simulator.workload import (DEFAULT_BLOCK_TOKENS, ROLE_TO_STAGE, CostModel, Kind,
                          Node, Task, block_chain_from_text)

CODER_MODELS = {"qwen2.5-coder:7b", "llama3.2"}

# CPU-side stages hold no KV and generate no tokens.
_CPU_ROLES = {"retriever"}


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def _open_maybe_gz(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def load_rows(path):
    """Read a JSONL trace, gz or plain, tolerating a file still being written."""
    rows = []
    try:
        with _open_maybe_gz(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except EOFError:
        pass
    return rows


def default_prompts_path(main_path: str) -> str:
    if main_path.endswith(".jsonl.gz"):
        return main_path[:-len(".jsonl.gz")] + ".prompts.jsonl.gz"
    if main_path.endswith(".jsonl"):
        return main_path[:-len(".jsonl")] + ".prompts.jsonl"
    return main_path + ".prompts"


def render_prompt(messages) -> str:
    """
    Flatten a messages list into the text the model sees, in order.

    Concatenating contents without role markers is deliberate. The real token
    sequence begins with the system message, so two calls with different system
    prompts genuinely share no KV prefix; preserving order reproduces that.
    What it cannot reproduce is the chat template's own tokens, which is one of
    the reasons the result is labelled APPROXIMATE.
    """
    return "\n".join(m.get("content", "") for m in messages)


# --------------------------------------------------------------------------
# Segmentation and config reconstruction
# --------------------------------------------------------------------------

def segment_task_runs(rows):
    """
    Split rows into contiguous per-task-run blocks.

    Grouped by run_id first. `seq` is per-process and restarts at 1 in every
    trace file, so sorting the merged rows by seq alone would interleave two
    files and fabricate task-run boundaries. Runs are then ordered by their
    earliest t_start so merged output stays chronological.
    """
    by_run = {}
    for r in rows:
        by_run.setdefault(r.get("run_id"), []).append(r)

    def run_start(rs):
        return min((r.get("t_start") or 0.0) for r in rs)

    blocks = []
    for _, rs in sorted(by_run.items(), key=lambda kv: run_start(kv[1])):
        # Two processes can share a run_id: MAARA_RUN_ID pins the filename, so
        # an interrupted run resumed under the same id appends a SECOND seq
        # sequence starting again at 1. Sorting by seq then interleaves them
        # and fabricates a task-run boundary on every other row.
        #
        # Order chronologically -- t_start is comparable across processes --
        # and start a new process group wherever seq fails to advance.
        for r in sorted(rs, key=lambda r: (r.get("t_start") or 0.0, r["seq"])):
            new_process = blocks and r["seq"] <= blocks[-1][-1]["seq"] and \
                r["task_id"] == blocks[-1][-1]["task_id"]
            if (not blocks or r["task_id"] != blocks[-1][-1]["task_id"]
                    or new_process):
                blocks.append([r])
            else:
                blocks[-1].append(r)
    return blocks


def fingerprint_config(block) -> Optional[str]:
    """Recover the experiment arm from which mechanisms fired. None if unclear."""
    roles = {r["agent_role"] for r in block}
    coder = [r["model"] for r in block if r["agent_role"] == "coder"]
    if not coder:
        return None
    model = coder[0]
    has_rag = "retriever" in roles
    has_sentinel = "evaluator" in roles
    if model == "qwen2.5-coder:7b":
        return "single_agent" if has_rag else "hetero_no_rag"
    if model == "llama3.2":
        return "homo_llama3" if has_sentinel else "hetero_no_sentinel"
    return None


def validate_segmentation(blocks) -> dict:
    """Checks that make the inference defensible rather than assumed."""
    ids = [{r["call_id"] for r in b} for b in blocks]
    everything = set().union(*ids) if ids else set()
    crossing = dangling = 0
    for own, b in zip(ids, blocks):
        for r in b:
            for p in r.get("parent_call_ids", []):
                if p in own:
                    continue
                if p in everything:
                    crossing += 1
                else:
                    dangling += 1
    configs = {}
    for b in blocks:
        configs[fingerprint_config(b)] = configs.get(fingerprint_config(b), 0) + 1
    return {
        "task_runs": len(blocks),
        "distinct_tasks": len({b[0]["task_id"] for b in blocks}),
        "parent_edges_crossing_task_runs": crossing,
        "parent_edges_dangling": dangling,
        "unidentified_configs": sum(1 for b in blocks if fingerprint_config(b) is None),
        "runs_per_config": configs,
    }


# --------------------------------------------------------------------------
# Ollama prefix-cache detection
# --------------------------------------------------------------------------

def detect_cache_hits(rows, prompts_by_call=None, speedup=1.5, cost=None) -> dict:
    """
    Find Ollama prefix-cache hits from prefill DURATION, not token count.

    The earlier version inferred a hit from prompt_eval_count falling below the
    prompt's true size. sim/check_ollama_cache proved that assumption wrong:
    Ollama reports the FULL count even when llama.cpp reuses the whole cached
    prefix. Measured on /api/generate, an identical repeat returned in 0.040s
    against a cold 13.400s -- 335x faster -- with the count unchanged. The
    count-based detector was therefore blind, and its "zero hits" on both
    committed traces was an artefact.

    A call is counted as a hit when its prefill rate exceeds `speedup` times
    the model's calibrated uncached rate. Hidden cached tokens are estimated as

        prompt_tokens - prefill_seconds x uncached_rate

    i.e. the tokens that could not have been computed in the time observed.
    """
    cost = cost or CostModel()
    per_call, hits = [], []
    for r in rows:
        if r.get("kind") != "chat":
            continue
        pt = r.get("prompt_tokens") or 0
        secs = r.get("prompt_eval_duration") or 0.0
        model = r.get("model") or ""
        if pt <= 0 or secs <= 0 or model not in cost.rates:
            continue
        uncached = cost.rates[model].prefill_rate
        rate = pt / secs
        hidden = max(0, int(round(pt - secs * uncached)))
        rec = {"row": r, "rate": rate, "uncached": uncached,
               "ratio": rate / uncached, "hidden": hidden}
        per_call.append(rec)
        if rate > speedup * uncached:
            hits.append(rec)

    if not per_call:
        return {"checked": 0, "likely_cache_hits": 0}

    def tally(field):
        out = {}
        for h in hits:
            k = h["row"].get(field)
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    hidden_total = sum(h["hidden"] for h in hits)
    return {
        "method": "prefill duration vs calibrated uncached rate",
        "speedup_threshold": speedup,
        "checked": len(per_call),
        "likely_cache_hits": len(hits),
        "hit_fraction": len(hits) / len(per_call),
        "hidden_prompt_tokens": hidden_total,
        "reported_prompt_tokens": sum(r["row"].get("prompt_tokens", 0) for r in per_call),
        "prefill_seconds_saved": round(
            sum(h["hidden"] / h["uncached"] for h in hits), 2),
        "by_role": tally("agent_role"),
        "by_model": tally("model"),
        "median_ratio_all": round(statistics.median(r["ratio"] for r in per_call), 2),
        "median_ratio_hits": (round(statistics.median(h["ratio"] for h in hits), 2)
                              if hits else None),
    }


# --------------------------------------------------------------------------
# Shared-prefix recovery
# --------------------------------------------------------------------------

def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    # Binary search on the common length: prompts here run to tens of
    # thousands of characters and this is O(n^2) in call count.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass
class PrefixInfo:
    call_id: str
    shared_chars: int = 0
    shared_tokens: int = 0
    shares_with: Optional[str] = None
    chars_per_token: float = 0.0
    prefix_key: Optional[str] = None
    prompt_chars: int = 0
    model: str = ""


def compute_prefixes(block, prompts_by_call, run_key: str, same_model_only=True):
    """
    For each call, the longest common prefix against every EARLIER call in the
    same task run, any role.

    Cross-role is the point: the coder and the Sentinel both carry the issue
    text and the test files, so restricting to same-role comparisons would miss
    the reuse the retention policy is supposed to exploit.

    Conservative where prefix lengths differ: two calls are only given the same
    cache key when their shared prefix text is byte-identical, so a pair
    sharing 500 chars and another sharing 700 will not share an entry. That
    understates reuse rather than inventing it.
    """
    texts, out = [], {}
    for r in block:
        if r.get("kind") != "chat":
            continue
        msgs = prompts_by_call.get(r["call_id"])
        text = render_prompt(msgs) if msgs else ""
        pt = r.get("prompt_tokens") or 0
        model = r.get("model", "")
        info = PrefixInfo(call_id=r["call_id"], prompt_chars=len(text), model=model,
                          chars_per_token=(len(text) / pt) if pt else 0.0)

        best_len, best_id = 0, None
        for prev_id, prev_text, prev_model in texts:
            # A KV cache belongs to one model instance. A prefix computed by
            # qwen cannot serve a llama3.2 call however identical the text, so
            # cross-model matches are not reuse -- they are a measurement
            # error. same_model_only=False reproduces the earlier, wrong
            # figures for comparison.
            if same_model_only and prev_model != model:
                continue
            n = _lcp_len(text, prev_text)
            if n > best_len:
                best_len, best_id = n, prev_id

        if best_len > 0 and info.chars_per_token > 0:
            info.shared_chars = best_len
            info.shared_tokens = int(best_len / info.chars_per_token)
            info.shares_with = best_id
            digest = hashlib.sha1(text[:best_len].encode()).hexdigest()[:10]
            # Model in the key so the simulator can never match across models
            # even if two of them produce byte-identical prefixes.
            info.prefix_key = f"{run_key}:{model}:{digest}"

        out[r["call_id"]] = info
        texts.append((r["call_id"], text, model))
    return out


# --------------------------------------------------------------------------
# DAG construction
# --------------------------------------------------------------------------

def _annotate_measured(task: Task) -> None:
    """
    Critical path and reuse edges from MEASURED durations.

    workload._annotate() derives expected_seconds from CostModel; here the
    trace already contains what each call actually took, so using the model
    would discard the measurement.
    """
    succs = {n.node_id: [] for n in task.nodes}
    for n in task.nodes:
        for d in n.deps:
            succs[d].append(n.node_id)

    by_id = {n.node_id: n for n in task.nodes}
    for n in reversed(task.nodes):
        downstream = [by_id[s].critical_path_remaining for s in succs[n.node_id]]
        n.critical_path_remaining = n.expected_seconds + (max(downstream) if downstream else 0.0)
    task.critical_path_total = max((n.critical_path_remaining for n in task.nodes),
                                   default=0.0)

    earliest, acc = {}, 0.0
    for n in task.nodes:
        earliest[n.node_id] = acc
        acc += n.expected_seconds

    last = {}
    for n in task.nodes:
        if n.prefix_key is None:
            continue
        if n.prefix_key in last:
            prior = last[n.prefix_key]
            prior.next_reuse_node = n.node_id
            prior.next_reuse_gap_s = earliest[n.node_id] - earliest[prior.node_id]
        last[n.prefix_key] = n


@dataclass
class LoadedTrace:
    tasks: list
    validation: dict
    cache: dict
    prefixes: dict = field(default_factory=dict)
    blocks: list = field(default_factory=list)
    configs: list = field(default_factory=list)


# Named trace sets, so the two prompt layouts can be analysed side by side
# without either overwriting the other's figures.
TRACE_SETS = {
    "original": ["data/traces/full_zs_4cfg.jsonl.gz",
                 "data/traces/full_zs_4cfg_smoke5.jsonl.gz"],
    "shared_first": ["data/traces/full_zs_4cfg_sharedfirst.jsonl.gz"],
}
DEFAULT_TRACES = TRACE_SETS["original"]


def trace_set(name):
    """Absolute paths for a named set. Missing files are dropped, so a set can
    be analysed before every part of it has been collected."""
    import os as _os
    repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    paths = [_os.path.join(repo, p) for p in TRACE_SETS[name]]
    return [p for p in paths if _os.path.exists(p)]


def load_trace(main_path, prompts_path=None,
               arrival_rate: float = None, seed: int = 7,
               use_sidecar_prompt_size: bool = True) -> LoadedTrace:
    """
    Build simulator Tasks from a recorded trace.

    arrival_rate: the real run was strictly sequential -- one task at a time,
    inter-block gaps of ~0s -- so replaying real timestamps produces a workload
    with no concurrency and therefore no contention. Passing an arrival rate
    re-times arrivals as a Poisson process while keeping every measured
    duration and token count, which is how a single-stream trace is turned into
    a contended one. Default None keeps the real timing.

    use_sidecar_prompt_size: prefer the sidecar text length over
    prompt_eval_count when the latter looks like an Ollama cache hit. See
    detect_cache_hits.
    """
    # main_path may be one path or several. The committed dataset is split in
    # two: the 55 runs of full_zs_4cfg, plus the 5 single_agent runs the smoke
    # run had already completed, which the full run then resumed past.
    paths = [main_path] if isinstance(main_path, str) else list(main_path)
    if prompts_path is None:
        prompt_paths = [default_prompts_path(p) for p in paths]
    else:
        prompt_paths = [prompts_path] if isinstance(prompts_path, str) else list(prompts_path)

    rows = []
    for p in paths:
        rows.extend(load_rows(p))
    prompts_by_call = {}
    for pp in prompt_paths:
        if os.path.exists(pp):
            for p in load_rows(pp):
                prompts_by_call[p["call_id"]] = p.get("messages", [])

    blocks = segment_task_runs(rows)
    validation = validate_segmentation(blocks)
    cache = detect_cache_hits(rows, prompts_by_call)
    median_cpt = cache.get("median_chars_per_token") or 4.0

    tasks, all_prefixes, configs = [], {}, []
    import random as _random
    rng = _random.Random(seed)
    clock = 0.0

    for idx, block in enumerate(blocks):
        config = fingerprint_config(block)
        configs.append(config)
        run_key = f"r{idx}"
        prefixes = compute_prefixes(block, prompts_by_call, run_key)
        all_prefixes.update(prefixes)

        # Fixed chunk width so two calls with an identical text prefix cut at
        # the same offsets and therefore produce identical leading block keys.
        chars_per_block = max(1, int(DEFAULT_BLOCK_TOKENS * median_cpt))

        nodes, node_of_call = [], {}
        for r in block:
            role = r["agent_role"]
            stage = ROLE_TO_STAGE.get(role, role)
            dur = max(0.0, (r.get("t_end") or 0.0) - (r.get("t_start") or 0.0))

            if r.get("kind") == "embedding" or role in _CPU_ROLES:
                n = Node(node_id=len(nodes), task_id=idx, stage=stage, kind=Kind.CPU,
                         deps=[], cpu_seconds=dur,
                         attempt_index=r.get("attempt_index", 0),
                         replan_index=r.get("replan_index", 0))
                n.expected_seconds = dur
            else:
                info = prefixes.get(r["call_id"])
                reported = r.get("prompt_tokens") or 0
                total_tokens = reported
                if use_sidecar_prompt_size and info and info.prompt_chars:
                    # An Ollama cache hit makes prompt_eval_count count only the
                    # recomputed tail. The simulator needs the full prompt, so
                    # fall back to the text estimate when they disagree sharply.
                    est = int(info.prompt_chars / median_cpt)
                    if est > reported * 1.5:
                        total_tokens = est
                prefix_tokens = min(info.shared_tokens if info else 0, total_tokens)
                n = Node(node_id=len(nodes), task_id=idx, stage=stage, kind=Kind.LLM,
                         deps=[], model=r.get("model", ""),
                         prefix_key=info.prefix_key if info else None,
                         prefix_tokens=prefix_tokens,
                         suffix_tokens=max(0, total_tokens - prefix_tokens),
                         output_tokens=r.get("output_tokens") or 0,
                         attempt_index=r.get("attempt_index", 0),
                         replan_index=r.get("replan_index", 0))
                msgs = prompts_by_call.get(r["call_id"])
                text = render_prompt(msgs) if msgs else ""
                n.block_keys = block_chain_from_text(
                    n.model, text[:int(total_tokens * median_cpt)], chars_per_block)
                n.expected_seconds = dur
            nodes.append(n)
            node_of_call[r["call_id"]] = n.node_id

        # Dependencies from recorded parent edges. Calls with no parent depend
        # on the previous node so the DAG stays a connected chain -- the engine
        # releases a node only once its deps are done, and an orphan would be
        # admitted at time zero regardless of what it actually waited for.
        for r, n in zip(block, nodes):
            deps = [node_of_call[p] for p in r.get("parent_call_ids", [])
                    if p in node_of_call]
            if not deps and n.node_id > 0:
                deps = [n.node_id - 1]
            n.deps = deps

        arrival = clock if arrival_rate else min(
            (r.get("t_start") or 0.0) for r in block)
        if arrival_rate:
            clock += rng.expovariate(arrival_rate)

        t = Task(task_id=idx, arrival_time=arrival, nodes=nodes,
                 n_replans=max((n.replan_index for n in nodes), default=0),
                 n_attempts=max((n.attempt_index for n in nodes), default=0),
                 n_files=1, succeeded=True)
        _annotate_measured(t)
        tasks.append(t)

    if not arrival_rate and tasks:
        base = min(t.arrival_time for t in tasks)
        for t in tasks:
            t.arrival_time -= base

    return LoadedTrace(tasks=tasks, validation=validation, cache=cache,
                       prefixes=all_prefixes, blocks=blocks, configs=configs)


# --------------------------------------------------------------------------
# PROPOSAL (not implemented -- needs a tracing.py change and a re-run)
# --------------------------------------------------------------------------
#
# Config reconstruction above is inference. It is well-founded and validated,
# but it depends on orchestrator.py continuing to gate mechanisms by config
# name, and it cannot distinguish two arms that differ only in something the
# trace does not observe (a prompting strategy, say -- zero_shot and few_shot
# would be indistinguishable here).
#
# The fix is one contextvar in tracing.py:
#
#     _task_run = contextvars.ContextVar("maara_task_run", default=None)
#
#     @contextlib.contextmanager
#     def task(task_id, task_run_id=None, config=None, strategy=None):
#         ...
#
# writing task_run_id, config and strategy onto every row. That makes
# segmentation and arm identification recorded fact instead of inference, and
# would also survive a future harness that interleaves tasks, which the
# contiguity assumption would not.
#
# Deliberately not done now: it changes tracing.py and would invalidate the
# committed trace, which currently has a clean provenance chain.
