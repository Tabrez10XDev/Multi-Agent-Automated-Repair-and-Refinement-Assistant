"""
Phase 2 tests: loading a real trace into the simulator's DAG.

No LLM is called. Most tests use hand-built rows; a few load the committed
trace to check the loader against real data.
"""

import gzip
import json
import os

import pytest

from sim.engine import Engine, EngineConfig
from sim.policies import ExecutionEstimator, FCFSScheduler, LRURetention
from sim.trace_loader import (compute_prefixes, default_prompts_path,
                              detect_cache_hits, fingerprint_config, load_rows,
                              load_trace, render_prompt, segment_task_runs,
                              validate_segmentation)
from sim.workload import (CostModel, CpuStageSeconds, Kind, Node, StageTokens,
                          WorkloadConfig, generate_trace)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_TRACE = os.path.join(_REPO, "traces", "full_zs_4cfg.jsonl.gz")
has_real = pytest.mark.skipif(not os.path.exists(REAL_TRACE),
                              reason="committed trace not present")


def row(seq, task_id, role, call_id, parents=(), model="qwen2.5-coder:7b",
        kind="chat", pt=100, ot=50, t0=0.0, t1=1.0, attempt=0, replan=0):
    return {"run_id": "t", "task_id": task_id, "call_id": call_id, "seq": seq,
            "agent_role": role, "kind": kind, "attempt_index": attempt,
            "replan_index": replan, "parent_call_ids": list(parents),
            "prompt_tokens": pt, "output_tokens": ot,
            "prompt_eval_duration": 0.1, "eval_duration": 0.5,
            "t_start": t0, "t_end": t1, "model": model}


# --- IO ------------------------------------------------------------------

def test_gz_and_plain_load_identically(tmp_path):
    rows = [row(1, "T/1", "coder", "a"), row(2, "T/1", "debugger", "b", ["a"])]
    plain = tmp_path / "t.jsonl"
    plain.write_text("".join(json.dumps(r) + "\n" for r in rows))
    gz = tmp_path / "t.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert load_rows(str(plain)) == load_rows(str(gz)) == rows


def test_load_rows_tolerates_a_truncated_final_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(row(1, "T/1", "coder", "a")) + "\n{\"partial\":")
    assert len(load_rows(str(p))) == 1


def test_default_prompts_path():
    assert default_prompts_path("x/t.jsonl.gz") == "x/t.prompts.jsonl.gz"
    assert default_prompts_path("x/t.jsonl") == "x/t.prompts.jsonl"


# --- segmentation and config reconstruction ------------------------------

def test_segmentation_splits_on_task_change():
    rows = [row(1, "T/1", "coder", "a"), row(2, "T/1", "debugger", "b"),
            row(3, "T/2", "coder", "c"), row(4, "T/1", "coder", "d")]
    blocks = segment_task_runs(rows)
    assert [len(b) for b in blocks] == [2, 1, 1]
    # The same task appearing twice is two runs, not one.
    assert [b[0]["task_id"] for b in blocks] == ["T/1", "T/2", "T/1"]


def test_segmentation_orders_by_seq_not_file_order():
    rows = [row(3, "T/2", "coder", "c"), row(1, "T/1", "coder", "a"),
            row(2, "T/1", "debugger", "b")]
    assert [len(b) for b in segment_task_runs(rows)] == [2, 1]


@pytest.mark.parametrize("model,rag,sentinel,expected", [
    ("qwen2.5-coder:7b", True, True, "single_agent"),
    ("qwen2.5-coder:7b", False, True, "hetero_no_rag"),
    ("llama3.2", True, True, "homo_llama3"),
    ("llama3.2", True, False, "hetero_no_sentinel"),
])
def test_fingerprint_recovers_every_arm(model, rag, sentinel, expected):
    block = [row(1, "T/1", "coder", "a", model=model)]
    if rag:
        block.append(row(2, "T/1", "retriever", "r", kind="embedding"))
    if sentinel:
        block.append(row(3, "T/1", "evaluator", "e", model="llama3.2"))
    assert fingerprint_config(block) == expected


def test_fingerprint_returns_none_without_a_coder_call():
    assert fingerprint_config([row(1, "T/1", "issue_parser", "a")]) is None


def test_validation_flags_a_parent_edge_crossing_a_task_run():
    rows = [row(1, "T/1", "coder", "a"), row(2, "T/2", "coder", "b", ["a"])]
    v = validate_segmentation(segment_task_runs(rows))
    assert v["parent_edges_crossing_task_runs"] == 1
    assert v["parent_edges_dangling"] == 0


def test_validation_flags_a_dangling_parent():
    v = validate_segmentation(segment_task_runs([row(1, "T/1", "coder", "a", ["ghost"])]))
    assert v["parent_edges_dangling"] == 1


# --- prefix computation --------------------------------------------------

def test_prefix_finds_the_shared_opening():
    shared = "SHARED PREFIX " * 20            # 280 chars
    prompts = {
        "a": [{"role": "user", "content": shared + "tail one"}],
        "b": [{"role": "user", "content": shared + "a completely different tail"}],
    }
    block = [row(1, "T/1", "coder", "a", pt=70), row(2, "T/1", "coder", "b", pt=75)]
    pf = compute_prefixes(block, prompts, "r0")
    assert pf["a"].shared_tokens == 0 and pf["a"].shares_with is None
    assert pf["b"].shares_with == "a"
    assert pf["b"].shared_chars == len(shared)
    # chars->tokens by b's own ratio: 285 chars / 75 tokens = 3.8 chars/token
    assert pf["b"].shared_tokens == int(len(shared) / (len(shared + "a completely different tail") / 75))


def test_prefix_matches_across_roles_not_just_within():
    """
    Cross-role is the point: the coder and the Sentinel both carry the issue
    text. Restricting to same-role would miss exactly the reuse under study.
    """
    shared = "ISSUE TEXT " * 30
    prompts = {"e": [{"role": "user", "content": shared + "evaluate"}],
               "c": [{"role": "user", "content": shared + "write code"}]}
    block = [row(1, "T/1", "evaluator", "e", pt=80, model="llama3.2"),
             row(2, "T/1", "coder", "c", pt=82)]
    pf = compute_prefixes(block, prompts, "r0")
    assert pf["c"].shares_with == "e"
    assert pf["c"].shared_tokens > 0


def test_prefix_picks_the_longest_of_several_earlier_calls():
    prompts = {"a": [{"role": "user", "content": "AAAA" + "x" * 100}],
               "b": [{"role": "user", "content": "AAAABBBB" + "y" * 100}],
               "c": [{"role": "user", "content": "AAAABBBB" + "z" * 100}]}
    block = [row(1, "T/1", "coder", "a", pt=30), row(2, "T/1", "coder", "b", pt=30),
             row(3, "T/1", "coder", "c", pt=30)]
    pf = compute_prefixes(block, prompts, "r0")
    assert pf["c"].shares_with == "b"      # 8 chars, not a's 4
    assert pf["c"].shared_chars == 8


def test_identical_shared_text_yields_the_same_cache_key():
    shared = "S" * 200
    prompts = {"a": [{"role": "user", "content": shared + "1"}],
               "b": [{"role": "user", "content": shared + "2"}],
               "c": [{"role": "user", "content": shared + "3"}]}
    block = [row(i + 1, "T/1", "coder", k, pt=50)
             for i, k in enumerate(("a", "b", "c"))]
    pf = compute_prefixes(block, prompts, "r0")
    assert pf["b"].prefix_key == pf["c"].prefix_key
    assert pf["b"].prefix_key is not None


def test_render_prompt_preserves_message_order():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    assert render_prompt(msgs) == "SYS\nUSR"


def test_no_prefix_when_nothing_is_shared():
    prompts = {"a": [{"role": "user", "content": "completely different"}],
               "b": [{"role": "user", "content": "nothing alike here"}]}
    block = [row(1, "T/1", "coder", "a", pt=10), row(2, "T/1", "coder", "b", pt=10)]
    pf = compute_prefixes(block, prompts, "r0")
    assert pf["b"].shared_tokens == 0 and pf["b"].prefix_key is None


# --- ollama cache detection ----------------------------------------------

def test_cache_detector_is_quiet_on_uncached_calls():
    text = "x" * 400
    rows = [row(i, "T/1", "coder", f"c{i}", pt=100) for i in range(1, 6)]
    prompts = {f"c{i}": [{"role": "user", "content": text}] for i in range(1, 6)}
    d = detect_cache_hits(rows, prompts)
    assert d["likely_cache_hits"] == 0
    assert d["median_chars_per_token"] == pytest.approx(4.0)


def test_cache_detector_spots_a_hit():
    """A cached call reports few prompt tokens for a large prompt."""
    rows = [row(i, "T/1", "coder", f"c{i}", pt=100) for i in range(1, 6)]
    rows.append(row(6, "T/1", "coder", "hit", pt=10))      # same text, 10x fewer tokens
    prompts = {r["call_id"]: [{"role": "user", "content": "x" * 400}] for r in rows}
    d = detect_cache_hits(rows, prompts)
    assert d["likely_cache_hits"] == 1
    assert d["hidden_prompt_tokens"] == pytest.approx(90, abs=2)
    assert d["by_role"] == {"coder": 1}


# --- DAG construction ----------------------------------------------------

def test_dag_edges_come_from_parent_call_ids(tmp_path):
    rows = [row(1, "T/1", "issue_parser", "a"),
            row(2, "T/1", "coder", "b", ["a"]),
            row(3, "T/1", "debugger", "c", ["b"]),
            row(4, "T/1", "coder", "d", ["a", "c"])]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lt = load_trace(str(p))
    nodes = lt.tasks[0].nodes
    assert [n.stage for n in nodes] == ["parse_issue", "generate_fix",
                                        "diagnose", "generate_fix"]
    assert nodes[0].deps == []
    assert nodes[1].deps == [0]
    assert nodes[2].deps == [1]
    assert sorted(nodes[3].deps) == [0, 2]


def test_orphan_calls_chain_to_the_previous_node(tmp_path):
    """
    A node with no recorded parent would otherwise be released at time zero,
    ignoring everything it actually waited for.
    """
    rows = [row(1, "T/1", "issue_parser", "a"), row(2, "T/1", "coder", "b")]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    nodes = load_trace(str(p)).tasks[0].nodes
    assert nodes[1].deps == [0]


def test_embedding_rows_become_cpu_nodes(tmp_path):
    rows = [row(1, "T/1", "coder", "a"),
            row(2, "T/1", "retriever", "r", kind="embedding", t0=0.0, t1=3.5,
                pt=None, ot=None)]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    nodes = load_trace(str(p)).tasks[0].nodes
    cpu = [n for n in nodes if n.kind is Kind.CPU]
    assert len(cpu) == 1
    assert cpu[0].stage == "navigate"
    assert cpu[0].cpu_seconds == pytest.approx(3.5)
    assert cpu[0].output_tokens == 0


# --- real trace ----------------------------------------------------------

@has_real
def test_real_trace_segments_cleanly():
    lt = load_trace(REAL_TRACE)
    v = lt.validation
    assert v["parent_edges_crossing_task_runs"] == 0
    assert v["parent_edges_dangling"] == 0
    assert v["unidentified_configs"] == 0
    assert set(v["runs_per_config"]) == {
        "single_agent", "homo_llama3", "hetero_no_rag", "hetero_no_sentinel"}


@has_real
def test_real_and_synthetic_traces_have_the_same_structure():
    real = load_trace(REAL_TRACE).tasks
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    synth = generate_trace(WorkloadConfig(n_tasks=5, seed=3), cost, tok, cpu)

    for group in (real, synth):
        assert group and all(isinstance(n, Node) for t in group for n in t.nodes)
    real_fields = {f for n in real[0].nodes for f in vars(n)}
    synth_fields = {f for n in synth[0].nodes for f in vars(n)}
    assert real_fields == synth_fields
    # task_id must index into the list, since the engine does self.tasks[task_id]
    assert all(t.task_id == i for i, t in enumerate(real))


@has_real
@pytest.mark.parametrize("source", ["real", "synthetic"])
def test_both_sources_run_through_the_engine(source):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    if source == "real":
        tasks = load_trace(REAL_TRACE, arrival_rate=0.2).tasks
    else:
        tasks = generate_trace(WorkloadConfig(n_tasks=20, arrival_rate=0.2, seed=3),
                               cost, tok, cpu)
    est = ExecutionEstimator(cost, tok, cpu)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=4096.0, max_batch=8),
                 retention=LRURetention(), scheduler=FCFSScheduler(), estimator=est)
    rep = eng.run()
    assert rep["tasks_completed"] == len(tasks)
    assert len(eng.done_nodes) == sum(len(t.nodes) for t in tasks)
    assert eng.pinned_bytes == 0


@has_real
def test_real_timing_is_sequential_and_retiming_creates_overlap():
    """
    The traced run was single-stream, so replaying real timestamps yields a
    workload with no contention. arrival_rate re-times it while keeping every
    measured duration.
    """
    seq = load_trace(REAL_TRACE).tasks
    fast = load_trace(REAL_TRACE, arrival_rate=1.0).tasks
    seq_span = max(t.arrival_time for t in seq)
    fast_span = max(t.arrival_time for t in fast)
    assert fast_span < seq_span
    assert [len(t.nodes) for t in seq] == [len(t.nodes) for t in fast]
