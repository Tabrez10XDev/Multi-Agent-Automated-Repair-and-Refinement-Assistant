"""
TraceLab adapter (DECISIONS 37). Everything runs on a hand-built tiny raw file
written here, never on the real release: the builder's rules (segmentation,
cleaning, clipping, tool span, epoch, split, sample), the adapter's DAG, block
keys and oracle fields, the estimators, and the registry hooks.
"""

import gzip
import hashlib
import importlib.util
import json
import os
import sys

import pytest

from experiments.common import make_estimator
from simulator import workload_sources as ws
from simulator.adapters import tracelab as tl
from simulator.engine import Engine, EngineConfig
from simulator.policies import SCHEDULERS, ExecutionEstimator, KeepAllRetention
from simulator.trace_loader import SealedTraceSet
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, Kind,
                                StageTokens)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_tracelab_under_test", os.path.join(REPO, "experiments", "build_tracelab.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_builder()


# ---------------------------------------------------------------- fixture

def _user_on(side, prefix="u"):
    for i in range(1000):
        u = f"{prefix}{i}"
        if B.split_of(u) == side:
            return u
    raise AssertionError


DEV_USER, SEALED_USER = _user_on("dev"), _user_on("sealed")


def ev(kind, sec, **kw):
    return {"event_type": kind, "timestamp": f"2026-01-01T00:{sec // 60:02d}:{sec % 60:02d}.000Z", **kw}


def tool(i, emit, result, wall=None, internal=None):
    t = {"tool_index": i, "tool_name": "Bash"}
    if emit is not None:
        t["emitted_at"] = f"2026-01-01T00:00:00.000Z" if emit == 0 else \
            f"2026-01-01T{emit // 3600:02d}:{emit % 3600 // 60:02d}:{emit % 60:02d}.000Z"
    if result is not None:
        t["result_at"] = f"2026-01-01T{result // 3600:02d}:{result % 3600 // 60:02d}:{result % 60:02d}.000Z"
    if wall is not None:
        t["tool_wall_latency_ms"] = wall
    if internal is not None:
        t["tool_internal_latency_ms"] = internal
    return t


def row(sid, ri, rid, pre, app, out, events, tools=(), user=DEV_USER, provider="claude",
        inp="auto"):
    return {"provider": provider, "session_id": sid, "round_index": ri, "round_id": rid,
            "input_tokens_total": pre + app if inp == "auto" else inp,
            "prefix_tokens": pre, "newly_append_tokens": app, "output_tokens": out,
            "timing_events": list(events), "tools": list(tools), "user": user}


USER_FIRST = lambda sec: [ev("user_message", sec), ev("text", sec + 1)]
CONT = lambda sec: [ev("tool_result", sec), ev("text", sec + 1)]


def session_s1():
    """One dev session. Rounds are written to the file OUT of round_index order."""
    s = "claude:S1"
    return [
        # r0: no trigger -> before the first trigger, dropped
        row(s, 0, "a", 0, 500, 20, [ev("text", 1)]),
        # r1 opens turn 0 (user_message before first output); one tool, 2 s
        row(s, 1, "b", 0, 1000, 50, USER_FIRST(10),
            [tool(0, 12, 14)]),
        # r2 continues turn 0; two overlapping tools: emit 20/21, results 25/23 -> 5 s
        row(s, 2, "c", 1000, 100, 60, CONT(18), [tool(0, 20, 25), tool(1, 21, 23)]),
        # r3 opens turn 1; no tools
        row(s, 3, "d", 1100, 40, 10, USER_FIRST(100)),
        # r4: user_message AFTER its first output -> does not open a turn
        row(s, 4, "e", 1140, 30, 11, [ev("text", 110), ev("user_message", 112)]),
        # r5: prefix far under half the previous input -> new epoch; stalled tool (700 s)
        row(s, 5, "f", 100, 900, 30, CONT(120), [tool(0, 0, 700)]),
        # r6 opens turn 2: clipping order; tool missing result_at -> fallback to internal
        row(s, 6, "g", 50000, 40000, 9000, USER_FIRST(200),
            [tool(0, 210, None, wall=5000, internal=3000)]),
        # r7: prefix alone clipped
        row(s, 7, "h", 40000, 10, 100, CONT(300)),
    ]


def write_raw(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def raw(tmp_path):
    rows = session_s1()
    rows = [rows[3], rows[1], rows[0], rows[2], rows[6], rows[5], rows[4], rows[7]]
    # null input -> dropped; exact duplicate key (different tokens) -> first kept
    rows.append(row("claude:S1", 8, "n", 5, 5, 5, CONT(400), inp=None) | {"output_tokens": 5})
    rows[-1]["input_tokens_total"] = None
    rows.append(row("claude:S1", 1, "b", 77, 77, 77, USER_FIRST(10)))
    p = tmp_path / "raw.jsonl.gz"
    write_raw(p, rows)
    return str(p)


def parse(raw):
    c = __import__("collections").Counter()
    sessions = B.read_rows(raw, c)
    turns = B.build_turns(sessions, c)
    return c, turns


# -------------------------------------------------------- builder: cleaning

def test_null_and_duplicate_rows_are_counted_and_dropped(raw):
    c, turns = parse(raw)
    assert c["rows_read"] == 10
    assert c["dropped_null_input_or_output"] == 1
    assert c["dropped_duplicate"] == 1
    # the duplicate of (claude, S1, 1, b) had 77-token fields; the first stays
    rounds = [x for t in turns for x in t["rounds"]]
    assert 77 not in [x["pre"] for x in rounds]


def test_turn_segmentation_and_pre_trigger_drop(raw):
    c, turns = parse(raw)
    assert c["dropped_pre_trigger_rounds"] == 1            # r0
    assert [len(t["rounds"]) for t in turns] == [2, 3, 2]   # r4 stays in turn 1
    assert [t["ordinal"] for t in turns] == [0, 1, 2]       # 0-based
    # file order was shuffled; rounds come back in round_index order
    assert [x["ri"] for x in turns[0]["rounds"]] == [1, 2]


# ----------------------------------------------- builder: clip / tool / epoch

def test_clipping_order_output_then_append_then_prefix():
    assert B.clip(50000, 40000, 9000) == (0, 24576, 8192)
    assert B.clip(40000, 10, 100) == (32658, 10, 100)
    assert B.clip(100, 200, 300) == (100, 200, 300)


def test_tool_span_overlap_cap_and_fallback(raw):
    c, turns = parse(raw)
    done = [B.finish_turn(t) for t in turns]
    t0, t1, t2 = (d[0] for d in done)
    assert t0["tool_s"] == [2.0, 5.0]                   # span, parallel calls overlap
    assert t0["raw"]["tool_src"] == ["span", "span"]
    assert t1["tool_s"] == [None, None, 600.0]          # no tools -> null; stall capped
    assert t1["raw"]["tool_s"][2] == 700.0
    assert done[1][1]["tool_nodes_capped"] == 1
    assert t2["tool_s"] == [3.0, None]                  # internal latency beats wall
    assert t2["raw"]["tool_src"][0] == "fallback"
    assert done[2][1]["tool_fallback"] == 1


def test_epoch_resets_on_a_collapsed_prefix_using_preclip_values(raw):
    _, turns = parse(raw)
    ep = [x["epoch"] for t in turns for x in t["rounds"]]
    # r1: prefix 0 < 0.5 * r0's input (the dropped pre-trigger round still counts as
    # the previous round) -> epoch 1; r2,r3,r4 stay; r5 prefix 100 < 0.5*1170 -> 2;
    # r6 prefix 50000 > 0.5*1000 stays; r7 prefix 40000 vs prev input 90000 (pre-clip!)
    # -> 40000 < 45000 -> 3
    assert ep == [1, 1, 1, 1, 2, 2, 3]
    done = B.finish_turn(turns[2])[0]
    assert done["epoch"] == [2, 3]
    assert done["prefix"][0] == 0 and done["raw"]["prefix"][0] == 50000


# --------------------------------------------------- builder: split + sample

def test_split_is_by_user_first_byte():
    for u in ("a", "b", "user_1", "user_2"):
        d = hashlib.sha256(("tracelab-split:" + u).encode()).digest()[0]
        assert B.split_of(u) == ("dev" if d < 128 else "sealed")


def test_build_is_deterministic_and_splits_users(tmp_path, raw):
    other = [row("codex:S2", i, f"x{i}", 10 * i, 5, 5,
                 USER_FIRST(10 * i), user=SEALED_USER, provider="codex")
             for i in range(6)]
    p2 = tmp_path / "raw2.jsonl.gz"
    write_raw(p2, [json.loads(l) for l in gzip.open(raw, "rt")] + other)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    B.build(str(p2), str(out1), 2)
    B.build(str(p2), str(out2), 2)
    for f in ("dev.jsonl.gz", "sealed.jsonl.gz", "dev_constants.json"):
        assert (out1 / f).read_bytes() == (out2 / f).read_bytes()
    hd, dev = tl.read_derived(str(out1 / "dev.jsonl.gz"))
    hs, sealed = tl.read_derived(str(out1 / "sealed.jsonl.gz"))
    assert {r["session_id"] for r in dev} == {"claude:S1"}
    assert {r["session_id"] for r in sealed} == {"codex:S2"}
    assert len(dev) == 2 and len(sealed) == 2            # N applied per split
    # the N smallest sample digests, in digest order
    dg = lambda r: hashlib.sha256(
        f"tracelab-sample:{r['session_id']}:{r['ordinal']}".encode()).hexdigest()
    assert [dg(r) for r in dev] == sorted(dg(r) for r in dev)
    assert hd["counts"]["turns_in_split"] == 3 and hd["counts"]["turns_sampled"] == 2
    # dev constants come from the dev side only (3 turns of S1: R = 2, 3, 2)
    consts = json.loads((out1 / "dev_constants.json").read_text())
    assert consts["rounds_per_turn_counts"] == {"2": 2, "3": 1}
    assert consts["n_rounds"] == 7


# -------------------------------------------------------------- the adapter

COST = CostModel.from_profile("gpu_5060ti_public")


def _turn(sid="claude:S1", epochs=(0, 0, 0), prefix=(0, 160, 330), append=(160, 190, 10),
          output=(5, 6, 7), tool_s=(1.5, None, 2.0)):
    return {"session_id": sid, "epoch": list(epochs), "prefix": list(prefix),
            "append": list(append), "output": list(output), "tool_s": list(tool_s)}


def test_chain_dag_shape_and_fields():
    t = tl.turn_to_task(_turn(), 7, COST)
    assert [(n.stage, n.kind) for n in t.nodes] == [
        ("agent_round", Kind.LLM), ("tool", Kind.CPU), ("agent_round", Kind.LLM),
        ("agent_round", Kind.LLM), ("tool", Kind.CPU)]
    assert [n.deps for n in t.nodes] == [[], [0], [1], [2], [3]]
    assert all(n.task_id == 7 for n in t.nodes)
    llm = [n for n in t.nodes if n.kind is Kind.LLM]
    assert all(n.model == CODER_MODEL for n in llm)
    assert [(n.prefix_tokens, n.suffix_tokens, n.output_tokens) for n in llm] == [
        (0, 160, 5), (160, 190, 6), (330, 10, 7)]
    assert [n.attempt_index for n in llm] == [0, 1, 2]
    assert t.nodes[1].cpu_seconds == 1.5


def test_zero_output_is_floored_at_one_token():
    t = tl.turn_to_task(_turn(output=(0, 0, 0)), 0, COST)
    assert all(n.output_tokens == 1 for n in t.nodes if n.kind is Kind.LLM)


def test_block_keys_shared_within_session_epoch_not_across_reset():
    a = tl.turn_to_task(_turn(), 0, COST)
    llm = [n for n in a.nodes if n.kind is Kind.LLM]
    k0, k1, k2 = (n.block_keys for n in llm)
    assert [len(k) for k in (k0, k1, k2)] == [10, 22, 22]    # ceil(input/16)
    assert k1[:10] == k0 and k2[:22] == k1                   # leading blocks shared
    # a reset (epoch change) shares nothing with the earlier epoch
    b = tl.turn_to_task(_turn(epochs=(0, 0, 1)), 0, COST)
    kb2 = [n for n in b.nodes if n.kind is Kind.LLM][2].block_keys
    assert not set(kb2) & set(k1)
    # another session shares nothing either
    c = tl.turn_to_task(_turn(sid="claude:OTHER"), 0, COST)
    assert not set(c.nodes[0].block_keys) & set(k0)
    # model is folded into the seed
    from simulator.workload import build_block_chain
    assert llm[0].block_keys == build_block_chain(CODER_MODEL, [("claude:S1|e0", 10)])
    assert llm[0].block_keys != build_block_chain("llama3.2", [("claude:S1|e0", 10)])


def test_oracle_fields_under_the_public_profile_at_batch_one():
    rows = [_turn(), _turn(sid="claude:S9")]
    tasks = tl.build_tasks(rows, seed=1, arrival_rate=0.1, cost=COST)
    t = tasks[0]
    for n in t.nodes:
        if n.kind is Kind.CPU:
            assert n.expected_seconds == n.cpu_seconds
        else:
            want = (COST.prefill_seconds(n.model, n.prompt_tokens)
                    + n.output_tokens * COST.decode_iteration_seconds(n.model, 1))
            assert n.expected_seconds == pytest.approx(want)
    total = sum(n.expected_seconds for n in t.nodes)          # a chain: path = sum
    assert t.critical_path_total == pytest.approx(total)
    assert t.nodes[0].critical_path_remaining == pytest.approx(total)
    assert t.nodes[-1].critical_path_remaining == pytest.approx(t.nodes[-1].expected_seconds)
    # within-turn reuse edge: round 0 -> round 1 share a prefix key only if prefix>0
    assert t.nodes[2].prefix_key == "claude:S1|e0"
    # cross-turn per-block gaps are filled
    assert any(n.block_reuse_gap for n in t.nodes if n.kind is Kind.LLM)


def test_arrivals_poisson_convention_and_sequential_default():
    rows = [_turn(sid=f"s{i}") for i in range(4)]
    pois = tl.build_tasks(rows, seed=3, arrival_rate=0.5, cost=COST)
    import random
    rng, clock, want = random.Random(3), 0.0, []
    for _ in rows:
        want.append(clock)
        clock += rng.expovariate(0.5)
    assert [t.arrival_time for t in pois] == pytest.approx(want)
    assert [t.task_id for t in pois] == [0, 1, 2, 3]
    seq = tl.build_tasks(rows, seed=3, arrival_rate=None, cost=COST)
    assert seq[0].arrival_time == 0.0
    for a, b in zip(seq, seq[1:]):
        assert b.arrival_time >= a.arrival_time + a.critical_path_total


# --------------------------------------------------------------- estimators

CONSTS = {"rounds_per_turn_counts": {"1": 2, "2": 1, "5": 1},
          "mean_prefix_tokens": 1000.0, "mean_append_tokens": 200.0,
          "mean_output_tokens": 100.0, "mean_tool_s_per_round": 2.0}


def _est(cls=tl.ChainEstimator):
    return cls(COST, constants=CONSTS)


def test_mean_residual_life_arithmetic():
    e = _est()
    assert e.mean_residual_rounds(0) == pytest.approx((1 * 2 + 2 * 1 + 5 * 1) / 4)
    assert e.mean_residual_rounds(1) == pytest.approx((1 * 1 + 4 * 1) / 2)   # R in {2, 5}
    assert e.mean_residual_rounds(2) == pytest.approx(3.0)                   # R = 5
    assert e.mean_residual_rounds(5) == 1.0          # no dev turn has R > 5
    assert e.mean_residual_rounds(99) == 1.0


def test_chain_estimator_remaining_total_and_output_prediction():
    e = _est()
    rc = e.mean_round_cost()
    want = (COST.prefill_seconds(CODER_MODEL, 1200) + 100.0 * COST.decode_iteration_seconds(CODER_MODEL, 1))
    assert rc == pytest.approx(want)
    # round j=1 queued: residual E[R-1|R>1]=2.5 rounds incl. this one -> 1.5 further
    assert e.remaining_s("agent_round", 1, CODER_MODEL) == pytest.approx(2.0 + 1.5 * (rc + 2.0))
    # past every dev turn: this round is the last, only its own tool remains
    assert e.remaining_s("agent_round", 7, CODER_MODEL) == pytest.approx(2.0)
    assert e.estimate_task_total() == pytest.approx(2.25 * (rc + 2.0))
    # output: dev mean cold, then the running mean of observations; true_output ignored
    assert e.predict_output("agent_round", 0, true_output=9999) == 100.0
    e.observe("agent_round", 0, 300)
    e.observe("agent_round", 1, 500)
    assert e.predict_output("agent_round", 0, true_output=1) == 400.0


def test_oracle_estimator_reads_the_recorded_remainder():
    t = tl.turn_to_task(_turn(), 0, COST)
    tl._annotate_measured(t)
    o = _est(tl.ChainOracleEstimator)
    n = t.nodes[0]
    assert o.is_oracle and o.name == "oracle"
    assert o.remaining_s("agent_round", 0, CODER_MODEL, node=n) == pytest.approx(
        n.critical_path_remaining - n.expected_seconds)
    assert o.predict_output("agent_round", 0, true_output=42) == 42.0
    assert not _est().is_oracle


@pytest.mark.parametrize("sched", ["fcfs", "sjf", "edf", "lsf", "plas", "program-fcfs",
                                   "program-fcfs-cap", "program-fcfs-qcap"])
@pytest.mark.parametrize("oracle", [False, True])
def test_engine_runs_chain_workload_under_every_supported_policy(sched, oracle):
    rows = [_turn(sid=f"s{i % 2}") for i in range(4)]
    tasks = tl.build_tasks(rows, seed=1, arrival_rate=2.0, cost=COST)
    est = _est(tl.ChainOracleEstimator if oracle else tl.ChainEstimator)
    eng = Engine(tasks, COST, EngineConfig(kv_budget_mb=512, max_batch=2,
                                           deadline_basis="task",
                                           slo_classes=(("tight", 1.0, 0.5), ("loose", 4.0, 0.5))),
                 retention=KeepAllRetention(), scheduler=SCHEDULERS[sched](), estimator=est)
    rep = eng.run()
    assert len(eng.task_done_at) == 4
    assert rep["p95_slowdown"] > 0


# ----------------------------------------------------------------- registry

def test_names_are_registered_without_a_flag():
    names = ws.known_sources()
    assert "tracelab_dev" in names and "tracelab_sealed" in names
    dev, sealed = ws.get_source("tracelab_dev"), ws.get_source("tracelab_sealed")
    assert not dev.sealed and sealed.sealed
    for s in (dev, sealed):
        assert s.supports_deadline_basis == frozenset({"task"})
        assert s.excluded_policies == frozenset({"hint-priority"})
        assert s.estimator_factory and s.oracle_factory
    ws.check_deadline_basis("tracelab_dev", "task")
    with pytest.raises(ValueError):
        ws.check_deadline_basis("tracelab_dev", "estimate")


def test_sealed_gate_refuses_tracelab_sealed(monkeypatch):
    monkeypatch.delenv("MAARA_POLICIES_FROZEN", raising=False)
    with pytest.raises(SealedTraceSet, match="DECISIONS 26"):
        ws.resolve_workload("tracelab_sealed", 1, 0.1)
    from experiments.common import build
    with pytest.raises(SealedTraceSet):
        build(1, 1, 0.1, None, None, None, trace_set="tracelab_sealed")


def test_make_estimator_dispatch_for_tracelab_and_maara_unchanged():
    cost, tok, cpu = COST, StageTokens(), CpuStageSeconds()
    assert type(make_estimator("tracelab_dev", cost, tok, cpu)) is tl.ChainEstimator
    assert type(make_estimator("tracelab_dev", cost, tok, cpu, name="oracle")) \
        is tl.ChainOracleEstimator
    # MAARA: exactly the stock classes, for every name
    assert type(make_estimator("holdout_qwen", cost, tok, cpu)) is ExecutionEstimator
    assert type(make_estimator("holdout_qwen", cost, tok, cpu, name="oracle")).__name__ \
        == "OracleEstimator"
    assert ws.get_source("holdout_qwen").excluded_policies == frozenset()


def test_scheduling_map_skips_excluded_policies_and_records_them(monkeypatch):
    from experiments import scheduling_map as sm

    seen = {}

    class Stop(Exception):
        pass

    def fake_start(kind, params):
        seen.update(params)
        raise Stop

    monkeypatch.setattr(sm.Manifest, "start", staticmethod(fake_start))
    monkeypatch.setattr(sys, "argv", ["scheduling_map", "--trace-set", "tracelab_dev",
                                      "--deadline-basis", "task", "--profile",
                                      "gpu_5060ti_public"])
    with pytest.raises(Stop):
        sm.main()
    assert "hint-priority" not in seen["policies"] and len(seen["policies"]) == len(sm.POLICIES) - 1
    assert seen["excluded_policies"] == ["hint-priority"]
    # a MAARA run keeps every policy and gains no new manifest key
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["scheduling_map", "--trace-set", "holdout_qwen"])
    with pytest.raises(Stop):
        sm.main()
    assert "hint-priority" in seen["policies"] and "excluded_policies" not in seen


def test_committed_dev_file_matches_the_pre_registered_source():
    h, rows = tl.read_derived(os.path.join(tl.DATA_DIR, "dev.jsonl.gz"))
    assert h["source_sha256"] == B.RAW_SHA256 and h["release"] == "v0.0.2"
    assert h["split"] == "dev" and len(rows) == h["counts"]["turns_sampled"] == 80
    tasks = ws.resolve_workload("tracelab_dev", 1, 0.05)
    assert [t.task_id for t in tasks] == list(range(80))
    assert all(n.model == CODER_MODEL for t in tasks for n in t.nodes if n.kind is Kind.LLM)
    c = tl.load_dev_constants()
    assert c["build_script_git_hash"] == h["build_script_git_hash"]
    assert os.path.exists(os.path.join(tl.DATA_DIR, "sealed.jsonl.gz"))   # existence only
