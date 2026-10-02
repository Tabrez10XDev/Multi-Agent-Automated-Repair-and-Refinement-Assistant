"""
Per-call tracing for MAARA's LLM pipeline.

Every ollama chat call in agents/ routes through `chat()` here, which records one
JSONL row per call. The pipeline's behaviour is unchanged: `chat()` forwards to
ollama.chat and returns its response object untouched, and when tracing is
disabled it is a straight pass-through.

ENABLING
--------
Off by default, so an untraced run behaves exactly as before.

    MAARA_TRACE=1                  enable; writes data/traces/<run_id>.jsonl.gz
    MAARA_TRACE_SMOKE=1            write data/traces/smoke/<run_id>.jsonl (uncompressed)
    MAARA_RUN_ID=<id>              pin the run id (default: UTC timestamp + short uuid)

One trace file per PROCESS, created lazily on the first traced call and closed
at interpreter exit. That matters because benchmark/run_experiments.py calls
orchestrator.run() once per task in a single process -- a per-call or per-task
file would shatter one experiment into hundreds of fragments.

ROW SCHEMA
----------
    run_id            stable for the whole process
    task_id           e.g. "HumanEval/7", or the case id under benchmark/runner
    call_id           unique within the run
    seq               monotonic call counter, gives total order
    agent_role        real module name: issue_parser | evaluator | coder |
                      debugger | retriever
    kind              "chat" | "embedding"
    attempt_index     0 before the retry loop, 1..max_attempts inside it
    replan_index      0, or 1..MAX_REPLAN inside the Sentinel replan loop
    parent_call_ids   calls whose OUTPUT is textually present in this prompt
    prompt_tokens     from ollama prompt_eval_count  (null for embeddings)
    output_tokens     from ollama eval_count         (null for embeddings)
    prompt_eval_duration   SECONDS (ollama reports ns; converted here)
    eval_duration          SECONDS
    t_start, t_end    wall-clock epoch seconds
    model
    n_documents       embeddings only
    error             present only if the call raised

Durations are stored in seconds rather than ollama's native nanoseconds so they
are directly comparable with everything else in sim/, which is all seconds.

PROMPT SIDECAR
--------------
The full messages list of every call goes to a separate file,
data/traces/<run_id>.prompts.jsonl(.gz), one row per call: run_id, call_id, task_id,
agent_role, model, messages. Joined to the main trace on call_id. Kept separate
so the main trace stays small enough to scan quickly, while the prompts -- which
are orders of magnitude larger -- are read only when needed.

Phase 2's loader uses these to recover the reusable prefix: for each call, the
longest common prefix against every EARLIER call in the same task (any role, not
only the same role, since coder and evaluator both carry the issue text and the
test files), converted from characters to tokens using that call's own
prompt_tokens / prompt character length.

That number is APPROXIMATE and is labelled as such in CostModel.provenance().
Three reasons it is not a measurement:

  1. Ollama applies its chat template server-side, so what the model actually
     tokenizes is not the concatenation of these message strings. Template
     tokens, role markers and BOS/EOS are invisible from here.
  2. chars -> tokens via one per-call ratio is a linear approximation, and
     tokenization density is not uniform across code, prose and JSON within a
     single prompt.
  3. The common prefix is measured in characters, so it can end mid-token,
     whereas a real KV prefix cache matches at token boundaries.

The error is small relative to the effect under study (prefix sizes span roughly
480 to 27,000 tokens), but it must never be reported as measured.

PARENT EDGES
------------
A parent is a call whose *output text* appears in this call's prompt. Derived by
reading orchestrator.py and the agents:

  issue_parser  first call   -> none (issue_text is external input)
                replan       -> the evaluator call whose feedback is appended
                                (orchestrator.py:80-83)
  retriever                  -> none. Retrieval embeds raw issue_text and repo
                                source, neither of which is LLM output. There is
                                a control dependency on issue_parser but not a
                                data one, so no edge is recorded.
  evaluator                  -> issue_parser (intent, affected_files) and
                                retriever (relevant_code)  [evaluator.py:11-22]
  coder                      -> issue_parser, retriever, plus PREVIOUS attempt's
                                coder (previous_attempt) and debugger
                                (diagnosis)  [coder.py:97-131]
  debugger                   -> this attempt's coder (fixed_code), plus previous
                                attempts' debugger (previous_diagnoses)
                                [debugger.py:15-36]

Call sites declare parents by ROLE, not by id, and unresolved roles are silently
dropped. That is what makes `parents=("evaluator",)` correct for issue_parser on
both paths: on the first call no evaluator has run yet, so it resolves to no
parents; on a replan it resolves to the evaluator that just ran.

KNOWN APPROXIMATION: generate_fix loops over files (coder.py:95), so one attempt
can emit several coder calls. Role resolution records only the most recent, so a
multi-file attempt's fan-out shows one parent edge rather than all of them. The
full fan-out is still recoverable offline from seq + attempt_index.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import datetime
import gzip
import json
import os
import threading
import time
import uuid

_REPO = os.path.dirname(os.path.abspath(__file__))

_task_id = contextvars.ContextVar("maara_task_id", default=None)
_attempt = contextvars.ContextVar("maara_attempt", default=0)
_replan = contextvars.ContextVar("maara_replan", default=0)
_last_by_role = contextvars.ContextVar("maara_last_by_role", default=None)
_prev_attempt_by_role = contextvars.ContextVar("maara_prev_attempt", default=None)

_run_lock = threading.Lock()
_run = None


class _Run:
    def __init__(self, run_id, path, fh, prompt_path, prompt_fh):
        self.run_id = run_id
        self.path = path
        self.fh = fh
        self.prompt_path = prompt_path
        self.prompt_fh = prompt_fh
        self.seq = 0
        self.lock = threading.Lock()

    def write(self, row):
        with self.lock:
            self.seq += 1
            row["seq"] = self.seq
            self.fh.write(json.dumps(row) + "\n")
            # Flush per row: a multi-hour experiment that dies partway should
            # still yield a usable trace.
            self.fh.flush()
        return row["seq"]

    def write_prompt(self, row):
        with self.lock:
            self.prompt_fh.write(json.dumps(row) + "\n")
            self.prompt_fh.flush()

    def close(self):
        with self.lock:
            for fh in (self.fh, self.prompt_fh):
                try:
                    fh.close()
                except Exception:
                    pass


def enabled() -> bool:
    return os.environ.get("MAARA_TRACE", "").strip() not in ("", "0", "false", "False")


def _smoke() -> bool:
    return os.environ.get("MAARA_TRACE_SMOKE", "").strip() not in ("", "0", "false", "False")


def _append_allowed() -> bool:
    return os.environ.get("MAARA_TRACE_APPEND", "").strip() not in ("", "0", "false", "False")


def _guard_existing(base: str, run_id: str) -> None:
    """
    Refuse to write into a trace file that already exists.

    MAARA_RUN_ID pins the filename, so a run interrupted and restarted under
    the same id used to append a SECOND seq sequence starting again at 1, into
    the same file, under the same run_id. Nothing in the data marked the
    boundary except seq ceasing to advance, and segmenting by seq then
    interleaved the two processes and fabricated a task-run boundary on nearly
    every row. That silently invalidated a whole run's reuse figures.

    Set MAARA_TRACE_APPEND=1 to allow it deliberately.
    """
    if _append_allowed():
        return
    smoke = _smoke()
    d = os.path.join(base, "smoke") if smoke else base
    ext = ".jsonl" if smoke else ".jsonl.gz"
    for path in (os.path.join(d, f"{run_id}{ext}"),
                 os.path.join(d, f"{run_id}.prompts{ext}")):
        if os.path.exists(path):
            raise RuntimeError(
                f"trace file already exists: {path}\n"
                f"MAARA_RUN_ID={run_id!r} would append a second call sequence to "
                f"it, which cannot be told apart from the first except by seq "
                f"failing to advance.\n"
                f"Either choose a new MAARA_RUN_ID, move the existing file "
                f"aside, or set MAARA_TRACE_APPEND=1 to append on purpose.")


def _get_run():
    """Lazily open the process-wide trace file. Returns None when disabled."""
    global _run
    if not enabled():
        return None
    if _run is not None:
        return _run
    with _run_lock:
        if _run is not None:
            return _run
        run_id = os.environ.get("MAARA_RUN_ID") or (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:6])
        base = os.environ.get("MAARA_TRACE_DIR") or os.path.join(_REPO, "data", "traces")
        _guard_existing(base, run_id)
        if _smoke():
            d = os.path.join(base, "smoke")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{run_id}.jsonl")
            ppath = os.path.join(d, f"{run_id}.prompts.jsonl")
            fh = open(path, "a", encoding="utf-8")
            pfh = open(ppath, "a", encoding="utf-8")
        else:
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"{run_id}.jsonl.gz")
            ppath = os.path.join(base, f"{run_id}.prompts.jsonl.gz")
            fh = gzip.open(path, "at", encoding="utf-8")
            pfh = gzip.open(ppath, "at", encoding="utf-8")
        _run = _Run(run_id, path, fh, ppath, pfh)
        atexit.register(_run.close)
        print(f"[tracing] writing {path}")
        print(f"[tracing] writing {ppath}")
        return _run


def current_path():
    r = _get_run()
    return r.path if r else None


def _reset():
    """Test-only: close and forget the active run so the next call reopens."""
    global _run
    with _run_lock:
        if _run is not None:
            _run.close()
        _run = None


# --- context -------------------------------------------------------------

@contextlib.contextmanager
def task(task_id):
    """Scope one workflow task. Resets parent tracking so tasks never link."""
    t1 = _task_id.set(str(task_id))
    t2 = _last_by_role.set({})
    t3 = _prev_attempt_by_role.set({})
    t4 = _attempt.set(0)
    t5 = _replan.set(0)
    try:
        yield
    finally:
        _replan.reset(t5)
        _attempt.reset(t4)
        _prev_attempt_by_role.reset(t3)
        _last_by_role.reset(t2)
        _task_id.reset(t1)


def set_attempt(i):
    """
    Mark the start of retry attempt `i`.

    Snapshots the current per-role last-call map, so coder/debugger resolve
    prev_attempt_parents to the PREVIOUS attempt's calls rather than picking up
    their own earlier call within this attempt.

    Deliberately not a context manager: wrapping orchestrator.py's ~70-line
    retry body in a `with` would mean re-indenting it, and a large whitespace
    diff through working pipeline code is not worth the tidiness. Values are
    bounded by the enclosing task() scope, which resets them on exit.
    """
    _prev_attempt_by_role.set(dict(_last_by_role.get() or {}))
    _attempt.set(int(i))


def set_replan(i):
    """Mark the start of Sentinel replan iteration `i`."""
    _replan.set(int(i))


@contextlib.contextmanager
def attempt(i):
    """Context-manager form of set_attempt, for tests and new code."""
    snapshot = dict(_last_by_role.get() or {})
    t1 = _prev_attempt_by_role.set(snapshot)
    t2 = _attempt.set(int(i))
    try:
        yield
    finally:
        _attempt.reset(t2)
        _prev_attempt_by_role.reset(t1)


@contextlib.contextmanager
def replan(i):
    """Context-manager form of set_replan."""
    t = _replan.set(int(i))
    try:
        yield
    finally:
        _replan.reset(t)


def _resolve(roles, mapping):
    out = []
    for role in roles:
        cid = (mapping or {}).get(role)
        if cid:
            out.append(cid)
    return out


def _record(role, cid):
    m = _last_by_role.get()
    if m is not None:
        m[role] = cid


# --- the wrapper ---------------------------------------------------------

def chat(model, messages, role, parents=(), prev_attempt_parents=(), **kwargs):
    """
    Drop-in replacement for ollama.chat that records one trace row.

    Returns exactly what ollama.chat returns. Exceptions propagate unchanged;
    a row is still written with an `error` field so failures are visible in the
    trace rather than appearing as a gap.
    """
    import ollama

    run = _get_run()
    if run is None:
        return ollama.chat(model=model, messages=messages, **kwargs)

    call_id = uuid.uuid4().hex[:12]
    parent_ids = (_resolve(parents, _last_by_role.get())
                  + _resolve(prev_attempt_parents, _prev_attempt_by_role.get()))

    t_start = time.time()
    # Written BEFORE the call: the prompt exists regardless of outcome, so a
    # process killed mid-call still leaves the prompt recoverable. Joined to the
    # main trace on call_id, so no seq is needed here.
    run.write_prompt({
        "run_id": run.run_id, "call_id": call_id, "task_id": _task_id.get(),
        "agent_role": role, "model": model, "messages": messages,
    })
    try:
        response = ollama.chat(model=model, messages=messages, **kwargs)
    except Exception as e:
        run.write({
            "run_id": run.run_id, "task_id": _task_id.get(), "call_id": call_id,
            "agent_role": role, "kind": "chat",
            "attempt_index": _attempt.get(), "replan_index": _replan.get(),
            "parent_call_ids": parent_ids,
            "prompt_tokens": None, "output_tokens": None,
            "prompt_eval_duration": None, "eval_duration": None,
            "t_start": t_start, "t_end": time.time(),
            "model": model, "error": repr(e),
        })
        raise
    t_end = time.time()

    def _sec(key):
        v = response.get(key)
        return (v / 1e9) if isinstance(v, (int, float)) else None

    run.write({
        "run_id": run.run_id, "task_id": _task_id.get(), "call_id": call_id,
        "agent_role": role, "kind": "chat",
        "attempt_index": _attempt.get(), "replan_index": _replan.get(),
        "parent_call_ids": parent_ids,
        "prompt_tokens": response.get("prompt_eval_count"),
        "output_tokens": response.get("eval_count"),
        "prompt_eval_duration": _sec("prompt_eval_duration"),
        "eval_duration": _sec("eval_duration"),
        "t_start": t_start, "t_end": t_end,
        "model": model,
    })
    _record(role, call_id)
    return response


@contextlib.contextmanager
def embedding_op(op, n_documents=0, model="nomic-embed-text"):
    """
    Time an embedding operation (ChromaDB index build or query).

    Logged with role "retriever" and kind "embedding" so downstream analysis can
    exclude it from decode work -- embeddings are a forward pass with no token
    generation, and counting them as LLM decode would inflate GPU-second totals.

    Token counts are null: ollama's embedding responses come back through
    ChromaDB's embedding function, which does not surface prompt_eval_count.
    This is operation-level timing, not per-embedding-call.
    """
    run = _get_run()
    if run is None:
        yield
        return
    call_id = uuid.uuid4().hex[:12]
    t_start = time.time()
    try:
        yield
    finally:
        run.write({
            "run_id": run.run_id, "task_id": _task_id.get(), "call_id": call_id,
            "agent_role": "retriever", "kind": "embedding", "op": op,
            "attempt_index": _attempt.get(), "replan_index": _replan.get(),
            "parent_call_ids": [],
            "prompt_tokens": None, "output_tokens": None,
            "prompt_eval_duration": None, "eval_duration": None,
            "t_start": t_start, "t_end": time.time(),
            "model": model, "n_documents": n_documents,
        })
        _record("retriever", call_id)
