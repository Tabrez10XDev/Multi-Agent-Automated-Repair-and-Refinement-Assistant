"""
Build the derived TraceLab workload files (DECISIONS 37).

    ./venv/bin/python -I experiments/build_tracelab.py [--n 80] [--raw PATH] [--out DIR]
    ./venv/bin/python -I experiments/build_tracelab.py --fresh-sealed 80
    ./venv/bin/python -I experiments/build_tracelab.py --stratified 160 --split dev --name dev3

STDLIB ONLY, NO REPO IMPORTS, run as a path with -I. The raw file is untrusted
downloaded data: this script only reads it (streamed, never extracted to disk,
nothing in it is executed) after checking its SHA-256 against the value
pre-registered in DECISIONS 37. The simulator never reads the raw file; it reads
the derived files this writes:

    data/external/tracelab/dev.jsonl.gz      80 dev turns (users with first byte
    data/external/tracelab/sealed.jsonl.gz   of sha256("tracelab-split:"+user)
                                             < 128 are dev) -- the sealed file is
                                             written but the SIMULATOR may only
                                             load it behind MAARA_POLICIES_FROZEN
    data/external/tracelab/dev_constants.json  estimator constants, dev ONLY
    data/external/tracelab/sealed2.jsonl.gz  --fresh-sealed only (DECISIONS 38)
    data/external/tracelab/{dev3,sealed3}.jsonl.gz  --stratified only (DECISIONS 39)

Each .jsonl.gz: line 1 is a provenance header {"header": {...}}, then one
compact row per turn. Deterministic: no timestamps or paths in the output, a
fixed gzip mtime, sorted keys, so a rerun is byte-identical.

RULES (DECISIONS 37; choices where it is silent are marked CHOICE)
  cleaning   rows with null input_tokens_total / output_tokens dropped
             (CHOICE: a null prefix_tokens / newly_append_tokens is dropped the
             same way, counted separately; none exist in v0.0.2); exact
             duplicates of (provider, session_id, round_index, round_id) keep
             the first in file order.
  order      rounds within a session by (round_index, file order).
  turns      TraceLab's rule: a round opens a turn if it has a user_message
             timing event at or before its earliest model-output event
             (reasoning / text / tool_call; a round with no model-output event
             opens if it has a user_message). Rounds before a session's first
             trigger are dropped. Turn ordinal is 0-BASED within the session,
             counted over all of the session's turns (not just sampled ones).
  split      first byte of sha256("tracelab-split:" + user) < 128 -> dev.
  sample     per split, the N turns with the smallest
             sha256("tracelab-sample:" + session_id + ":" + ordinal) hex digest.
  clipping   output <= 8192; append <= 32768 - output; prefix <= 32768 - output
             - append. Raw values are kept beside the clipped ones.
  tool time  latest result_at - earliest emitted_at over the round's tools,
             capped at 600 s. CHOICE: if ANY tool of the round lacks emitted_at
             or result_at, the round falls back to max over its tools of the
             effective latency (internal, else wall), counted; if even that is
             absent the tool time is 0.0 (counted as tool_no_info). A negative
             span is clamped to 0.
  epoch      epoch += 1 when prefix_tokens < 0.5 x the previous round's
             input_tokens_total (pre-clip values), over the session's cleaned
             rounds in order, including pre-trigger rounds (CHOICE: the previous
             round is the previous one in the session, whichever turn it is in).
  constants  CHOICE: dev_constants.json is computed from ALL dev-side turns (the
             whole dev population of users, clipped values), not only the 80
             sampled, so the estimator's R distribution is not the sample it
             is later run on. Sealed users never enter it.
  sealed2    --fresh-sealed N (DECISIONS 38): the sealed users' turns in the
             same sample-digest order, minus the first --n (the turns E1 ran
             on) and minus every other turn of their sessions; the first N of
             what is left. Writes sealed2.jsonl.gz and nothing else, and stops
             unless the first --n match the committed sealed.jsonl.gz turn for
             turn and every other file in the output directory is unchanged.
"""

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
import sys

RELEASE = "v0.0.2"
RAW_NAME = "syfi_coding_trace.jsonl.gz"
RAW_SHA256 = "11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65"
DEFAULT_RAW = os.path.join("~", "maara_external", "tracelab-v0.0.2", RAW_NAME)
CTX = 32768
MAX_OUT = 8192
TOOL_CAP_S = 600.0
EPOCH_FRAC = 0.5
MODEL_EVENTS = ("reasoning", "text", "tool_call")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_hash(path):
    """Git's own object hash of this script (content-addressed, so it does not
    depend on commits and keeps reruns byte-identical)."""
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _ts(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def opens_turn(events):
    user = [t for t in (_ts(e.get("timestamp")) for e in events
                        if e.get("event_type") == "user_message") if t is not None]
    if not user:
        return False
    out = [t for t in (_ts(e.get("timestamp")) for e in events
                       if e.get("event_type") in MODEL_EVENTS) if t is not None]
    return min(user) <= min(out) if out else True


def tool_seconds(tools):
    """(seconds or None, source). source: span | fallback | none | None."""
    if not tools:
        return None, None
    emits = [_ts(t.get("emitted_at")) for t in tools]
    results = [_ts(t.get("result_at")) for t in tools]
    if all(x is not None for x in emits) and all(x is not None for x in results):
        return max(0.0, max(results) - min(emits)), "span"
    lat = []
    for t in tools:
        ms = t.get("tool_internal_latency_ms")
        if ms is None:
            ms = t.get("tool_wall_latency_ms")
        if ms is not None:
            lat.append(ms / 1000.0)
    if lat:
        return max(lat), "fallback"
    return 0.0, "none"


def clip(prefix, append, out):
    o = min(out, MAX_OUT)
    a = min(append, CTX - o)
    p = min(prefix, CTX - o - a)
    return p, a, o


def read_rows(raw, c):
    """Stream the file; keep only what the build needs, per session."""
    sessions = collections.defaultdict(list)
    seen = set()
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            c["rows_read"] += 1
            if r.get("input_tokens_total") is None or r.get("output_tokens") is None:
                c["dropped_null_input_or_output"] += 1
                continue
            if r.get("prefix_tokens") is None or r.get("newly_append_tokens") is None:
                c["dropped_null_prefix_or_append"] += 1
                continue
            key = (r["provider"], r["session_id"], r["round_index"], r["round_id"])
            if key in seen:
                c["dropped_duplicate"] += 1
                continue
            seen.add(key)
            sec, src = tool_seconds(r.get("tools") or [])
            sessions[(r["provider"], r["session_id"])].append({
                "ri": r["round_index"], "fi": idx, "user": r["user"],
                "inp": r["input_tokens_total"], "pre": r["prefix_tokens"],
                "app": r["newly_append_tokens"], "out": r["output_tokens"],
                "opens": opens_turn(r.get("timing_events") or []),
                "tool_raw": sec, "tool_src": src})
    return sessions


def build_turns(sessions, c):
    """-> list of turn dicts (all sessions, all turns)."""
    turns = []
    for (provider, sid), rows in sessions.items():
        rows.sort(key=lambda x: (x["ri"], x["fi"]))
        user = rows[0]["user"]
        if any(x["user"] != user for x in rows):
            c["sessions_with_mixed_users"] += 1
        epoch, prev_inp = 0, None
        for x in rows:
            if prev_inp is not None and x["pre"] < EPOCH_FRAC * prev_inp:
                epoch += 1
            x["epoch"] = epoch
            prev_inp = x["inp"]
        started, ordinal, cur = False, -1, None
        for x in rows:
            if x["opens"]:
                started = True
                ordinal += 1
                cur = {"provider": provider, "session_id": sid, "user": user,
                       "ordinal": ordinal, "rounds": []}
                turns.append(cur)
            elif not started:
                c["dropped_pre_trigger_rounds"] += 1
                continue
            cur["rounds"].append(x)
    return turns


def finish_turn(t):
    """Clip, apply the tool cap, and lay the turn out as parallel arrays."""
    rows = t["rounds"]
    cols = {k: [] for k in ("epoch", "prefix", "append", "output", "tool_s")}
    raw = {k: [] for k in ("input_total", "prefix", "append", "output", "tool_s",
                           "tool_src")}
    stats = collections.Counter()
    prev_epoch = None
    for x in rows:
        p, a, o = clip(x["pre"], x["app"], x["out"])
        cols["epoch"].append(x["epoch"])
        cols["prefix"].append(p)
        cols["append"].append(a)
        cols["output"].append(o)
        stats["rounds"] += 1
        if (p, a, o) != (x["pre"], x["app"], x["out"]):
            stats["clipped_rounds"] += 1
            stats["clipped_prefix_tokens_removed"] += x["pre"] - p
            stats["clipped_append_tokens_removed"] += x["app"] - a
            stats["clipped_output_tokens_removed"] += x["out"] - o
        sec = x["tool_raw"]
        if sec is None:
            cols["tool_s"].append(None)
        else:
            stats["tool_nodes"] += 1
            if sec > TOOL_CAP_S:
                stats["tool_nodes_capped"] += 1
            if x["tool_src"] == "fallback":
                stats["tool_fallback"] += 1
            elif x["tool_src"] == "none":
                stats["tool_no_info"] += 1
            cols["tool_s"].append(round(min(sec, TOOL_CAP_S), 3))
        if prev_epoch is not None and x["epoch"] != prev_epoch:
            stats["epoch_resets_within_turn"] += 1
        prev_epoch = x["epoch"]
        raw["input_total"].append(x["inp"])
        raw["prefix"].append(x["pre"])
        raw["append"].append(x["app"])
        raw["output"].append(x["out"])
        raw["tool_s"].append(None if sec is None else round(sec, 3))
        raw["tool_src"].append(x["tool_src"])
    out = {"turn": f"{t['session_id']}#{t['ordinal']}", "provider": t["provider"],
           "session_id": t["session_id"], "ordinal": t["ordinal"], **cols, "raw": raw}
    return out, stats


def digest(kind, *parts):
    return hashlib.sha256((f"tracelab-{kind}:" + ":".join(map(str, parts))).encode()
                          ).hexdigest()


def split_of(user):
    return "dev" if hashlib.sha256(("tracelab-split:" + user).encode()).digest()[0] < 128 \
        else "sealed"


def write_gz(path, header, rows):
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=9) as gz:
            gz.write((json.dumps({"header": header}, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode())
            for r in rows:
                gz.write((json.dumps(r, sort_keys=True, separators=(",", ":"))
                          + "\n").encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out", default=os.path.join(_REPO, "data", "external", "tracelab"))
    ap.add_argument("--n", type=int, default=80, help="turns per split (default 80)")
    ap.add_argument("--fresh-sealed", type=int, default=None, metavar="N",
                    help="write only sealed2.jsonl.gz with N new sealed turns")
    ap.add_argument("--stratified", type=int, default=None, metavar="N",
                    help="write only <name>.jsonl.gz: N turns stratified by rounds (DECISIONS 39)")
    ap.add_argument("--split", choices=("dev", "sealed"))
    ap.add_argument("--name", help="output name for --stratified, e.g. dev3 or sealed3")
    args = ap.parse_args()
    if args.stratified is not None and not (args.split and args.name):
        sys.exit("--stratified needs --split and --name")
    raw = os.path.expanduser(args.raw)
    got = sha256_file(raw)
    if got != RAW_SHA256:
        sys.exit(f"SHA-256 mismatch for {raw}: {got} != pre-registered {RAW_SHA256}")
    build(raw, args.out, args.n, fresh_sealed=args.fresh_sealed,
          stratified=(args.stratified, args.split, args.name) if args.stratified else None)


def _dir_hashes(out_dir, skip):
    return {f: sha256_file(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))
            if f != skip and os.path.isfile(os.path.join(out_dir, f))}


def build_fresh_sealed(pop, header_common, out_dir, n_used, n_new):
    """DECISIONS 38: sealed2.jsonl.gz, the next n_new sealed turns."""
    pop.sort(key=lambda t: digest("sample", t["session_id"], t["ordinal"]))
    used = pop[:n_used]
    committed = os.path.join(out_dir, "sealed.jsonl.gz")
    with gzip.open(committed, "rt", encoding="utf-8") as f:
        f.readline()
        had = [json.loads(line)["turn"] for line in f if line.strip()]
    if had != [f"{t['session_id']}#{t['ordinal']}" for t in used]:
        sys.exit(f"the first {n_used} sealed turns do not match {committed}")
    used_sessions = {(t["provider"], t["session_id"]) for t in used}
    rest = pop[n_used:]
    fresh = [t for t in rest if (t["provider"], t["session_id"]) not in used_sessions]
    chosen = fresh[:n_new]
    if len(chosen) < n_new:
        sys.exit(f"only {len(chosen)} fresh sealed turns, {n_new} requested")
    rows, agg = [], collections.Counter()
    for t in chosen:
        row, st = finish_turn(t)
        rows.append(row)
        agg.update(st)
    users_used = {t["user"] for t in used}
    users_new = {t["user"] for t in chosen}
    counts = {
        "turns_in_split": len(pop), "turns_used_by_sealed": len(used),
        "turns_skipped_same_session": len(rest) - len(fresh),
        "sessions_skipped": len(used_sessions),
        "turns_sampled": len(chosen), "rounds_sampled": agg["rounds"],
        "users_sampled": len(users_new),
        "users_sampled_also_in_sealed": len(users_new & users_used),
        "sampled": dict(sorted(agg.items())),
        "providers_sampled": dict(sorted(collections.Counter(
            t["provider"] for t in chosen).items()))}
    hdr = dict(header_common, split="sealed2", n_turns_requested=n_new,
               spec="docs/DECISIONS.md sections 37 and 38", n_used_by_sealed=n_used,
               counts=counts)
    before = _dir_hashes(out_dir, "sealed2.jsonl.gz")
    write_gz(os.path.join(out_dir, "sealed2.jsonl.gz"), hdr, rows)
    if _dir_hashes(out_dir, "sealed2.jsonl.gz") != before:
        sys.exit("an existing file changed while writing sealed2.jsonl.gz")
    print(f"sealed2: sampled={len(chosen)} rounds={agg['rounds']} "
          f"users={len(users_new)} (also in sealed: {len(users_new & users_used)}) "
          f"skipped_same_session={counts['turns_skipped_same_session']} "
          f"clipped_rounds={agg['clipped_rounds']} capped={agg['tool_nodes_capped']}")
    return counts


R_CAP = 116            # DECISIONS 39: smallest r with >= 99.5% of dev turns at or below
SEALED2_CANDIDATES = 80   # first sealed2 build: its aggregate was seen (DECISIONS 38)


def _used_turns(pop, out_dir, split, n_used):
    """Turns of `pop` (sorted by sample hash) already written to a committed
    file of this split: dev.jsonl.gz, or sealed.jsonl.gz plus the first
    SEALED2_CANDIDATES fresh turns of DECISIONS 38's sealed2 builder."""
    used = pop[:n_used]
    committed = os.path.join(out_dir, f"{split}.jsonl.gz")
    with gzip.open(committed, "rt", encoding="utf-8") as f:
        f.readline()
        had = [json.loads(line)["turn"] for line in f if line.strip()]
    if had != [f"{t['session_id']}#{t['ordinal']}" for t in used]:
        sys.exit(f"the first {n_used} {split} turns do not match {committed}")
    if split == "sealed":
        sess = {(t["provider"], t["session_id"]) for t in used}
        fresh = [t for t in pop[n_used:] if (t["provider"], t["session_id"]) not in sess]
        used = used + fresh[:SEALED2_CANDIDATES]
    return used


def build_stratified(pop, header_common, out_dir, split, name, n_used, n_new):
    """DECISIONS 39: <name>.jsonl.gz, n_new turns of `split` stratified by R.

    Pool: the split's turns minus every turn already used (see _used_turns)
    and every session they belong to, then R <= R_CAP. Sorted by (R, h) with
    h = sha256("tracelab-sample3:" + session + ":" + ordinal) and cut into
    n_new equal-count strata; each stratum gives its smallest-h turn whose
    session has no pick yet. Rows are written in h order, not by R, because
    the adapter replays turns in file order before any shuffle."""
    pop.sort(key=lambda t: digest("sample", t["session_id"], t["ordinal"]))
    used = _used_turns(pop, out_dir, split, n_used)
    used_ids = {(t["provider"], t["session_id"], t["ordinal"]) for t in used}
    used_sessions = {(t["provider"], t["session_id"]) for t in used}
    left = [t for t in pop if (t["provider"], t["session_id"]) not in used_sessions]
    assert not any((t["provider"], t["session_id"], t["ordinal"]) in used_ids for t in left)
    capped = [t for t in left if len(t["rounds"]) > R_CAP]
    pool = [t for t in left if len(t["rounds"]) <= R_CAP]
    h = {id(t): digest("sample3", t["session_id"], t["ordinal"]) for t in pool}
    pool.sort(key=lambda t: (len(t["rounds"]), h[id(t)]))
    M = len(pool)
    if M < n_new:
        sys.exit(f"only {M} turns in the {split} pool, {n_new} requested")
    chosen, picked_sessions, skips = [], set(), 0
    for i in range(n_new):
        stratum = sorted(pool[i * M // n_new:(i + 1) * M // n_new], key=lambda t: h[id(t)])
        for t in stratum:
            if (t["provider"], t["session_id"]) not in picked_sessions:
                chosen.append(t)
                picked_sessions.add((t["provider"], t["session_id"]))
                break
            skips += 1
        else:
            sys.exit(f"stratum {i} has no turn from an unpicked session")
    chosen.sort(key=lambda t: h[id(t)])
    rows, agg = [], collections.Counter()
    for t in chosen:
        row, st = finish_turn(t)
        rows.append(row)
        agg.update(st)
    rs = [len(t["rounds"]) for t in chosen]
    mean_r = sum(rs) / len(rs)
    cv_r = (sum((r - mean_r) ** 2 for r in rs) / len(rs)) ** 0.5 / mean_r
    users_used = {t["user"] for t in used}
    users_new = {t["user"] for t in chosen}
    counts = {
        "turns_in_split": len(pop), "turns_excluded_used": len(used),
        "sessions_excluded": len(used_sessions),
        "turns_excluded_same_session": len(pop) - len(left),
        "turns_removed_by_cap": len(capped),
        "rounds_removed_by_cap": sum(len(t["rounds"]) for t in capped),
        "rounds_left_before_cap": sum(len(t["rounds"]) for t in left),
        "pool_turns": M, "r_cap": R_CAP, "strata": n_new,
        "skips_one_per_session": skips,
        "turns_sampled": len(chosen), "rounds_sampled": agg["rounds"],
        "rounds_per_turn_mean": mean_r, "rounds_per_turn_cv": cv_r,
        "users_sampled": len(users_new),
        "users_sampled_also_in_used": len(users_new & users_used),
        "sampled": dict(sorted(agg.items())),
        "providers_sampled": dict(sorted(collections.Counter(
            t["provider"] for t in chosen).items()))}
    hdr = dict(header_common, split=name, n_turns_requested=n_new,
               spec="docs/DECISIONS.md sections 37 and 39", n_used=len(used),
               counts=counts)
    fname = f"{name}.jsonl.gz"
    if os.path.exists(os.path.join(out_dir, fname)):
        sys.exit(f"{fname} exists; derived data is never overwritten")
    before = _dir_hashes(out_dir, fname)
    write_gz(os.path.join(out_dir, fname), hdr, rows)
    if _dir_hashes(out_dir, fname) != before:
        sys.exit(f"an existing file changed while writing {fname}")
    print(f"{name}: sampled={len(chosen)} rounds={agg['rounds']} R mean={mean_r:.2f} "
          f"cv={cv_r:.2f} users={len(users_new)} pool={M} capped={len(capped)} "
          f"skips={skips}")
    return counts


def build(raw, out_dir, n_req, fresh_sealed=None, stratified=None):
    """Everything after the SHA check (tests call this on a hand-built file)."""

    counts = collections.Counter({k: 0 for k in (
        "rows_read", "dropped_null_input_or_output", "dropped_null_prefix_or_append",
        "dropped_duplicate", "dropped_pre_trigger_rounds",
        "sessions_with_mixed_users")})
    sessions = read_rows(raw, counts)
    turns = build_turns(sessions, counts)
    counts["sessions"] = len(sessions)
    counts["turns_total"] = len(turns)

    by_split = {"dev": [], "sealed": []}
    users = {"dev": set(), "sealed": set()}
    for t in turns:
        s = split_of(t["user"])
        by_split[s].append(t)
        users[s].add(t["user"])

    os.makedirs(out_dir, exist_ok=True)
    script = os.path.abspath(__file__)
    header_common = {
        "source_file": RAW_NAME, "source_sha256": sha256_file(raw), "release": RELEASE,
        "spec": "docs/DECISIONS.md section 37",
        "build_script": "experiments/build_tracelab.py",
        "build_script_git_hash": git_blob_hash(script),
        "n_turns_requested": n_req, "ctx_cap": CTX, "max_output": MAX_OUT,
        "tool_cap_s": TOOL_CAP_S, "epoch_fraction": EPOCH_FRAC,
        "turn_ordinal_base": 0,
        "global_counts": dict(sorted(counts.items()))}
    if fresh_sealed is not None:
        return build_fresh_sealed(by_split["sealed"], header_common, out_dir, n_req,
                                  fresh_sealed)
    if stratified is not None:
        n_new, split, name = stratified
        return build_stratified(by_split[split], header_common, out_dir, split, name,
                                n_req, n_new)

    dev_pop = None
    for name in ("dev", "sealed"):
        pop = by_split[name]
        pop.sort(key=lambda t: digest("sample", t["session_id"], t["ordinal"]))
        chosen = pop[:n_req]
        rows, agg = [], collections.Counter()
        for t in chosen:
            row, st = finish_turn(t)
            rows.append(row)
            agg.update(st)
        pop_stats = collections.Counter()
        for t in pop:
            _, st = finish_turn(t)
            pop_stats.update(st)
        if name == "dev":
            dev_pop = pop
        counts_split = {
            "users": len(users[name]),
            "turns_in_split": len(pop),
            "rounds_in_split": pop_stats["rounds"],
            "turns_sampled": len(chosen),
            "rounds_sampled": agg["rounds"],
            "sampled": dict(sorted(agg.items())),
            "split_population": dict(sorted(pop_stats.items())),
            "providers_sampled": dict(sorted(collections.Counter(
                t["provider"] for t in chosen).items()))}
        hdr = dict(header_common, split=name, counts=counts_split)
        write_gz(os.path.join(out_dir, f"{name}.jsonl.gz"), hdr, rows)
        print(f"{name}: users={len(users[name])} turns_in_split={len(pop)} "
              f"sampled={len(chosen)} rounds={agg['rounds']} "
              f"clipped_rounds={agg['clipped_rounds']} "
              f"tool_nodes={agg['tool_nodes']} capped={agg['tool_nodes_capped']} "
              f"fallback={agg['tool_fallback']}")

    # Estimator constants: DEV split only, whole dev population, clipped values.
    r_counts = collections.Counter()
    n_rounds = s_pre = s_app = s_out = 0
    s_tool = 0.0
    for t in dev_pop:
        row, _ = finish_turn(t)
        k = len(row["epoch"])
        r_counts[k] += 1
        for i in range(k):
            n_rounds += 1
            s_pre += row["prefix"][i]
            s_app += row["append"][i]
            s_out += max(1, row["output"][i])
            s_tool += row["tool_s"][i] or 0.0
    consts = {
        "source": "dev-side users only (DECISIONS 37); whole dev population, "
                  "clipped values; no sealed data",
        "build_script_git_hash": header_common["build_script_git_hash"],
        "n_turns": len(dev_pop), "n_rounds": n_rounds,
        "rounds_per_turn_counts": {str(k): v for k, v in sorted(r_counts.items())},
        "mean_prefix_tokens": s_pre / n_rounds,
        "mean_append_tokens": s_app / n_rounds,
        "mean_output_tokens": s_out / n_rounds,
        "mean_tool_s_per_round": s_tool / n_rounds}
    with open(os.path.join(out_dir, "dev_constants.json"), "w") as f:
        json.dump(consts, f, indent=1, sort_keys=True)
        f.write("\n")
    print("global:", dict(sorted(counts.items())))
    print("dev constants:", {k: v for k, v in consts.items()
                             if k.startswith("mean_") or k.startswith("n_")})


if __name__ == "__main__":
    main()
