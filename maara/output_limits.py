"""
Per-role output-token limits for MAARA's chat calls (docs/DECISIONS.md §32).

Every agent passes options=output_limits.options(<role>) to tracing.chat. A
role with no limit gets None, which is ollama.chat's own default, so its
request is exactly what it was before limits existed.

Values come from the committed traces (data/traces/*.jsonl.gz, output_tokens by
agent_role, the one 40,960-token runaway excluded): twice the role's observed
maximum, rounded up to a multiple of 256.

    role          n     p99    max     2x max -> limit
    issue_parser  759   269    2,286   4,608   APPLIED
    coder         683   358    553     1,280   proposed
    debugger      392   247    373     768     proposed
    evaluator     591   319    372     768     proposed

Only issue_parser is applied: it is the only role with a heavy tail. The others
stay in PROPOSED_NUM_PREDICT until the data shows they are at risk; capping the
coder would truncate code and shift pass@1 for no measured gain.

These limits affect future trace collections only. Committed traces are not
recollected, and data/ is never regenerated.
"""

from __future__ import annotations

NUM_PREDICT = {
    "issue_parser": 4608,
}

# Derived the same way, NOT applied. Move an entry into NUM_PREDICT to apply it,
# and record that in DECISIONS.
PROPOSED_NUM_PREDICT = {
    "coder": 1280,
    "debugger": 768,
    "evaluator": 768,
}


def options(role: str) -> dict | None:
    """The ollama `options` for `role`: {"num_predict": N}, or None if unlimited."""
    n = NUM_PREDICT.get(role)
    return {"num_predict": n} if n is not None else None
