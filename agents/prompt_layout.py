"""
Prompt layout, switchable between the original and a shared-prefix-first form.

WHY
---
Phase 2 measured that the median reusable prefix across a real run is 4 tokens,
while 37.8% of all prompt tokens are content repeated from an earlier call in
the same task. The gap is caused by ordering, not by content: every role's
prompt opens with its own system message, so two calls carrying the identical
issue text and test files diverge at character 10, on "You are a ".

24.0% of all prompt tokens -- 87.1 seconds of prefill -- are repeated but
unreachable by any prefix cache for that reason alone.

THE CHANGE
----------
Under MAARA_PROMPT_LAYOUT=shared_first every chat call is built as:

    system:  one message, byte-identical for every role
    user:    [code block]  [tests block]  [issue block]  [role-specific section]

Block order is by how many roles carry each block, most-shared first, because a
prefix match ends at the first difference:

    common system   all four roles
    code            evaluator, coder, debugger
    tests           evaluator, coder
    issue           all four, but rendered from different source material
    role section    by definition unique

NO ROLE GAINS OR LOSES INFORMATION. A role that does not receive the test files
today still does not receive them; its tests block is simply absent. Only the
order and the system message change.

WHAT STILL CANNOT SHARE, AND WHY
--------------------------------
  * issue_parser runs before a parsed issue exists, so its issue block is the
    raw issue text while evaluator's and coder's are rendered from the parsed
    dict. Different bytes, unavoidably.
  * evaluator's issue block carries intent + affected_files; coder's carries
    intent + expected_behavior + test_hints. Rendering them identically would
    hand each role fields it does not receive today, which rule 2 forbids.
  * debugger receives no issue text and no test files today.

Roles that could share more if they were given more (recorded, not done):
  * debugger could take the test files, which it currently never sees, and
    would then share the whole code+tests head with coder.
  * evaluator and coder could take one identical rendering of the full parsed
    issue, which would extend their shared head past the tests block.
"""

import os

# Deliberately generic: it must not hint at any one role's job, or roles would
# need different wording and the shared head would end at the system message.
SHARED_SYSTEM = (
    "You are an expert Python software engineer working on an automated "
    "bug-fixing task. Read the context below, then follow the instructions in "
    "the final section exactly."
)

ORIGINAL = "original"
SHARED_FIRST = "shared_first"


def layout() -> str:
    """Active layout. Defaults to the original so untouched runs are unchanged."""
    v = (os.environ.get("MAARA_PROMPT_LAYOUT") or ORIGINAL).strip().lower()
    return SHARED_FIRST if v == SHARED_FIRST else ORIGINAL


def is_shared_first() -> bool:
    return layout() == SHARED_FIRST


# --- canonical blocks ----------------------------------------------------
#
# Every role renders a given block through the same function, so identical
# material produces identical bytes. That is the whole mechanism: a prefix
# cache matches bytes, not intent.

def code_block(files: dict) -> str:
    if not files:
        return ""
    out = ["### CODE"]
    for name in sorted(files):
        out.append(f"\n{name}:\n```python\n{files[name]}\n```")
    return "".join(out) + "\n"


def tests_block(test_files: dict) -> str:
    if not test_files:
        return ""
    out = ["### TESTS"]
    for name in sorted(test_files):
        out.append(f"\n{name}:\n```python\n{test_files[name]}\n```")
    return "".join(out) + "\n"


def issue_block(text: str) -> str:
    if not text:
        return ""
    return f"### ISSUE\n{text}\n"


def role_block(role: str, body: str) -> str:
    return f"### TASK: {role}\n{body}"


def compose(code=None, tests=None, issue=None, role=None, role_body=""):
    """
    Assemble a shared-first prompt. Returns (system_message, user_prompt).

    Absent blocks are omitted entirely rather than emitted empty, so a role
    that has no test files does not carry a "### TESTS" header that would
    itself break the match for the next role along.
    """
    parts = [code_block(code or {}), tests_block(tests or {}), issue_block(issue or "")]
    parts.append(role_block(role or "work", role_body))
    return SHARED_SYSTEM, "\n".join(p for p in parts if p)
