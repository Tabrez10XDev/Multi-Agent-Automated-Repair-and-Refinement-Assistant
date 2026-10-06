# Brief 03 — A status index at the top of FINDINGS

**Owns:** `docs/FINDINGS.md` only.
**Must not touch:** anything else. No code.

Read `docs/briefs/README.md` first.

## Why this exists

`docs/FINDINGS.md` is now 1000+ lines across 17 sections, and some of its
numbers are dead. Two sections are retracted outright and several are superseded
by later ones. A reader landing in the middle has no way to tell a live result
from a withdrawn one, and at least one headline figure has already been quoted
after it was corrected.

## What to build

A status table at the **top** of `docs/FINDINGS.md`, before the first section.
One row per section, in section order, with:

| column | content |
|---|---|
| section | number and short title |
| status | `current` / `superseded by §N` / `retracted` |
| one-line current result | what a reader should take away *today*, or the reason it is dead |

The one-line result is the valuable column. Make it a real finding, not a
restatement of the title — a number where there is one.

## The sections, as they stand

`4, 5, 7, 14a, 14b, 14c, 14d, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27`

Determine each status **by reading the sections**, not by guessing. What you can
rely on:

- §17 and §23 are explicitly **retracted** and say so in their own headings.
  §17 is superseded by §19; §23 by §27.
- §19's mechanism was later superseded by §20, though §19's empirical content
  stands — §19 says this itself. Reflect that nuance rather than flattening it.
- §24 through §27 are the current state and the most load-bearing.

Where a section is partly superseded, say which part. Do not mark something
`current` if a later section contradicts it.

## Also fix, while you are in the file

A few numbers appear in more than one place and at least one is stale. Check
these specifically and make them consistent, citing the newest section:

- the Ollama cache hit rate: the standing figure is **about 1 call in 6 on
  disjoint problems**, almost entirely partial prefix reuse; the earlier
  "1 in 3" was roughly half replay artefact (§22)
- the retention headline: recompute is **0.8-1.1% of makespan**, so retention is
  a GPU-work mechanism here and not a latency one (§24)
- the scheduler headline: **no scheduling rule differs significantly** from
  arrival order, at either concurrency limit, with paired CIs over 20 seeds
  (§27)

If you find a fourth inconsistency, fix it and list it in your report.

## Acceptance

- The table is at the top, complete, and every status is justified by the
  section's own text
- No section's body is rewritten. You are adding an index and reconciling
  repeated numbers, not re-editing findings.
- `./venv/bin/python -m pytest -q` still green (it should be untouched by this,
  but run it — one test does read documentation paths)
- Report: the table, plus a list of every inconsistency you found and what you
  changed

## Do not

- Delete or shorten a retracted section. They are kept deliberately: the
  mistakes are instructive and figures were drawn from them.
- Renumber anything. Cross-references in code comments and commit messages
  point at these numbers.
- Add new findings of your own. If you notice something, report it; do not write
  it into FINDINGS.
