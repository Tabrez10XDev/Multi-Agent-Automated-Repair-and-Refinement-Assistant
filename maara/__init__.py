"""
MAARA — the multi-agent repair pipeline that generates the workload.

`orchestrator.run` is the entry point. Everything it calls lives here:
`agents/` holds the role agents, `sandbox` defines what the coding agent is
allowed to see and touch, `prompt_layout` switches prompt block ordering, and
`tracing` records one JSONL row per LLM call.
"""
