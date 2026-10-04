PY := ./venv/bin/python

# Targets that call a live model server are separated out and gated. Running
# one by accident costs hours, so each requires CONFIRM=1.
.PHONY: help test analyze pass-at-1 grid check-traces validate-cache calibrate probe-cache collect figures

help:
	@echo "Offline (no model server, safe to run anytime):"
	@echo "  make test          full test suite"
	@echo "  make analyze       trace stats + figures for both trace sets"
	@echo "  make pass-at-1     pass@1 per configuration from committed outcomes"
	@echo "  make grid          retention x scheduler grid (quick: 1 util, 1 scale)"
	@echo "  make check-traces  verify every committed trace stayed in its sandbox"
	@echo "  make validate-cache TRACE=<path>   frozen cache rule vs a held-out trace"
	@echo ""
	@echo "Calls the live model (needs Ollama; requires CONFIRM=1):"
	@echo "  make calibrate     re-measure rates and prompt sizes -> data/calibration"
	@echo "  make probe-cache   probe Ollama's prompt cache (slots, capacity, affinity)"
	@echo "  make collect       run the HumanEval tasks and collect a trace"

test:
	$(PY) -m pytest -q

analyze:
	$(PY) -m analysis.analyze_trace --set original
	$(PY) -m analysis.analyze_trace --set shared_first

figures: analyze

pass-at-1:
	$(PY) -m analysis.pass_at_1

grid:
	$(PY) -u -m experiments.run_retention_grid --quick

validate-cache:
	@if [ -z "$(TRACE)" ]; then \
		echo "Usage: make validate-cache TRACE=data/traces/<holdout>.jsonl.gz"; \
		echo "Runs the FROZEN cache rule against a held-out trace. No tuning."; \
		exit 1; \
	fi
	$(PY) -m analysis.validate_cache_model $(TRACE)

check-traces:
	@for t in data/traces/*.prompts.jsonl.gz; do \
		printf '%-50s ' "$$(basename $$t)"; \
		$(PY) -m analysis.check_trace_clean "$$t" | tail -1; \
	done

# ---- model-calling targets -------------------------------------------------

define REQUIRE_CONFIRM
@if [ "$(CONFIRM)" != "1" ]; then \
	echo "This target calls the live model and can run for a long time."; \
	echo "Re-run with CONFIRM=1 if that is what you want:"; \
	echo "    make $@ CONFIRM=1"; \
	exit 1; \
fi
endef

calibrate:
	$(REQUIRE_CONFIRM)
	$(PY) -u -m probes.calibrate

probe-cache:
	$(REQUIRE_CONFIRM)
	$(PY) -u -m probes.ollama_cache_probe --slots --capacity 40 --affinity

collect:
	$(REQUIRE_CONFIRM)
	MAARA_TRACE=1 $(PY) -u -m experiments.collect_humaneval
