#!/bin/zsh
# Sealed runs, DECISIONS 31, in order, each once.
set -e
cd /Users/tabrezlj/Deck/self-healing-agent
export MAARA_POLICIES_FROZEN=f4c889b
SLO=tight:1.0:0.3,standard:2.0:0.4,loose:4.0:0.3
./venv/bin/python -u -m experiments.scheduling_map --trace-set final_qwen --slo-mix $SLO --workers 6 \
  --out results/sealed/scheduling_map_final_qwen.json > results/sealed/scheduling_map_final_qwen.log 2>&1
./venv/bin/python -u -m experiments.run_retention_grid --trace-set final_qwen --scales 1 --headrooms 0.25,0.5,1.0 \
  --util 0.8 --batch 4 --seeds 20 --out results/sealed/retention_headroom_final_qwen.json > results/sealed/retention_headroom_final_qwen.log 2>&1
./venv/bin/python -u -m experiments.scheduling_map --trace-set final_llama --slo-mix $SLO --workers 6 \
  --out results/sealed/scheduling_map_final_llama.json > results/sealed/scheduling_map_final_llama.log 2>&1
echo SEALED_DONE
