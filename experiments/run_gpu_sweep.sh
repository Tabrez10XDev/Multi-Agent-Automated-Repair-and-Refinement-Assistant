#!/usr/bin/env bash
# Batching-cost sweep pre-registered in DECISIONS 33.
#   PROFILE=mac_ollama_measured experiments/run_gpu_sweep.sh   # pass 1
#   PROFILE=gpu_5060ti          experiments/run_gpu_sweep.sh   # pass 2 (needs G1 committed)
# Runs one job at a time, stops on the first failure; outputs in results/gpu_sweep/.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-mac_ollama_measured}"
OUT="results/gpu_sweep/$PROFILE"
mkdir -p "$OUT"
PY="caffeinate -i ./venv/bin/python -u -m"

$PY experiments.sweep_decode_coeff --profile "$PROFILE" --coeffs 0.15 0.3 0.5 0.841 \
    --seeds 20 --out "$OUT/sweep_decode_coeff.json" 2>&1 | tee "$OUT/sweep_decode_coeff.log"

for c in 0.841 0.5 0.3 0.15; do
    $PY experiments.scheduling_map --profile "$PROFILE" --trace-set holdout_qwen \
        --batch-decode-coeff "$c" --load-axis capacity --kv-mb 1024 --retention lru \
        --batches 2 4 8 16 --seeds 20 --deadline-basis estimate --workers 8 \
        --slo-mix tight:1.0:0.3,standard:2.0:0.4,loose:4.0:0.3 \
        --out "$OUT/map_c$c.json" 2>&1 | tee "$OUT/map_c$c.log"
done
echo "sweep done -> $OUT"
