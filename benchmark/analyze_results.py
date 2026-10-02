# benchmark/analyze_results.py
# Run this after experiments to compute pass@1 per configuration
# Usage: python benchmark/analyze_results.py

import json
import os
from collections import defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = defaultdict(list)

with open(os.path.join(_REPO, "data", "outcomes", "results.jsonl")) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)   # note: json.loads not json.load
        key = (r["configuration"], r["prompting_strategy"])
        results[key].append(r["passed"])

print("\n" + "="*60)
print("MAARA RESULTS SUMMARY")
print("="*60)

for (config, strategy), passes in sorted(results.items()):
    pass_at_1 = sum(passes) / len(passes) * 100
    print(f"{config:<25} | {strategy:<15} | pass@1: {pass_at_1:5.1f}%  (n={len(passes)})")

print("="*60)