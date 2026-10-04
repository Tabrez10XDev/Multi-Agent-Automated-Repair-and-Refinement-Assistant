"""
pass@1 per configuration, from the committed HumanEval outcome rows.

    python -m analysis.pass_at_1
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.trace_loader import config_label

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    results = defaultdict(list)
    with open(os.path.join(_REPO, "data", "outcomes", "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)   # note: json.loads not json.load
            key = (r["configuration"], r["prompting_strategy"])
            results[key].append(r["passed"])

    print("\n" + "=" * 60)
    print("MAARA RESULTS SUMMARY")
    print("=" * 60)
    for (config, strategy), passes in sorted(results.items()):
        pass_at_1 = sum(passes) / len(passes) * 100
        # Display name only; the stored label stays as written in the data.
        print(f"{config_label(config):<25} | {strategy:<15} | "
              f"pass@1: {pass_at_1:5.1f}%  (n={len(passes)})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
