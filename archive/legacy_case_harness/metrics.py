import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CaseResult:
    case_id: str
    difficulty: str
    success: bool
    attempts: int
    total_tokens: int
    elapsed_seconds: float
    failure_reason: Optional[str] = None


class MetricsTracker:
    def __init__(self):
        self.results: list[CaseResult] = []
        self._start_time = None
        self._attempt_tokens = 0

    def start_case(self):
        self._start_time = time.time()
        self._attempt_tokens = 0

    def record_tokens(self, tokens: int):
        self._attempt_tokens += tokens

    def finish_case(self, case_id: str, difficulty: str,
                    success: bool, attempts: int,
                    failure_reason: str = None) -> CaseResult:
        elapsed = time.time() - self._start_time
        result = CaseResult(
            case_id=case_id,
            difficulty=difficulty,
            success=success,
            attempts=attempts,
            total_tokens=self._attempt_tokens,  # use accumulated tokens
            elapsed_seconds=round(elapsed, 2),
            failure_reason=failure_reason
        )
        self.results.append(result)
        return result

    def summary(self) -> dict:
        if not self.results:
            return {}

        total = len(self.results)
        succeeded = [r for r in self.results if r.success]

        # pass@1 — fixed on first attempt
        pass_at_1 = [r for r in succeeded if r.attempts == 1]

        # by difficulty
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            cases = [r for r in self.results if r.difficulty == diff]
            wins = [r for r in cases if r.success]
            by_diff[diff] = {
                "total": len(cases),
                "passed": len(wins),
                "pass_rate": round(len(wins) / len(cases) * 100, 1) if cases else 0
            }

        return {
            "total_cases": total,
            "total_passed": len(succeeded),
            "pass_at_1": len(pass_at_1),
            "pass_at_3": len(succeeded),
            "pass_at_1_rate": round(len(pass_at_1) / total * 100, 1),
            "pass_at_3_rate": round(len(succeeded) / total * 100, 1),
            "avg_attempts": round(sum(r.attempts for r in succeeded) / len(succeeded), 2) if succeeded else 0,
            "avg_tokens": round(sum(r.total_tokens for r in self.results) / total, 0),
            "avg_seconds": round(sum(r.elapsed_seconds for r in self.results) / total, 2),
            "by_difficulty": by_diff
        }

    def save(self, path: str):
        data = {
            "results": [asdict(r) for r in self.results],
            "summary": self.summary()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Metrics] Saved results to {path}")

    def print_summary(self):
        s = self.summary()
        if not s:
            return

        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(f"Total cases:     {s['total_cases']}")
        print(f"Total passed:    {s['total_passed']}")
        print(f"pass@1:          {s['pass_at_1']}/{s['total_cases']} ({s['pass_at_1_rate']}%)")
        print(f"pass@3:          {s['pass_at_3']}/{s['total_cases']} ({s['pass_at_3_rate']}%)")
        print(f"Avg attempts:    {s['avg_attempts']}")
        print(f"Avg tokens:      {s['avg_tokens']}")
        print(f"Avg time:        {s['avg_seconds']}s")
        print("\nBy difficulty:")
        for diff, stats in s["by_difficulty"].items():
            bar = "█" * stats["passed"] + "░" * (stats["total"] - stats["passed"])
            print(f"  {diff:8s} {bar} {stats['passed']}/{stats['total']} ({stats['pass_rate']}%)")
        print("="*50)