# benchmark/run_experiments.py
# Runs all 6 configurations x 3 prompting strategies across 30 HumanEval tasks
# Usage: python benchmark/run_experiments.py

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import results_path, run

TASKS = [
    {
        "task_id": "HumanEval/1",
        "issue": "The separate_paren_groups function does not correctly separate balanced parenthesis groups.",
        "buggy_code": """def separate_paren_groups(paren_string: str):
    result = []
    current = []
    depth = 0
    for ch in paren_string.replace(' ', ''):
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            current.append(ch)
            depth -= 1
            if depth == 0:
                result.append(''.join(current))
                current = []
    return result[1:]
""",
        "test_code": """from target import separate_paren_groups
def test_basic(): assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
def test_simple(): assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
def test_single(): assert separate_paren_groups('(()(())(()))') == ['(()(())(()))']
"""
    },
    {
        "task_id": "HumanEval/2",
        "issue": "The truncate_number function should return only the decimal part of a float but returns the wrong value.",
        "buggy_code": """def truncate_number(number: float) -> float:
    return number % 1.0 + 1.0
""",
        "test_code": """from target import truncate_number
def test_basic(): assert truncate_number(3.5) == 0.5
def test_small(): assert abs(truncate_number(1.33) - 0.33) < 1e-6
def test_large(): assert abs(truncate_number(123.456) - 0.456) < 1e-6
"""
    },
    {
        "task_id": "HumanEval/3",
        "issue": "The below_zero function should return True when balance goes below zero but has inverted return values.",
        "buggy_code": """def below_zero(operations):
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return False
    return True
""",
        "test_code": """from target import below_zero
def test_goes_negative(): assert below_zero([1, 2, -4, 5]) == True
def test_stays_positive(): assert below_zero([1, 2, 3]) == False
def test_empty(): assert below_zero([]) == False
def test_exact_zero(): assert below_zero([1, -1]) == False
"""
    },
    {
        "task_id": "HumanEval/4",
        "issue": "The mean_absolute_deviation function returns the mean instead of the mean absolute deviation.",
        "buggy_code": """def mean_absolute_deviation(numbers):
    mean = sum(numbers) / len(numbers)
    return sum(numbers) / len(numbers)
""",
        "test_code": """from target import mean_absolute_deviation
def test_basic(): assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
def test_same(): assert mean_absolute_deviation([5.0, 5.0, 5.0]) == 0.0
def test_two(): assert abs(mean_absolute_deviation([1.0, 3.0]) - 1.0) < 1e-6
"""
    },
    {
        "task_id": "HumanEval/5",
        "issue": "The intersperse function inserts a delimiter but slices off the first element.",
        "buggy_code": """def intersperse(numbers, delimeter):
    result = []
    for i, n in enumerate(numbers):
        if i != 0:
            result.append(delimeter)
        result.append(n)
    return result[1:]
""",
        "test_code": """from target import intersperse
def test_basic(): assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]
def test_empty(): assert intersperse([], 4) == []
def test_single(): assert intersperse([1], 4) == [1]
def test_two(): assert intersperse([1, 2], 0) == [1, 0, 2]
"""
    },
    {
        "task_id": "HumanEval/6",
        "issue": "The parse_nested_parens function should return max depth per group but always returns 0.",
        "buggy_code": """def parse_nested_parens(paren_string):
    groups = paren_string.split()
    result = []
    for group in groups:
        result.append(0)
    return result
""",
        "test_code": """from target import parse_nested_parens
def test_basic(): assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]
def test_single(): assert parse_nested_parens('((()))') == [3]
"""
    },
    {
        "task_id": "HumanEval/7",
        "issue": "The filter_by_substring function returns all strings instead of filtering by substring.",
        "buggy_code": """def filter_by_substring(strings, substring):
    return strings
""",
        "test_code": """from target import filter_by_substring
def test_basic(): assert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']
def test_empty(): assert filter_by_substring([], 'a') == []
def test_none_match(): assert filter_by_substring(['xyz', 'zzz'], 'a') == []
def test_all_match(): assert filter_by_substring(['aa', 'ab', 'ac'], 'a') == ['aa', 'ab', 'ac']
"""
    },
    {
        "task_id": "HumanEval/8",
        "issue": "The sum_product function returns sum and product in the wrong order.",
        "buggy_code": """def sum_product(numbers):
    product = 1
    for n in numbers:
        product *= n
    return (product, sum(numbers))
""",
        "test_code": """from target import sum_product
def test_basic(): assert sum_product([1, 2, 3, 4]) == (10, 24)
def test_empty(): assert sum_product([]) == (0, 1)
def test_single(): assert sum_product([5]) == (5, 5)
"""
    },
    {
        "task_id": "HumanEval/9",
        "issue": "The rolling_max function returns a flat list of the global max instead of the rolling maximum.",
        "buggy_code": """def rolling_max(numbers):
    if not numbers:
        return []
    m = max(numbers)
    return [m] * len(numbers)
""",
        "test_code": """from target import rolling_max
def test_basic(): assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]
def test_empty(): assert rolling_max([]) == []
def test_decreasing(): assert rolling_max([5, 4, 3, 2, 1]) == [5, 5, 5, 5, 5]
def test_single(): assert rolling_max([3]) == [3]
"""
    },
    {
        "task_id": "HumanEval/10",
        "issue": "The make_palindrome function returns the input doubled instead of the shortest palindrome.",
        "buggy_code": """def is_palindrome(s):
    return s == s[::-1]

def make_palindrome(s):
    return s + s
""",
        "test_code": """from target import make_palindrome
def test_empty(): assert make_palindrome('') == ''
def test_basic(): assert make_palindrome('cat') == 'catac'
def test_already(): assert make_palindrome('cata') == 'catac'
def test_single(): assert make_palindrome('a') == 'a'
"""
    },
    {
        "task_id": "HumanEval/11",
        "issue": "The string_xor function concatenates binary strings instead of XORing them.",
        "buggy_code": """def string_xor(a, b):
    return a + b
""",
        "test_code": """from target import string_xor
def test_basic(): assert string_xor('010', '110') == '100'
def test_zeros(): assert string_xor('000', '000') == '000'
def test_ones(): assert string_xor('111', '111') == '000'
def test_mixed(): assert string_xor('1100', '1010') == '0110'
"""
    },
    {
        "task_id": "HumanEval/12",
        "issue": "The longest function returns the shortest string instead of the longest.",
        "buggy_code": """def longest(strings):
    if not strings:
        return None
    return min(strings, key=len)
""",
        "test_code": """from target import longest
def test_basic(): assert longest(['a', 'bb', 'ccc']) == 'ccc'
def test_empty(): assert longest([]) is None
def test_single(): assert longest(['abc']) == 'abc'
def test_tie(): assert longest(['ab', 'cd', 'eee']) == 'eee'
"""
    },
    {
        "task_id": "HumanEval/13",
        "issue": "The greatest_common_divisor function uses subtraction instead of modulo.",
        "buggy_code": """def greatest_common_divisor(a, b):
    while b:
        a = a - b
        if a < 0:
            a, b = b, a
    return a
""",
        "test_code": """from target import greatest_common_divisor
def test_basic(): assert greatest_common_divisor(3, 5) == 1
def test_divisible(): assert greatest_common_divisor(25, 15) == 5
def test_same(): assert greatest_common_divisor(7, 7) == 7
def test_one(): assert greatest_common_divisor(1, 1000) == 1
"""
    },
    {
        "task_id": "HumanEval/14",
        "issue": "The all_prefixes function only returns the full string instead of all prefixes.",
        "buggy_code": """def all_prefixes(string):
    return [string]
""",
        "test_code": """from target import all_prefixes
def test_basic(): assert all_prefixes('abc') == ['a', 'ab', 'abc']
def test_empty(): assert all_prefixes('') == []
def test_single(): assert all_prefixes('a') == ['a']
def test_four(): assert all_prefixes('abcd') == ['a', 'ab', 'abc', 'abcd']
"""
    },
    {
        "task_id": "HumanEval/15",
        "issue": "The string_sequence function is off by one and does not include n in the output.",
        "buggy_code": """def string_sequence(n):
    return ' '.join(str(i) for i in range(n))
""",
        "test_code": """from target import string_sequence
def test_zero(): assert string_sequence(0) == '0'
def test_three(): assert string_sequence(3) == '0 1 2 3'
def test_one(): assert string_sequence(1) == '0 1'
"""
    },
    {
        "task_id": "HumanEval/16",
        "issue": "The count_distinct_characters function is case-sensitive but should be case-insensitive.",
        "buggy_code": """def count_distinct_characters(string):
    return len(set(string))
""",
        "test_code": """from target import count_distinct_characters
def test_basic(): assert count_distinct_characters('xyzXYZ') == 3
def test_empty(): assert count_distinct_characters('') == 0
def test_all_same(): assert count_distinct_characters('AaAa') == 1
def test_mixed(): assert count_distinct_characters('Hello World') == 7
"""
    },
    {
        "task_id": "HumanEval/17",
        "issue": "The parse_music function should convert note strings to beat counts but returns an empty list.",
        "buggy_code": """def parse_music(music_string):
    return []
""",
        "test_code": """from target import parse_music
def test_basic(): assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
def test_empty(): assert parse_music('') == []
def test_single_o(): assert parse_music('o') == [4]
def test_single_pipe(): assert parse_music('o|') == [2]
def test_single_dot(): assert parse_music('.|') == [1]
"""
    },
    {
        "task_id": "HumanEval/18",
        "issue": "The how_many_times function skips overlapping occurrences of the substring.",
        "buggy_code": """def how_many_times(string, substring):
    count = 0
    i = 0
    while i <= len(string) - len(substring):
        if string[i:i+len(substring)] == substring:
            count += 1
            i += len(substring)
        else:
            i += 1
    return count
""",
        "test_code": """from target import how_many_times
def test_basic(): assert how_many_times('aaa', 'a') == 3
def test_overlap(): assert how_many_times('aaaa', 'aa') == 3
def test_empty(): assert how_many_times('', 'a') == 0
def test_no_match(): assert how_many_times('abc', 'x') == 0
"""
    },
    {
        "task_id": "HumanEval/19",
        "issue": "The sort_numbers function returns the input unchanged instead of sorting number words.",
        "buggy_code": """def sort_numbers(numbers):
    return numbers
""",
        "test_code": """from target import sort_numbers
def test_basic(): assert sort_numbers('three one five') == 'one three five'
def test_empty(): assert sort_numbers('') == ''
def test_single(): assert sort_numbers('zero') == 'zero'
def test_full(): assert sort_numbers('nine eight seven six five four three two one zero') == 'zero one two three four five six seven eight nine'
"""
    },
    {
        "task_id": "HumanEval/20",
        "issue": "The find_closest_elements function returns the two farthest elements instead of the two closest.",
        "buggy_code": """def find_closest_elements(numbers):
    numbers = sorted(numbers)
    return (numbers[0], numbers[-1])
""",
        "test_code": """from target import find_closest_elements
def test_basic(): assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)
def test_equal(): assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)
"""
    },
    {
        "task_id": "HumanEval/21",
        "issue": "The rescale_to_unit function divides by sum instead of the range of values.",
        "buggy_code": """def rescale_to_unit(numbers):
    s = sum(numbers)
    return [n / s for n in numbers]
""",
        "test_code": """from target import rescale_to_unit
def test_basic(): assert rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]
def test_two(): assert rescale_to_unit([1.0, 5.0]) == [0.0, 1.0]
def test_same(): assert rescale_to_unit([2.0, 2.0]) == [0.0, 0.0]
"""
    },
    {
        "task_id": "HumanEval/22",
        "issue": "The filter_integers function does not filter and returns all values including non-integers.",
        "buggy_code": """def filter_integers(values):
    return values
""",
        "test_code": """from target import filter_integers
def test_basic(): assert filter_integers(['a', 3.14, 5, None, [], 2]) == [5, 2]
def test_empty(): assert filter_integers([]) == []
def test_all_int(): assert filter_integers([1, 2, 3]) == [1, 2, 3]
def test_none_int(): assert filter_integers(['a', 'b', []]) == []
"""
    },
    {
        "task_id": "HumanEval/23",
        "issue": "The strlen function returns the list of characters instead of the length.",
        "buggy_code": """def strlen(string):
    return list(string)
""",
        "test_code": """from target import strlen
def test_basic(): assert strlen('hello') == 5
def test_empty(): assert strlen('') == 0
def test_spaces(): assert strlen('hello world') == 11
def test_single(): assert strlen('a') == 1
"""
    },
    {
        "task_id": "HumanEval/24",
        "issue": "The largest_divisor function finds the smallest divisor instead of the largest one below n.",
        "buggy_code": """def largest_divisor(n):
    for i in range(2, n):
        if n % i == 0:
            return i
    return 1
""",
        "test_code": """from target import largest_divisor
def test_basic(): assert largest_divisor(15) == 5
def test_prime(): assert largest_divisor(7) == 1
def test_even(): assert largest_divisor(100) == 50
def test_small(): assert largest_divisor(4) == 2
"""
    },
    {
        "task_id": "HumanEval/25",
        "issue": "The factorize function returns only the first prime factor instead of all prime factors.",
        "buggy_code": """def factorize(n):
    for i in range(2, n + 1):
        if n % i == 0:
            return [i]
    return []
""",
        "test_code": """from target import factorize
def test_basic(): assert factorize(8) == [2, 2, 2]
def test_prime(): assert factorize(7) == [7]
def test_mixed(): assert factorize(12) == [2, 2, 3]
def test_one(): assert factorize(1) == []
"""
    },
    {
        "task_id": "HumanEval/26",
        "issue": "The remove_duplicates function keeps duplicates instead of removing them.",
        "buggy_code": """def remove_duplicates(numbers):
    return numbers
""",
        "test_code": """from target import remove_duplicates
def test_basic(): assert remove_duplicates([1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]
def test_empty(): assert remove_duplicates([]) == []
def test_no_dups(): assert remove_duplicates([1, 2, 3]) == [1, 2, 3]
def test_all_dups(): assert remove_duplicates([1, 1, 1]) == []
"""
    },
    {
        "task_id": "HumanEval/27",
        "issue": "The flip_case function lowercases everything instead of flipping case.",
        "buggy_code": """def flip_case(string):
    return string.lower()
""",
        "test_code": """from target import flip_case
def test_basic(): assert flip_case('Hello') == 'hELLO'
def test_empty(): assert flip_case('') == ''
def test_all_upper(): assert flip_case('ABC') == 'abc'
def test_all_lower(): assert flip_case('abc') == 'ABC'
def test_mixed(): assert flip_case('hElLo') == 'HeLlO'
"""
    },
    {
        "task_id": "HumanEval/28",
        "issue": "The concatenate function joins strings with a space instead of no separator.",
        "buggy_code": """def concatenate(strings):
    return ' '.join(strings)
""",
        "test_code": """from target import concatenate
def test_basic(): assert concatenate(['a', 'b', 'c']) == 'abc'
def test_empty_list(): assert concatenate([]) == ''
def test_single(): assert concatenate(['hello']) == 'hello'
def test_words(): assert concatenate(['hello', 'world']) == 'helloworld'
"""
    },
    {
        "task_id": "HumanEval/29",
        "issue": "The filter_by_prefix function returns all strings instead of only those starting with the prefix.",
        "buggy_code": """def filter_by_prefix(strings, prefix):
    return strings
""",
        "test_code": """from target import filter_by_prefix
def test_basic(): assert filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a') == ['abc', 'array']
def test_empty(): assert filter_by_prefix([], 'a') == []
def test_none_match(): assert filter_by_prefix(['xyz', 'zzz'], 'a') == []
def test_all_match(): assert filter_by_prefix(['ab', 'ac', 'ad'], 'a') == ['ab', 'ac', 'ad']
"""
    },
    {
        "task_id": "HumanEval/30",
        "issue": "The get_positive function returns all numbers including negatives instead of only positives.",
        "buggy_code": """def get_positive(l):
    return l
""",
        "test_code": """from target import get_positive
def test_basic(): assert get_positive([-1, 2, -4, 3, 5]) == [2, 3, 5]
def test_empty(): assert get_positive([]) == []
def test_all_neg(): assert get_positive([-1, -2, -3]) == []
def test_all_pos(): assert get_positive([1, 2, 3]) == [1, 2, 3]
"""
    },
]

TASKS = TASKS[:15]

CONFIGURATIONS = [
    {"config": "single_agent",       "models": {"coder": "qwen2.5-coder:7b"}},
    {"config": "homo_llama3",        "models": {"coder": "llama3.2"}},
    {"config": "hetero_no_rag",      "models": {"coder": "qwen2.5-coder:7b"}},
    {"config": "hetero_no_sentinel", "models": {"coder": "llama3.2"}},
]

PROMPTING_STRATEGIES = ["zero_shot", "few_shot"]
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Each task runs against its own throwaway directory rather than the repo. The
# agent's repo_path IS this directory, so get_file_structure, build_index and
# apply_and_test can only ever see the two files the task is about.
#
# Before this, repo_path was REPO_PATH, which is how a HumanEval run came to
# retrieve and overwrite sim/workload.py. See agents/paths.py.
SANDBOX_DIR = os.path.join(REPO_PATH, "benchmark", "exp_sandbox")


def setup_task_sandbox(task: dict) -> str:
    """
    Build a clean directory holding exactly the two files the task is about.

    That file set is not arbitrary: before sim/ existed, orchestrator's
    available_files filter over the whole repo resolved to exactly
    ['target.py'], and get_test_file_contents to exactly {'test_target.py'}.
    A sandbox containing those two files reproduces that view, so traces stay
    comparable with the original 120 runs.
    """
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR)

    with open(os.path.join(SANDBOX_DIR, "target.py"), "w") as f:
        f.write(task["buggy_code"])
    with open(os.path.join(SANDBOX_DIR, "test_target.py"), "w") as f:
        f.write(task["test_code"])

    return SANDBOX_DIR


def teardown_task_sandbox():
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)


def main():
    os.makedirs(os.path.join(REPO_PATH, "benchmark"), exist_ok=True)

    # load already completed runs so we can skip them on resume
    # Honours MAARA_RESULTS_PATH for BOTH reading and writing, so pointing a
    # traced run at a fresh file also gives it an empty resume set.
    results_file = results_path()
    logged = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    logged.add((r["configuration"], r["prompting_strategy"], r["task_id"]))
                except Exception:
                    continue
        print(f"Resuming — {len(logged)} runs already completed, skipping those.")

    # The grid is already complete in results.jsonl, so resume would skip every
    # combination and a trace run would capture nothing. This forces a re-run.
    # Note it APPENDS duplicate rows to results.jsonl.
    if os.environ.get("MAARA_IGNORE_RESUME"):
        logged = set()
        print("Resume disabled (MAARA_IGNORE_RESUME) — re-running everything in scope.")

    # back up original files
    # No backup/restore of repo files needed any more: tasks run in a sandbox
    # and the repo's own target.py / test_target.py are never written.

    # Optional scoping, for smoke runs and for collecting a trace without
    # paying for the whole grid. All unset => identical behaviour to before.
    tasks = TASKS
    configs = CONFIGURATIONS
    strategies = PROMPTING_STRATEGIES
    if os.environ.get("MAARA_LIMIT_TASKS"):
        tasks = tasks[:int(os.environ["MAARA_LIMIT_TASKS"])]
    if os.environ.get("MAARA_ONLY_CONFIG"):
        configs = [c for c in configs if c["config"] == os.environ["MAARA_ONLY_CONFIG"]]
    if os.environ.get("MAARA_ONLY_STRATEGY"):
        strategies = [s for s in strategies if s == os.environ["MAARA_ONLY_STRATEGY"]]

    total = len(tasks) * len(configs) * len(strategies)
    completed = 0

    print(f"\n{'='*60}")
    print(f"MAARA EXPERIMENT RUNNER")
    print(f"Tasks: {len(tasks)} | Configs: {len(configs)} | Strategies: {len(strategies)}")
    print(f"Total runs: {total}")
    print(f"{'='*60}\n")

    for cfg in configs:
        for strategy in strategies:
            for task in tasks:
                completed += 1
                print(f"\n[{completed}/{total}] Config: {cfg['config']} | Strategy: {strategy} | Task: {task['task_id']}")

                if (cfg["config"], strategy, task["task_id"]) in logged:
                    print(f"  Already completed — skipping.")
                    continue

                try:
                    sandbox = setup_task_sandbox(task)
                    run(
                        issue_text=task["issue"],
                        repo_path=sandbox,
                        test_file="test_target.py",
                        max_attempts=3,
                        config=cfg["config"],
                        prompting_strategy=strategy,
                        task_id=task["task_id"],
                        model=cfg["models"]["coder"]
                    )

                except Exception as e:
                    print(f"  ERROR: {e}")
                    # log the failure so it still shows in results
                    import time
                    error_log = {
                        "configuration": cfg["config"],
                        "prompting_strategy": strategy,
                        "task_id": task["task_id"],
                        "passed": False,
                        "attempts": 0,
                        "evaluator_approved_on_attempt": None,
                        "evaluator_score": None,
                        "rag_chunks_retrieved": 0,
                        "latency_seconds": 0,
                        "error": str(e)
                    }
                    with open(results_file, "a") as f:
                        f.write(json.dumps(error_log) + "\n")

                finally:
                    # always restore originals after each task
                    teardown_task_sandbox()

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {results_file}")
    print(f"Run: python benchmark/analyze_results.py to see pass@1 scores")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()