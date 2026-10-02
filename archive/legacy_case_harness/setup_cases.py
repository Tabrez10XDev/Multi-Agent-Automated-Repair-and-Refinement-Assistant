import os

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")
os.makedirs(CASES_DIR, exist_ok=True)

cases = {
    "case_01.py": '''def add(a, b):
    return a - b  # bug: should be a + b
''',
    "case_01_tests.py": '''from case_01 import add

def test_add_positive(): assert add(2, 3) == 5
def test_add_zero(): assert add(0, 5) == 5
def test_add_negative(): assert add(-1, 1) == 0
def test_add_both_negative(): assert add(-2, -3) == -5
''',
    "case_02.py": '''def multiply(a, b):
    result = 0
    for _ in range(b):
        result == result + a  # bug: == instead of =
    return result
''',
    "case_02_tests.py": '''from case_02 import multiply

def test_multiply_basic(): assert multiply(3, 4) == 12
def test_multiply_zero(): assert multiply(5, 0) == 0
def test_multiply_one(): assert multiply(7, 1) == 7
def test_multiply_large(): assert multiply(10, 10) == 100
''',
    "case_03.py": '''def is_even(n):
    return n % 2 != 0  # bug: should be == 0
''',
    "case_03_tests.py": '''from case_03 import is_even

def test_even_number(): assert is_even(4) == True
def test_odd_number(): assert is_even(3) == False
def test_zero(): assert is_even(0) == True
def test_negative_even(): assert is_even(-2) == True
def test_negative_odd(): assert is_even(-3) == False
''',
    "case_04.py": '''def get_first_element(lst):
    return lst[0]  # bug: crashes on empty list
''',
    "case_04_tests.py": '''from case_04 import get_first_element

def test_normal_list(): assert get_first_element([1, 2, 3]) == 1
def test_single_element(): assert get_first_element([42]) == 42
def test_empty_list(): assert get_first_element([]) is None
def test_string_list(): assert get_first_element(["a", "b"]) == "a"
''',
    "case_05.py": '''def celsius_to_fahrenheit(c):
    return c * 9 + 32  # bug: should be (c * 9/5) + 32
''',
    "case_05_tests.py": '''from case_05 import celsius_to_fahrenheit

def test_freezing(): assert celsius_to_fahrenheit(0) == 32.0
def test_boiling(): assert celsius_to_fahrenheit(100) == 212.0
def test_body_temp(): assert abs(celsius_to_fahrenheit(37) - 98.6) < 0.1
def test_negative(): assert celsius_to_fahrenheit(-40) == -40.0
''',
    "case_06.py": '''def count_words(s):
    return len(s)  # bug: counts characters not words
''',
    "case_06_tests.py": '''from case_06 import count_words

def test_basic(): assert count_words("hello world") == 2
def test_single(): assert count_words("hello") == 1
def test_multiple(): assert count_words("one two three four") == 4
def test_empty(): assert count_words("") == 0
''',
    "case_07.py": '''def fibonacci(n):
    if n == 0: return 1  # bug: should return 0
    if n == 1: return 0  # bug: should return 1
    return fibonacci(n-1) + fibonacci(n-2)
''',
    "case_07_tests.py": '''from case_07 import fibonacci

def test_fib_0(): assert fibonacci(0) == 0
def test_fib_1(): assert fibonacci(1) == 1
def test_fib_5(): assert fibonacci(5) == 5
def test_fib_10(): assert fibonacci(10) == 55
''',
    "case_08.py": '''def flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)  # bug: doesn't recurse
        else:
            result.append(item)
    return result
''',
    "case_08_tests.py": '''from case_08 import flatten_list

def test_flat(): assert flatten_list([1, 2, 3]) == [1, 2, 3]
def test_one_level(): assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]
def test_nested(): assert flatten_list([1, [2, [3, [4]]]]) == [1, 2, 3, 4]
def test_empty(): assert flatten_list([]) == []
''',
    "case_09.py": '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            right = mid - 1  # bug: should be left = mid + 1
        else:
            left = mid + 1   # bug: should be right = mid - 1
    return -1
''',
    "case_09_tests.py": '''from case_09 import binary_search

def test_found_middle(): assert binary_search([1,2,3,4,5], 3) == 2
def test_found_first(): assert binary_search([1,2,3,4,5], 1) == 0
def test_found_last(): assert binary_search([1,2,3,4,5], 5) == 4
def test_not_found(): assert binary_search([1,2,3,4,5], 6) == -1
''',
    "case_10.py": '''def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
''',
    "case_10_tests.py": '''from case_10 import remove_duplicates

def test_with_dups(): assert remove_duplicates([1,2,2,3,3,3]) == [1,2,3]
def test_no_dups(): assert remove_duplicates([1,2,3]) == [1,2,3]
def test_all_same(): assert remove_duplicates([5,5,5]) == [5]
def test_empty(): assert remove_duplicates([]) == []
''',
    "case_11.py": '''def is_palindrome(s):
    return s == s[::-1]  # bug: doesn't normalize spaces/case
''',
    "case_11_tests.py": '''from case_11 import is_palindrome

def test_simple(): assert is_palindrome("racecar") == True
def test_not_palindrome(): assert is_palindrome("hello") == False
def test_with_spaces(): assert is_palindrome("race car") == True
def test_mixed_case(): assert is_palindrome("Race Car") == True
def test_empty(): assert is_palindrome("") == True
''',
    "case_12.py": '''def group_by(items, key_func):
    groups = {}
    for item in items:
        key = key_func(item)
        groups[key] = [item]  # bug: overwrites instead of appending
    return groups
''',
    "case_12_tests.py": '''from case_12 import group_by

def test_basic():
    result = group_by([1,2,3,4,5,6], lambda x: x % 2)
    assert set(result[0]) == {2, 4, 6}
    assert set(result[1]) == {1, 3, 5}

def test_strings():
    result = group_by(["cat","car","bar","bat"], lambda x: x[0])
    assert set(result["c"]) == {"cat","car"}
    assert set(result["b"]) == {"bar","bat"}

def test_single():
    result = group_by([1], lambda x: x)
    assert result[1] == [1]
''',
    "case_13.py": '''def calculate_grade(score):
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 50: return "D"  # bug: should be >= 60
    return "F"
''',
    "case_13_tests.py": '''from case_13 import calculate_grade

def test_a(): assert calculate_grade(95) == "A"
def test_b(): assert calculate_grade(85) == "B"
def test_c(): assert calculate_grade(75) == "C"
def test_d(): assert calculate_grade(65) == "D"
def test_f(): assert calculate_grade(55) == "F"
def test_boundary_b(): assert calculate_grade(80) == "B"
def test_boundary_c(): assert calculate_grade(70) == "C"
def test_boundary_d(): assert calculate_grade(60) == "D"
''',
    "case_14.py": '''def merge_sorted_lists(l1, l2):
    result = []
    i = j = 0
    while i < len(l1) and j < len(l2):
        if l1[i] <= l2[j]:
            result.append(l1[i])
            i += 1
        else:
            result.append(l2[j])
            j += 1
    return result  # bug: missing remainder
''',
    "case_14_tests.py": '''from case_14 import merge_sorted_lists

def test_basic(): assert merge_sorted_lists([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
def test_empty_first(): assert merge_sorted_lists([],[1,2,3]) == [1,2,3]
def test_empty_second(): assert merge_sorted_lists([1,2,3],[]) == [1,2,3]
def test_unequal_lengths(): assert merge_sorted_lists([1,2],[3,4,5,6]) == [1,2,3,4,5,6]
def test_duplicates(): assert merge_sorted_lists([1,2,2],[2,3]) == [1,2,2,2,3]
''',
    "case_15.py": '''def parse_csv_line(line):
    return line.split(",")  # bug: doesn't handle quoted fields
''',
    "case_15_tests.py": '''from case_15 import parse_csv_line

def test_simple(): assert parse_csv_line("a,b,c") == ["a","b","c"]
def test_quoted(): assert parse_csv_line('"hello, world",foo,bar') == ["hello, world","foo","bar"]
def test_empty_field(): assert parse_csv_line("a,,c") == ["a","","c"]
def test_single(): assert parse_csv_line("hello") == ["hello"]
''',
    "case_16.py": '''from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=True)  # bug: should be last=False
''',
    "case_16_tests.py": '''from case_16 import LRUCache

def test_basic_eviction():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    assert c.get(1) == -1
    assert c.get(2) == 2
    assert c.get(3) == 3

def test_get_updates_order():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.get(1)
    c.put(3, 3)
    assert c.get(2) == -1
    assert c.get(1) == 1
''',
    "case_17.py": '''def tokenize(text):
    return text.split(" ")  # bug: multiple spaces, punctuation, empty string
''',
    "case_17_tests.py": '''from case_17 import tokenize

def test_basic(): assert tokenize("hello world") == ["hello", "world"]
def test_empty(): assert tokenize("") == []
def test_multiple_spaces(): assert tokenize("hello  world") == ["hello", "world"]
def test_punctuation(): assert tokenize("hello, world!") == ["hello", "world"]
def test_single_word(): assert tokenize("hello") == ["hello"]
''',
    "case_18.py": '''def dfs(graph, start):
    visited = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited
''',
    "case_18_tests.py": '''from case_18 import dfs

def test_basic():
    graph = {1: [2, 3], 2: [4], 3: [4], 4: []}
    assert dfs(graph, 1) == [1, 3, 4, 2]

def test_single_node():
    graph = {1: []}
    assert dfs(graph, 1) == [1]

def test_cycle():
    graph = {1: [2], 2: [3], 3: [1]}
    result = dfs(graph, 1)
    assert set(result) == {1, 2, 3}
''',
    "case_19.py": '''import time

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def is_allowed(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c >= self.period]
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
''',
    "case_19_tests.py": '''from case_19 import RateLimiter
import time

def test_allows_within_limit():
    rl = RateLimiter(3, 1.0)
    assert rl.is_allowed() == True
    assert rl.is_allowed() == True
    assert rl.is_allowed() == True

def test_blocks_over_limit():
    rl = RateLimiter(3, 1.0)
    rl.is_allowed()
    rl.is_allowed()
    rl.is_allowed()
    assert rl.is_allowed() == False

def test_resets_after_period():
    rl = RateLimiter(2, 0.1)
    rl.is_allowed()
    rl.is_allowed()
    time.sleep(0.15)
    assert rl.is_allowed() == True
''',
    "case_20.py": '''def evaluate(expression):
    tokens = expression.split()
    result = int(tokens[0])
    i = 1
    while i < len(tokens):
        op = tokens[i]
        val = int(tokens[i+1])
        if op == "+": result += val
        elif op == "-": result -= val
        elif op == "*": result *= val
        elif op == "/": result //= val
        i += 2
    return result
''',
    "case_20_tests.py": '''from case_20 import evaluate

def test_addition(): assert evaluate("2 + 3") == 5
def test_multiplication(): assert evaluate("3 * 4") == 12
def test_precedence(): assert evaluate("2 + 3 * 4") == 14
def test_complex(): assert evaluate("10 - 2 * 3") == 4
def test_division(): assert evaluate("10 / 2") == 5
''',
}

for filename, content in cases.items():
    path = os.path.join(CASES_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"Created {filename}")

print(f"\nDone. Created {len(cases)} files in {CASES_DIR}")