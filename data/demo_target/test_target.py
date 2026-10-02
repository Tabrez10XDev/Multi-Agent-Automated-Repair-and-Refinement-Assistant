from target import calculate_average, reverse_string, count_vowels


def test_average_normal():
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0

def test_average_returns_float():
    assert isinstance(calculate_average([1, 2]), float)

def test_average_empty():
    result = calculate_average([])
    assert result == 0.0
    assert isinstance(result, float)  # must be float, not int

def test_average_single():
    assert calculate_average([5]) == 5.0

def test_average_negative():
    assert calculate_average([-1, -2, -3]) == -2.0

def test_reverse_normal():
    assert reverse_string("hello") == "olleh"

def test_reverse_single():
    assert reverse_string("a") == "a"

def test_reverse_empty():
    assert reverse_string("") == ""

def test_reverse_palindrome():
    assert reverse_string("racecar") == "racecar"

def test_reverse_spaces():
    assert reverse_string("hello world") == "dlrow olleh"

def test_vowels_lowercase():
    assert count_vowels("hello") == 2

def test_vowels_uppercase():
    assert count_vowels("HELLO") == 2

def test_vowels_mixed():
    assert count_vowels("HeLLo") == 2

def test_vowels_none():
    assert count_vowels("rhythm") == 0

def test_vowels_all():
    assert count_vowels("aeiouAEIOU") == 10

def test_vowels_empty():
    assert count_vowels("") == 0