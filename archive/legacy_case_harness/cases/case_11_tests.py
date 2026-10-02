from case_11 import is_palindrome

def test_simple(): assert is_palindrome("racecar") == True
def test_not_palindrome(): assert is_palindrome("hello") == False
def test_with_spaces(): assert is_palindrome("race car") == True
def test_mixed_case(): assert is_palindrome("Race Car") == True
def test_empty(): assert is_palindrome("") == True
