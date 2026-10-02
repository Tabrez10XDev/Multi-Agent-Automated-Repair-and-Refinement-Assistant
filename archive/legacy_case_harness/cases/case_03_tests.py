from case_03 import is_even

def test_even_number(): assert is_even(4) == True
def test_odd_number(): assert is_even(3) == False
def test_zero(): assert is_even(0) == True
def test_negative_even(): assert is_even(-2) == True
def test_negative_odd(): assert is_even(-3) == False
