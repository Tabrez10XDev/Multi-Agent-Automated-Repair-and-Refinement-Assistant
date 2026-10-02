from case_01 import add

def test_add_positive(): assert add(2, 3) == 5
def test_add_zero(): assert add(0, 5) == 5
def test_add_negative(): assert add(-1, 1) == 0
def test_add_both_negative(): assert add(-2, -3) == -5
