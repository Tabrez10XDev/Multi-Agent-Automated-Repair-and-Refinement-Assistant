from case_02 import multiply

def test_multiply_basic(): assert multiply(3, 4) == 12
def test_multiply_zero(): assert multiply(5, 0) == 0
def test_multiply_one(): assert multiply(7, 1) == 7
def test_multiply_large(): assert multiply(10, 10) == 100
