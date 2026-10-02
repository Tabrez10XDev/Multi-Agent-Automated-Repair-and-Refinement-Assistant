from case_20 import evaluate

def test_addition(): assert evaluate("2 + 3") == 5
def test_multiplication(): assert evaluate("3 * 4") == 12
def test_precedence(): assert evaluate("2 + 3 * 4") == 14
def test_complex(): assert evaluate("10 - 2 * 3") == 4
def test_division(): assert evaluate("10 / 2") == 5
