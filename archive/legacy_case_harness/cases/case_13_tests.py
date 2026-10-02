from case_13 import calculate_grade

def test_a(): assert calculate_grade(95) == "A"
def test_b(): assert calculate_grade(85) == "B"
def test_c(): assert calculate_grade(75) == "C"
def test_d(): assert calculate_grade(65) == "D"
def test_f(): assert calculate_grade(55) == "F"
def test_boundary_b(): assert calculate_grade(80) == "B"
def test_boundary_c(): assert calculate_grade(70) == "C"
def test_boundary_d(): assert calculate_grade(60) == "D"
