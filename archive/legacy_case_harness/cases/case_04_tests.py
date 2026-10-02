from case_04 import get_first_element

def test_normal_list(): assert get_first_element([1, 2, 3]) == 1
def test_single_element(): assert get_first_element([42]) == 42
def test_empty_list(): assert get_first_element([]) is None
def test_string_list(): assert get_first_element(["a", "b"]) == "a"
