from case_08 import flatten_list

def test_flat(): assert flatten_list([1, 2, 3]) == [1, 2, 3]
def test_one_level(): assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]
def test_nested(): assert flatten_list([1, [2, [3, [4]]]]) == [1, 2, 3, 4]
def test_empty(): assert flatten_list([]) == []
