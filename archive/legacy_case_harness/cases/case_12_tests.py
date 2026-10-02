from case_12 import group_by

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
