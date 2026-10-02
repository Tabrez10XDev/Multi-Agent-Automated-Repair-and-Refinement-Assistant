from case_18 import dfs

def test_basic():
    graph = {1: [2, 3], 2: [4], 3: [4], 4: []}
    assert dfs(graph, 1) == [1, 3, 4, 2]

def test_single_node():
    graph = {1: []}
    assert dfs(graph, 1) == [1]

def test_cycle():
    graph = {1: [2], 2: [3], 3: [1]}
    result = dfs(graph, 1)
    assert set(result) == {1, 2, 3}
