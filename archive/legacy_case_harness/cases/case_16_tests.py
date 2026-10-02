from case_16 import LRUCache

def test_basic_eviction():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    assert c.get(1) == -1
    assert c.get(2) == 2
    assert c.get(3) == 3

def test_get_updates_order():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.get(1)
    c.put(3, 3)
    assert c.get(2) == -1
    assert c.get(1) == 1
