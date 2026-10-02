from case_14 import merge_sorted_lists

def test_basic(): assert merge_sorted_lists([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
def test_empty_first(): assert merge_sorted_lists([],[1,2,3]) == [1,2,3]
def test_empty_second(): assert merge_sorted_lists([1,2,3],[]) == [1,2,3]
def test_unequal_lengths(): assert merge_sorted_lists([1,2],[3,4,5,6]) == [1,2,3,4,5,6]
def test_duplicates(): assert merge_sorted_lists([1,2,2],[2,3]) == [1,2,2,2,3]
