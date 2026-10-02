from case_10 import remove_duplicates

def test_with_dups(): assert remove_duplicates([1,2,2,3,3,3]) == [1,2,3]
def test_no_dups(): assert remove_duplicates([1,2,3]) == [1,2,3]
def test_all_same(): assert remove_duplicates([5,5,5]) == [5]
def test_empty(): assert remove_duplicates([]) == []
