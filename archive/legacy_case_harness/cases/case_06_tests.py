from case_06 import count_words

def test_basic(): assert count_words("hello world") == 2
def test_single(): assert count_words("hello") == 1
def test_multiple(): assert count_words("one two three four") == 4
def test_empty(): assert count_words("") == 0
