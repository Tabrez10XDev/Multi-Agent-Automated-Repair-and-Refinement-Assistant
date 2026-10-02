from case_17 import tokenize

def test_basic(): assert tokenize("hello world") == ["hello", "world"]
def test_empty(): assert tokenize("") == []
def test_multiple_spaces(): assert tokenize("hello  world") == ["hello", "world"]
def test_punctuation(): assert tokenize("hello, world!") == ["hello", "world"]
def test_single_word(): assert tokenize("hello") == ["hello"]
