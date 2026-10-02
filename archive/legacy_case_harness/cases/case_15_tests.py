from case_15 import parse_csv_line

def test_simple(): assert parse_csv_line("a,b,c") == ["a","b","c"]
def test_quoted(): assert parse_csv_line('"hello, world",foo,bar') == ["hello, world","foo","bar"]
def test_empty_field(): assert parse_csv_line("a,,c") == ["a","","c"]
def test_single(): assert parse_csv_line("hello") == ["hello"]
