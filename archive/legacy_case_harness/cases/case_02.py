def multiply(a, b):
    result = 0
    for _ in range(b):
        result == result + a  # bug: == instead of =
    return result
