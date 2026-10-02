def fibonacci(n):
    if n == 0: return 1  # bug: should return 0
    if n == 1: return 0  # bug: should return 1
    return fibonacci(n-1) + fibonacci(n-2)
