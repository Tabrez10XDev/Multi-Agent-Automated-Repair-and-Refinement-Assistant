def evaluate(expression):
    tokens = expression.split()
    result = int(tokens[0])
    i = 1
    while i < len(tokens):
        op = tokens[i]
        val = int(tokens[i+1])
        if op == "+": result += val
        elif op == "-": result -= val
        elif op == "*": result *= val
        elif op == "/": result //= val
        i += 2
    return result
