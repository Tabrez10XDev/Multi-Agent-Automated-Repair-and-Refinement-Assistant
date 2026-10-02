def flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)  # bug: doesn't recurse
        else:
            result.append(item)
    return result
