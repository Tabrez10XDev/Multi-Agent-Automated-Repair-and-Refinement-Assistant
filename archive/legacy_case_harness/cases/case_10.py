def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
