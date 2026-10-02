def group_by(items, key_func):
    groups = {}
    for item in items:
        key = key_func(item)
        groups[key] = [item]  # bug: overwrites instead of appending
    return groups
