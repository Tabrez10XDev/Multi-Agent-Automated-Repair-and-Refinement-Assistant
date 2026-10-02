def calculate_average(numbers):
    if not numbers:
        raise ValueError("Cannot compute average of an empty list")
    total = sum(numbers)
    return total / len(numbers)

def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

def reverse_string(s):
    return s[::-1]