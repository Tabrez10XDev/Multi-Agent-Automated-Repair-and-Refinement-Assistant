def calculate_average(numbers):
    if not numbers:
        return 0.0
    total = sum(numbers)
    return total / len(numbers)

def reverse_string(s):
    return s[::-1]

def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)