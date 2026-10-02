def is_palindrome(s):
    return s == s[::-1]  # bug: doesn't normalize spaces/case
