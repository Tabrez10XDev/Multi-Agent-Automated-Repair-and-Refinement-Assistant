import time

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def is_allowed(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c >= self.period]
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
