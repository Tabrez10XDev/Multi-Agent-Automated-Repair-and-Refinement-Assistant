from case_19 import RateLimiter
import time

def test_allows_within_limit():
    rl = RateLimiter(3, 1.0)
    assert rl.is_allowed() == True
    assert rl.is_allowed() == True
    assert rl.is_allowed() == True

def test_blocks_over_limit():
    rl = RateLimiter(3, 1.0)
    rl.is_allowed()
    rl.is_allowed()
    rl.is_allowed()
    assert rl.is_allowed() == False

def test_resets_after_period():
    rl = RateLimiter(2, 0.1)
    rl.is_allowed()
    rl.is_allowed()
    time.sleep(0.15)
    assert rl.is_allowed() == True
