import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

@timer
def fast_sum(n):
    return n * (n - 1) // 2

if __name__ == "__main__":
    n = 10_000_000
    result1 = slow_sum(n)
    result2 = fast_sum(n)
    print(f"Results match: {result1 == result2}")
