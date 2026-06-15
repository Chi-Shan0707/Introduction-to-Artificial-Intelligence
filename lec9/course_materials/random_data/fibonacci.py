def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def is_fibonacci(num):
    a, b = 0, 1
    while b < num:
        a, b = b, a + b
    return b == num or num == 0

if __name__ == "__main__":
    print("First 15 Fibonacci numbers:", fibonacci(15))
    for x in [0, 1, 4, 5, 13, 20, 21]:
        print(f"{x} is Fibonacci? {is_fibonacci(x)}")
