import random
import string

def generate_password(length=16, use_symbols=True):
    chars = string.ascii_letters + string.digits
    if use_symbols:
        chars += "!@#$%^&*"
    password = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
    ]
    if use_symbols:
        password.append(random.choice("!@#$%^&*"))
    password += [random.choice(chars) for _ in range(length - len(password))]
    random.shuffle(password)
    return "".join(password)

if __name__ == "__main__":
    for i in range(5):
        print(f"Password {i+1}: {generate_password()}")
