def create_matrix(rows, cols, fill=0):
    return [[fill] * cols for _ in range(rows)]

def matrix_add(a, b):
    rows, cols = len(a), len(a[0])
    return [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]

def matrix_transpose(m):
    rows, cols = len(m), len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]

def matrix_multiply(a, b):
    rows_a, cols_b, cols_a = len(a), len(b[0]), len(a[0])
    result = create_matrix(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result

def print_matrix(m, label=""):
    if label:
        print(f"{label}:")
    for row in m:
        print("  ", row)

if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    print_matrix(a, "A")
    print_matrix(b, "B")
    print_matrix(matrix_add(a, b), "A + B")
    print_matrix(matrix_multiply(a, b), "A * B")
    print_matrix(matrix_transpose(a), "A^T")
