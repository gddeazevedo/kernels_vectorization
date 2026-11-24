def lu_decomp_1(A):
    n = len(A)

    for i in range(1, n):
        for k in range(i):
            A[i][k] = A[i][k] / A[k][k]

            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]


def lu_decomp_2(A):
    n = len(A[0])
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1

    for i in range(n):
        for j in range(n):
            s = 0

            for k in range(n):
                s += L[i][k] * U[k][j]

            if i <= j:
                U[i][j] = (A[i][j] - s) / L[j][j]
            else:
                L[i][j] = (A[i][j] - s) / U[j][j]

    return L, U


def lu_decomp_3(A):
    n = len(A[0])
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i][j] = A[i][j]
            else:
                L[i][j] = A[i][j]

        L[i][i] = 1

    for i in range(1, n):
        for k in range(i):
            L[i][k] = L[i][k] / U[k][k]

            for j in range(k + 1, i):
                L[i][j] = L[i][j] - L[i][k] * U[k][j]

            for j in range(i, n):
                U[i][j] = U[i][j] - L[i][k] * U[k][j]

    return L, U


def print_matrix(A):
    for i in range(len(A)):
        print(A[i])

    print()


A = [
    [5, 2, 1],
    [3, 1, 4],
    [1, 1, 3]
]


L, U = lu_decomp_2(A)
l, u = lu_decomp_3(A)

print_matrix(L)
print_matrix(U)

print_matrix(l)
print_matrix(u)

lu_decomp_1(A)

print_matrix(A)

