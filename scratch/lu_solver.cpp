#include <iostream>
#include <vector>

template <typename T>
using matrix_t = std::vector<std::vector<T>>;

struct lu_decomposition_t {
    matrix_t<double> L;
    matrix_t<double> U;
};

std::vector<double> solve_inf(matrix_t<double> A, std::vector<double> b) {
    std::vector<double> x(b.size(), 0.0);

    x[0] = b[0] / A[0][0];

    for (int i = 1; i < A.size(); i++) {
        double sum = 0.0;

        for (int j = 0; j < i; j++) {
            sum += A[i][j] * x[j];
        }

        x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
}

std::vector<double> solve_sup(matrix_t<double> A, std::vector<double> b) {
    int n = b.size();
    std::vector<double> x(n, 0.0);

    x[n - 1] = b[n - 1] / A[n - 1][n - 1];

    for (int i = n - 2; i >= 0; i--) {
        double sum = 0.0;

        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }

        x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
}

lu_decomposition_t lu_decompose(matrix_t<double> A) {
    int n = A.size();
    matrix_t<double> L(n, std::vector<double>(n, 0.0));
    matrix_t<double> U(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;

        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            if (i <= j) {
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }

                U[i][j] = A[i][j] - sum;
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * U[k][j];
                }

                L[i][j] = (A[i][j] - sum) / U[j][j];
            }
        }
    }

    return {L, U};
}

std::vector<double> lu_solve(matrix_t<double> A, std::vector<double> b) {
    lu_decomposition_t lu = lu_decompose(A);
    std::vector<double> y = solve_inf(lu.L, b);
    std::vector<double> x = solve_sup(lu.U, y);

    return x;
}

int main() {
    matrix_t<double> A = {
        {4, -2, 1},
        {3, 6, -4},
        {2, 1, 8}
    };

    std::vector<double> b = {12, -25, 32};

    std::vector<double> x = lu_solve(A, b);

    std::cout << "Solution: ";
    for (double xi : x) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;

    return 0;
}