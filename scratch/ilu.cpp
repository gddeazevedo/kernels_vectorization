#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

struct ilu_decomposition_t {
    double **L;
    double **U;
};

double **create_sparse_matrix(int n, double density) {
    double **mat = new double *[n];

    for (int i = 0; i < n; i++) {
        mat[i] = new double[n]();

        for (int j = 0; j < n; j++) {
            if (i == j) {
                mat[i][j] = 10.0; // Diagonal entries
            } else if ((double)std::rand() / RAND_MAX < density) {
                mat[i][j] = (double)(std::rand() % 100 + 1); // Random non-zero off-diagonal entries
            }
        }
    }

    return mat;
}

void print_matrix(double **mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.1f ", mat[i][j]);
        }

        printf("\n");
    }
}

void free_matrix(double **mat, int n) {
    for (int i = 0; i < n; i++) {
        delete[] mat[i];
    }

    delete[] mat;
}

ilu_decomposition_t ilu0_decompose(const double **A, int n) {
    double **L = new double *[n];
    double **U = new double *[n];

    for (int i = 0; i < n; i++) {
        L[i] = new double[n]();
        U[i] = new double[n]();

        for (int j = 0; j < n; j++) {
            if (i <= j) {
                U[i][j] = A[i][j];
            } else {
                L[i][j] = A[i][j];
            }
        }

        L[i][i] = 1.0;
    }

    for (int i = 1; i < n; i++) {
        for (int k = 0; k < i; k++) {
            if (A[i][k] == 0.0) {
                continue;
            }

            L[i][k] = L[i][k] / U[k][k];

            for (int j = k + 1; j < i; j++) {
                if (A[i][j] == 0.0) {
                    continue;
                }

                L[i][j] = L[i][j] - L[i][k] * U[k][j];
            }

            for (int j = i; j < n; j++) {
                if (A[i][j] == 0.0) {
                    continue;
                }

                U[i][j] = U[i][j] - L[i][k] * U[k][j];
            }
        }
    }

    return {L, U};
}

double **matmat(double **A, double **B, int n) {
    double **C = new double *[n];

    for (int i = 0; i < n; i++) {
        C[i] = new double[n]();

        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

bool verify_ilu0(double **A, double **LU, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i][j] == 0.0) {
                continue;
            }

            if (std::abs(A[i][j] - LU[i][j]) > 1e-6) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    std::srand(std::time(nullptr));

    constexpr int n = 10;
    constexpr double density = 0.2;

    double **mat = create_sparse_matrix(n, density);

    printf("Original Matrix:\n");
    print_matrix(mat, n);
    
    ilu_decomposition_t ilu = ilu0_decompose(const_cast<const double **>(mat), n);

    double **L = ilu.L;
    double **U = ilu.U;

    printf("\nL:\n");
    print_matrix(L, n);
    
    printf("\nU:\n");
    print_matrix(U, n);

    double **LU = matmat(L, U, n);

    printf("\nMatrices are the same: %s\n", verify_ilu0(mat, LU, n) ? "Yes" : "No");
    
    free_matrix(mat, n);
    free_matrix(L, n);
    free_matrix(U, n);
    free_matrix(LU, n);

    return 0;
}