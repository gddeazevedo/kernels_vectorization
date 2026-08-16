#include "bcsr.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

void invert_matrix(const double *M, double *M_inv) {
    double C00 = M[idx(1,1)] * M[idx(2,2)] - M[idx(1,2)] * M[idx(2,1)];
    double C01 = M[idx(1,2)] * M[idx(2,0)] - M[idx(1,0)] * M[idx(2,2)];
    double C02 = M[idx(1,0)] * M[idx(2,1)] - M[idx(1,1)] * M[idx(2,0)];
    double det = M[idx(0,0)] * C00 + M[idx(0,1)] * C01 + M[idx(0,2)] * C02;
    M_inv[idx(0,0)] = C00;
    M_inv[idx(1,0)] = C01;
    M_inv[idx(2,0)] = C02;
    M_inv[idx(0,1)] = M[idx(0,2)] * M[idx(2,1)] - M[idx(0,1)] * M[idx(2,2)];
    M_inv[idx(1,1)] = M[idx(0,0)] * M[idx(2,2)] - M[idx(0,2)] * M[idx(2,0)];
    M_inv[idx(2,1)] = M[idx(0,1)] * M[idx(2,0)] - M[idx(0,0)] * M[idx(2,1)];
    M_inv[idx(0,2)] = M[idx(0,1)] * M[idx(1,2)] - M[idx(0,2)] * M[idx(1,1)];
    M_inv[idx(1,2)] = M[idx(0,2)] * M[idx(1,0)] - M[idx(0,0)] * M[idx(1,2)];
    M_inv[idx(2,2)] = M[idx(0,0)] * M[idx(1,1)] - M[idx(0,1)] * M[idx(1,0)];
    for (int i = 0; i < BS * BS; i++) M_inv[i] /= det;
}

void matmat(const double *A, const double *B, double *C) {
    for (int i = 0; i < BS; i++)
        for (int j = 0; j < BS; j++)
            for (int k = 0; k < BS; k++)
                C[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
}

void matsub(const double *A, const double *B, double *C) {
    for (int i = 0; i < BS; i++)
        for (int j = 0; j < BS; j++)
            C[idx(i,j)] = A[idx(i,j)] - B[idx(i,j)];
}

void ilu0_decomposition(BlockedCSR *A) {
    double *prod = (double *) calloc(BS * BS, sizeof(double));
    double *diff = (double *) calloc(BS * BS, sizeof(double));
    double *inv  = (double *) calloc(BS * BS, sizeof(double));

    for (int i = 0; i < A->nb; i++) {
        int row_start = A->ia[i];
        int row_end   = A->ia[i + 1];
        for (int p = row_start; p < row_end; p++) {
            int k = A->ja[p];
            if (k >= i) break;
            double *block_ik = &A->vals[(size_t)p * A->bs * A->bs];
            double *diag_kk  = get_block(A, k, k);
            bool allocated_diag = false;
            if (diag_kk == nullptr) {
                diag_kk = (double *) calloc(A->bs * A->bs, sizeof(double));
                diag_kk[0] = 1.0; diag_kk[4] = 1.0; diag_kk[8] = 1.0;
                allocated_diag = true;
            }
            invert_matrix(diag_kk, inv);
            matmat(block_ik, inv, prod);
            memcpy(block_ik, prod, sizeof(double) * A->bs * A->bs);
            memset(prod, 0, sizeof(double) * BS * BS);
            if (allocated_diag) free(diag_kk);
            for (int q = p + 1; q < row_end; q++) {
                int j = A->ja[q];
                double *block_kj = get_block(A, k, j);
                if (block_kj == nullptr) continue;
                double *block_ij = &A->vals[(size_t)q * A->bs * A->bs];
                matmat(block_ik, block_kj, prod);
                matsub(block_ij, prod, diff);
                memset(prod, 0, sizeof(double) * BS * BS);
                memcpy(block_ij, diff, sizeof(double) * A->bs * A->bs);
            }
        }
    }
    free(prod); free(diff); free(inv);
}

// Teste correto: verifica que L*U == A nas posicoes dos blocos existentes
// L: blocos abaixo da diagonal (i>j), com identidade na diagonal
// U: blocos na diagonal e acima (i<=j)
void test_block_ilu0() {
    BlockedCSR *A_orig = generate_blocked27_3x3(3, 3, 3);

    // Copia dos valores originais antes do ILU
    double *orig_vals = (double *) malloc(A_orig->nnzb * A_orig->bs * A_orig->bs * sizeof(double));
    memcpy(orig_vals, A_orig->vals, A_orig->nnzb * A_orig->bs * A_orig->bs * sizeof(double));

    // Copia da estrutura para acessar valores originais
    int *orig_ia = (int *) malloc((A_orig->nb + 1) * sizeof(int));
    int *orig_ja = (int *) malloc(A_orig->nnzb * sizeof(int));
    memcpy(orig_ia, A_orig->ia, (A_orig->nb + 1) * sizeof(int));
    memcpy(orig_ja, A_orig->ja, A_orig->nnzb * sizeof(int));

    // Faz ILU0 in-place
    ilu0_decomposition(A_orig);

    int nb = A_orig->nb;
    int bs = A_orig->bs;
    double prod[BS * BS];
    double max_err = 0.0;

    // Para cada bloco (i,j) que existe na matriz original,
    // computa (L*U)(i,j) = sum_k L(i,k)*U(k,j)
    // e compara com A_orig(i,j)
    for (int i = 0; i < nb; i++) {
        for (int p = orig_ia[i]; p < orig_ia[i + 1]; p++) {
            int j = orig_ja[p];
            double *orig_block = &orig_vals[(size_t)p * bs * bs];

            // Computa (L*U)(i,j) = sum_{k} L(i,k) * U(k,j)
            double lu_block[BS * BS];
            memset(lu_block, 0, sizeof(lu_block));

            for (int k = 0; k < nb; k++) {
                // L(i,k): se k < i, esta no resultado ILU; se k == i, identidade; se k > i, zero
                // U(k,j): se k <= j, esta no resultado ILU; se k > j, zero
                // Portanto so contribui quando k <= min(i, j)

                double L_ik[BS * BS];
                double *U_kj = nullptr;

                if (k < i) {
                    double *ptr = get_block(A_orig, i, k);
                    if (ptr == nullptr) continue;
                    memcpy(L_ik, ptr, sizeof(L_ik));
                } else if (k == i) {
                    // L(i,i) = identidade
                    memset(L_ik, 0, sizeof(L_ik));
                    L_ik[0] = 1.0; L_ik[4] = 1.0; L_ik[8] = 1.0;
                } else {
                    continue; // L(i,k) = 0 para k > i
                }

                if (k <= j) {
                    U_kj = get_block(A_orig, k, j);
                    if (U_kj == nullptr) continue;
                } else {
                    continue; // U(k,j) = 0 para k > j
                }

                matmat(L_ik, U_kj, lu_block);
            }

            // Compara com original
            for (int r = 0; r < bs; r++) {
                for (int c = 0; c < bs; c++) {
                    double err = (fabs(orig_block[r * bs + c] - lu_block[r * bs + c])) / (fabs(orig_block[r * bs + c]) + 1e-12);
                    if (err > max_err) max_err = err;
                }
            }
        }
    }

    printf("Erro maximo relativo (L*U vs A nos blocos existentes): %e\n", max_err);
    if (max_err < 1e-10) {
        printf("TESTE OK\n");
    } else {
        printf("TESTE FALHOU\n");
    }

    free(orig_vals);
    free(orig_ia);
    free(orig_ja);
    bc_free(A_orig);
}

int main() {
    test_block_ilu0();
    return 0;
}