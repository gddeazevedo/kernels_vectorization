#include <ilu0.h>

void ilu0_decomposition(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *diff = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    for (int i = 0; i < A.nb; i++) {
        int row_start = A.ia[i];
        int row_end   = A.ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A.ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A.vals[(size_t) p * bs2];
            double *diag_kk  = A.get_block(k, k);
            
            invert_3x3_matrix(inv, diag_kk);
            matmat(prod, block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * bs2);
            memset(prod, 0, sizeof(double) * bs2);

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A.vals[(size_t) q * bs2];

                matmat(prod, block_ij, block_kj);
                matsub(diff, block_ij, prod);
                memset(prod, 0, sizeof(double) * bs2);
                memcpy(block_ij, diff, sizeof(double) * bs2);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}

void ilu0_decomposition_omp(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *diff = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    for (int i = 0; i < A.nb; i++) {
        int row_start = A.ia[i];
        int row_end   = A.ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A.ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A.vals[(size_t) p * bs2];
            double *diag_kk  = A.get_block(k, k);
            
            invert_3x3_matrix_omp(inv, diag_kk);
            matmat_omp(prod, block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * bs2);
            memset(prod, 0, sizeof(double) * bs2);

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A.vals[(size_t) q * bs2];

                matmat_omp(prod, block_ij, block_kj);
                matsub_omp(diff, block_ij, prod);
                memset(prod, 0, sizeof(double) * bs2);
                memcpy(block_ij, diff, sizeof(double) * bs2);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}

void ilu0_decomposition_avx256(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *diff = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    for (int i = 0; i < A.nb; i++) {
        int row_start = A.ia[i];
        int row_end   = A.ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A.ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A.vals[(size_t) p * bs2];
            double *diag_kk  = A.get_block(k, k);
            
            invert_3x3_matrix_avx256(inv, diag_kk);
            matmat_avx256(prod, block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * bs2);
            memset(prod, 0, sizeof(double) * bs2);

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A.vals[(size_t) q * bs2];

                matmat_avx256(prod, block_ij, block_kj);
                matsub_avx256(diff, block_ij, prod);
                memset(prod, 0, sizeof(double) * bs2);
                memcpy(block_ij, diff, sizeof(double) * bs2);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}

void ilu0_decomposition_avx512(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *diff = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    for (int i = 0; i < A.nb; i++) {
        int row_start = A.ia[i];
        int row_end   = A.ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A.ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A.vals[(size_t) p * bs2];
            double *diag_kk  = A.get_block(k, k);
            
            invert_3x3_matrix_avx512(inv, diag_kk);
            matmat_avx512(prod, block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * bs2);
            memset(prod, 0, sizeof(double) * bs2);

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A.vals[(size_t) q * bs2];

                matmat_avx512(prod, block_ij, block_kj);
                matsub_avx512(diff, block_ij, prod);
                memset(prod, 0, sizeof(double) * bs2);
                memcpy(block_ij, diff, sizeof(double) * bs2);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}

void ilu0_decomposition_hwy256(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *diff = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    for (int i = 0; i < A.nb; i++) {
        int row_start = A.ia[i];
        int row_end   = A.ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A.ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A.vals[(size_t) p * bs2];
            double *diag_kk  = A.get_block(k, k);
            
            invert_3x3_matrix_hwy256(inv, diag_kk);
            matmat_hwy256(prod, block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * bs2);
            memset(prod, 0, sizeof(double) * bs2);

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A.vals[(size_t) q * bs2];

                matmat_hwy256(prod, block_ij, block_kj);
                matsub_hwy256(diff, block_ij, prod);
                memset(prod, 0, sizeof(double) * bs2);
                memcpy(block_ij, diff, sizeof(double) * bs2);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}

void ilu0_decomposition_hwy512(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *diff = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    for (int i = 0; i < A.nb; i++) {
        int row_start = A.ia[i];
        int row_end   = A.ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A.ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A.vals[(size_t) p * bs2];
            double *diag_kk  = A.get_block(k, k);
            
            invert_3x3_matrix_hwy512(inv, diag_kk);
            matmat_hwy512(prod, block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * bs2);
            memset(prod, 0, sizeof(double) * bs2);

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A.vals[(size_t) q * bs2];

                matmat_hwy512(prod, block_ij, block_kj);
                matsub_hwy512(diff, block_ij, prod);
                memset(prod, 0, sizeof(double) * bs2);
                memcpy(block_ij, diff, sizeof(double) * bs2);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}