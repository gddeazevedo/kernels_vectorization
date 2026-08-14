#include <ilu0.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

static inline void matmat_batch8(__m512d C[9], const __m512d A[9], const __m512d B[9]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            __m512d acc = _mm512_mul_pd  (A[r*3 + 0], B[0*3 + c]);
            acc         = _mm512_fmadd_pd(A[r*3 + 1], B[1*3 + c], acc);
            acc         = _mm512_fmadd_pd(A[r*3 + 2], B[2*3 + c], acc);
            C[r*3 + c]  = acc;
        }
    }
}

static inline void matsub_batch8(__m512d D[9], const __m512d A[9], const __m512d B[9]) {
    for (int e = 0; e < 9; e++) D[e] = _mm512_sub_pd(A[e], B[e]);
}

static inline void pack_blocks(__m512d out[9], const double *vals, __m512i voff, __mmask8 m) {
    const __m512d zero = _mm512_setzero_pd();
    for (int e = 0; e < 9; e++) {
        __m512i idx = _mm512_add_epi64(voff, _mm512_set1_epi64(e));
        out[e] = _mm512_mask_i64gather_pd(zero, m, idx, vals, sizeof(double));
    }
}

static inline void unpack_blocks(double *vals, __m512i voff, const __m512d in[9], __mmask8 m) {
    for (int e = 0; e < 9; e++) {
        __m512i idx = _mm512_add_epi64(voff, _mm512_set1_epi64(e));
        _mm512_mask_i64scatter_pd(vals, m, idx, in[e], sizeof(double));
    }
}

static inline void broadcast_block(__m512d out[9], const double *blk) {
    for (int e = 0; e < 9; e++) out[e] = _mm512_set1_pd(blk[e]);
}

static inline void process_tile(double *vals,
                                const int64_t *off_ij, const int64_t *off_kj, int n) {
    __mmask8 m  = (__mmask8)((1u << n) - 1);
    __m512i vij = _mm512_loadu_si512((const void *)off_ij);
    __m512i vkj = _mm512_loadu_si512((const void *)off_kj);

    __m512d Bij[9], Bkj[9], P[9], R[9];

    pack_blocks(Bij, vals, vij, m);
    pack_blocks(Bkj, vals, vkj, m);

    matmat_batch8(P, Bij, Bkj);     // prod = block_ij * block_kj  (igual à base)
    matsub_batch8(R, Bij, P);       // block_ij - prod
    unpack_blocks(vals, vij, R, m);
}

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

void ilu0_decomposition_batch8(BlockedCSR &A) {
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
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

            int64_t off_ij[8], off_kj[8];
            int cnt = 0;

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                off_ij[cnt] = (int64_t) q * bs2;
                off_kj[cnt] = block_kj - A.vals;

                if (++cnt == 8) {
                    process_tile(A.vals, off_ij, off_kj, cnt);
                    cnt = 0;
                }
            }

            if (cnt) {
                process_tile(A.vals, off_ij, off_kj, cnt);
            }
        }
    }

    free(prod);
    free(inv);
}