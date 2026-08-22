#include <ilu0.h>

void ilu0_decomposition(BlockedCSR &A)
{
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

                matmat(prod, block_ik, block_kj);
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

void ilu0_decomposition_omp(BlockedCSR &A)
{
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

            int64_t offsets_i[BATCH];
            int64_t offsets_k[BATCH];
            int counter = 0;

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                offsets_i[counter] = (int64_t) q * bs2;
                offsets_k[counter] = block_kj - A.vals;

                counter++;

                if (counter == BATCH) {
                    process_blocks_omp(A.vals, block_ik, offsets_i, offsets_k, counter);
                    counter = 0;
                }
            }

            if (counter > 0) {
                process_blocks_omp(A.vals, block_ik, offsets_i, offsets_k, counter);
            }
        }
    }

    free(prod);
    free(inv);
}

void ilu0_decomposition_avx256(BlockedCSR &A)
{
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    constexpr int batch_size = sizeof(__m256d) / sizeof(double);

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

            int64_t offsets_i[batch_size];
            int64_t offsets_k[batch_size]; 
            int counter = 0;

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                offsets_i[counter] = q * bs2;
                offsets_k[counter] = block_kj - A.vals;

                counter++;

                if (counter == batch_size) {
                    process_blocks_avx256(A.vals, block_ik, offsets_i, offsets_k, counter);
                    counter = 0;
                }
            }

            if (counter > 0) {
                process_blocks_avx256(A.vals, block_ik, offsets_i, offsets_k, counter);
            }
        }
    }

    free(prod);
    free(inv);
}

void ilu0_decomposition_avx512(BlockedCSR &A)
{
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    constexpr int batch_size = sizeof(__m512d) / sizeof(double);

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

            int64_t offsets_i[batch_size];
            int64_t offsets_k[batch_size];
            int counter = 0;

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                offsets_i[counter] = (int64_t) q * bs2;
                offsets_k[counter] = block_kj - A.vals;

                counter++;

                if (counter == batch_size) {
                    process_blocks_avx512(A.vals, block_ik, offsets_i, offsets_k, counter);
                    counter = 0;
                }
            }

            if (counter > 0) {
                process_blocks_avx512(A.vals, block_ik, offsets_i, offsets_k, counter);
            }
        }
    }

    free(prod);
    free(inv);
}

void ilu0_decomposition_hwy256(BlockedCSR &A)
{
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    const hn::FixedTag<double, 4> d;
    const int batch_size = hn::Lanes(d);

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

            int64_t offsets_i[batch_size];
            int64_t offsets_k[batch_size];
            int counter = 0;

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                offsets_i[counter] = q * bs2;
                offsets_k[counter] = block_kj - A.vals;
                counter++;

                if (counter == batch_size) {
                    process_blocks_hwy256(A.vals, block_ik, offsets_i, offsets_k, counter);
                    counter = 0;
                }
            }

            if (counter > 0) {
                process_blocks_hwy256(A.vals, block_ik, offsets_i, offsets_k, counter);
            }
        }
    }

    free(prod);
    free(inv);
}

void ilu0_decomposition_hwy512(BlockedCSR &A)
{
    int bs2 = A.bs * A.bs;

    double *prod = (double *) calloc(bs2, sizeof(double));
    double *inv  = (double *) calloc(bs2, sizeof(double));

    const hn::FixedTag<double, 8> d;
    const int batch_size = hn::Lanes(d);

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

            int64_t offsets_i[batch_size];
            int64_t offsets_k[batch_size];
            int counter = 0;

            for (int q = p + 1; q < row_end; q++) {
                int j = A.ja[q];

                double *block_kj = A.get_block(k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                offsets_i[counter] = (int64_t) q * bs2;
                offsets_k[counter] = block_kj - A.vals;

                counter++;

                if (counter == batch_size) {
                    process_blocks_hwy512(A.vals, block_ik, offsets_i, offsets_k, counter);
                    counter = 0;
                }
            }

            if (counter > 0) {
                process_blocks_hwy512(A.vals, block_ik, offsets_i, offsets_k, counter);
            }
        }
    }

    free(prod);
    free(inv);
}