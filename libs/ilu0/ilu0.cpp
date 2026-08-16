#include <ilu0.h>

static void invert_common(double *dst, double &det, const double *M)
{
    double C00 = M[idx(1,1)] * M[idx(2,2)] - M[idx(1,2)] * M[idx(2,1)];
    double C01 = M[idx(1,2)] * M[idx(2,0)] - M[idx(1,0)] * M[idx(2,2)];
    double C02 = M[idx(1,0)] * M[idx(2,1)] - M[idx(1,1)] * M[idx(2,0)];

    det = M[idx(0,0)] * C00 + M[idx(0,1)] * C01 + M[idx(0,2)] * C02;

    dst[idx(0,0)] = C00;
    dst[idx(1,0)] = C01;
    dst[idx(2,0)] = C02;
    dst[idx(0,1)] = M[idx(0,2)] * M[idx(2,1)] - M[idx(0,1)] * M[idx(2,2)];
    dst[idx(1,1)] = M[idx(0,0)] * M[idx(2,2)] - M[idx(0,2)] * M[idx(2,0)];
    dst[idx(2,1)] = M[idx(0,1)] * M[idx(2,0)] - M[idx(0,0)] * M[idx(2,1)];
    dst[idx(0,2)] = M[idx(0,1)] * M[idx(1,2)] - M[idx(0,2)] * M[idx(1,1)];
    dst[idx(1,2)] = M[idx(0,2)] * M[idx(1,0)] - M[idx(0,0)] * M[idx(1,2)];
    dst[idx(2,2)] = M[idx(0,0)] * M[idx(1,1)] - M[idx(0,1)] * M[idx(1,0)];
}

static void transpose(double *dst, const double *M)
{
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            dst[idx(i,j)] = M[idx(j,i)];
        }
    }
}

static inline void gather_blocks_avx512(
    __m512d dst[BS2],
    const double *blocks,
    __m512i offsets,
    __mmask8 mask
)
{
    const __m512d zeros = _mm512_setzero_pd();

    for (int reg = 0; reg < BS2; reg++) {
        __m512i indices = _mm512_add_epi64(offsets, _mm512_set1_epi64(reg));
        dst[reg] = _mm512_mask_i64gather_pd(zeros, mask, indices, blocks, sizeof(double));
    }
}

static inline void scatter_blocks_avx512(
    double *blocks,
    const __m512d src[BS2],
    __m512i offsets,
    __mmask8 mask
)
{
    for (int reg = 0; reg < BS2; reg++) {
        __m512i indices = _mm512_add_epi64(offsets, _mm512_set1_epi64(reg));
        _mm512_mask_i64scatter_pd(blocks, mask, indices, src[reg], sizeof(double));
    }
}

static inline void matmat_avx512(__m512d dst[BS2], const __m512d A[BS2], const __m512d B[BS2])
{
    for (int row = 0; row < BS; row++) {
        for (int col = 0; col < BS; col++) {
            __m512d acc = _mm512_mul_pd  (A[idx(row, 0)], B[idx(0, col)]);
            acc         = _mm512_fmadd_pd(A[idx(row, 1)], B[idx(1, col)], acc);
            acc         = _mm512_fmadd_pd(A[idx(row, 2)], B[idx(2, col)], acc);
            dst[idx(row, col)] = acc;
        }
    }
}

static inline void matsub_avx512(__m512d dst[BS2], const __m512d A[BS2], const __m512d B[BS2])
{
    for (int reg = 0; reg < BS2; reg++) {
        dst[reg] = _mm512_sub_pd(A[reg], B[reg]);
    }
}

static inline void process_blocks_avx512(
    double *blocks,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
)
{
    __mmask8 mask = (__mmask8) ((1u << n_blocks) - 1);

    __m512i voffsets_i = _mm512_loadu_si512(offsets_i);
    __m512i voffsets_k = _mm512_loadu_si512(offsets_k);

    __m512d Bij[BS2];
    __m512d Bkj[BS2];
    __m512d prod[BS2];
    __m512d diff[BS2];

    gather_blocks_avx512(Bij, blocks, voffsets_i, mask);
    gather_blocks_avx512(Bkj, blocks, voffsets_k, mask);

    matmat_avx512(prod, Bij, Bkj);
    matsub_avx512(diff, Bij, prod);

    scatter_blocks_avx512(blocks, diff, voffsets_i, mask);
}

static void invert_3x3_matrix(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    for (int i = 0; i < BS * BS; i++) {
        dst[i] /= det;
    }
}

static void matmat(double *dst, const double *A, const double *B)
{
    double Bt[BS * BS];

    transpose(Bt, B);

    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst[idx(i, j)] += A[idx(i, k)] * Bt[idx(j, k)];
            }
        }
    }
}

static void matsub(double *dst, const double *A, const double *B)
{
    for (int i = 0; i < BS * BS; i++) {
        dst[i] = A[i] - B[i];
    }
}

static void invert_3x3_matrix_omp(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst[i] /= det;
    }
}

static void matmat_omp(double *dst, const double *A, const double *B)
{
    for (int i = 0; i < BS; i++) {
        for (int k = 0; k < BS; k++) {
            #pragma omp simd simdlen(3)
            for (int j = 0; j < BS; j++) {
                dst[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

static void matsub_omp(double *dst, const double *A, const double *B)
{
    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst[i] = A[i] - B[i];
    }
}

static void invert_3x3_matrix_avx256(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    __m256d vdet = _mm256_set1_pd(det);
    __m256d vdst = _mm256_loadu_pd(dst);
    vdst = _mm256_div_pd(vdst, vdet);
    _mm256_storeu_pd(dst, vdst);

    vdst = _mm256_loadu_pd(&dst[4]);
    vdst = _mm256_div_pd(vdst, vdet);
    _mm256_storeu_pd(&dst[4], vdst);

    dst[8] = dst[8] / det;
}

static void matmat_avx256(double *dst, const double *A, const double *B)
{
    const __mmask8 k = 0x7;

    const __m256d b0 = _mm256_maskz_loadu_pd(k, &B[idx(0,0)]);
    const __m256d b1 = _mm256_maskz_loadu_pd(k, &B[idx(1,0)]);
    const __m256d b2 = _mm256_maskz_loadu_pd(k, &B[idx(2,0)]);

    for (int i = 0; i < BS; i++) {
        __m256d sum = _mm256_setzero_pd();

        sum = _mm256_fmadd_pd(_mm256_set1_pd(A[idx(i,0)]), b0, sum);
        sum = _mm256_fmadd_pd(_mm256_set1_pd(A[idx(i,1)]), b1, sum);
        sum = _mm256_fmadd_pd(_mm256_set1_pd(A[idx(i,2)]), b2, sum);

        _mm256_mask_storeu_pd(&dst[idx(i,0)], k, sum);
    }
}

static void matsub_avx256(double *dst, const double *A, const double *B)
{
    __m256d va   = _mm256_loadu_pd(&A[0]);
    __m256d vb   = _mm256_loadu_pd(&B[0]);
    __m256d vdst = _mm256_sub_pd(va, vb);
    _mm256_storeu_pd(&dst[0], vdst);

    va   = _mm256_loadu_pd(&A[4]);
    vb   = _mm256_loadu_pd(&B[4]);
    vdst = _mm256_sub_pd(va, vb);
    _mm256_storeu_pd(&dst[4], vdst);

    dst[8] = A[8] - B[8];
}

static void invert_3x3_matrix_hwy256(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    const hn::FixedTag<double, 4> d;

    auto vdet = hn::Set(d, det);
    auto vdst = hn::LoadU(d, dst);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst);

    vdst = hn::LoadU(d, dst + 4);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst + 4);

    dst[8] = dst[8] / det;
}

static void matmat_hwy256(double *dst, const double *A, const double *B)
{
    const hn::FixedTag<double, 4> d;
    const auto mask = hn::FirstN(d, 3);

    const auto b0 = hn::MaskedLoad(mask, d, &B[idx(0,0)]);
    const auto b1 = hn::MaskedLoad(mask, d, &B[idx(1,0)]);
    const auto b2 = hn::MaskedLoad(mask, d, &B[idx(2,0)]);

    for (int i = 0; i < BS; i++) {
        auto sum = hn::Zero(d);

        sum = hn::MulAdd(hn::Set(d, A[idx(i,0)]), b0, sum);
        sum = hn::MulAdd(hn::Set(d, A[idx(i,1)]), b1, sum);
        sum = hn::MulAdd(hn::Set(d, A[idx(i,2)]), b2, sum);

        hn::BlendedStore(sum, mask, d, &dst[idx(i,0)]);
    }
}

static void matsub_hwy256(double *dst, const double *A, const double *B)
{
    const hn::FixedTag<double, 4> d;

    auto va   = hn::LoadU(d, A);
    auto vb   = hn::LoadU(d, B);
    auto vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst);

    va = hn::LoadU(d, A + 4);
    vb = hn::LoadU(d, B + 4);
    vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst + 4);

    dst[8] = A[8] - B[8];
}

static void invert_3x3_matrix_hwy512(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    const hn::FixedTag<double, 8> d;

    auto vdet = hn::Set(d, det);
    auto vdst = hn::LoadU(d, dst);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst);

    dst[8] = dst[8] / det;
}

static void matmat_hwy512(double *dst, const double *A, const double *B)
{
    const hn::FixedTag<double, 8> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    HWY_ALIGN const int64_t a_perm_lanes[8] = {0,1,2, 0,1,2, 0,1};
    HWY_ALIGN const int64_t b_perm_lanes[8] = {0,3,6, 1,4,7, 2,5};
    const auto a_perm = hn::IndicesFromVec(d, hn::Load(di, a_perm_lanes));
    const auto b_perm = hn::IndicesFromVec(d, hn::Load(di, b_perm_lanes));

    const auto m_first_3 = hn::FirstN(d, 3);
    const auto m_first_6 = hn::FirstN(d, 6);
    const auto m_mid_3   = hn::AndNot(m_first_3, m_first_6);
    const auto m_last_2  = hn::Not(m_first_6);

    const auto b = hn::TableLookupLanes(hn::LoadU(d, &B[idx(0,0)]), b_perm);

    for (int i = 0; i < BS; i++) {
        auto a = hn::TableLookupLanes(hn::LoadN(d, &A[idx(i,0)], 3), a_perm);
        auto prod = hn::Mul(a, b);

        dst[idx(i, 0)] = hn::MaskedReduceSum(d, m_first_3, prod);
        dst[idx(i, 1)] = hn::MaskedReduceSum(d, m_mid_3,   prod);
        dst[idx(i, 2)] = hn::MaskedReduceSum(d, m_last_2,  prod) + A[idx(i, 2)] * B[idx(2, 2)];
    }
}

static void matsub_hwy512(double *dst, const double *A, const double *B)
{
    const hn::FixedTag<double, 8> d;

    auto va   = hn::LoadU(d, A);
    auto vb   = hn::LoadU(d, B);
    auto vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst);

    dst[8] = A[8] - B[8];
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
                    process_blocks_avx512(A.vals, offsets_i, offsets_k, counter);
                    counter = 0;
                }
            }

            if (counter > 0) {
                process_blocks_avx512(A.vals, offsets_i, offsets_k, counter);
            }
        }
    }

    free(prod);
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