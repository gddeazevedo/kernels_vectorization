#include <block_matrix_ops.h>

static void invert_common(double *dst, double &det, const double *M) {
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

void invert_3x3_matrix(double *dst, const double *M) {
    double det;
    invert_common(dst, det, M);

    for (int i = 0; i < BS * BS; i++) {
        dst[i] /= det;
    }
}

void invert_3x3_matrix_omp(double *dst, const double *M) {
    double det;
    invert_common(dst, det, M);

    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst[i] /= det;
    }
}

void invert_3x3_matrix_avx256(double *dst, const double *M) {
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

void invert_3x3_matrix_avx512(double *dst, const double *M) {
    double det;
    invert_common(dst, det, M);

    __m512d vdet = _mm512_set1_pd(det);
    __m512d vdst = _mm512_loadu_pd(dst);
    vdst = _mm512_div_pd(vdst, vdet);
    _mm512_storeu_pd(dst, vdst);

    dst[8] = dst[8] / det;
}

void invert_3x3_matrix_hwy256(double *dst, const double *M) {
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

void invert_3x3_matrix_hwy512(double *dst, const double *M) {
    double det;
    invert_common(dst, det, M);

    const hn::FixedTag<double, 8> d;

    auto vdet = hn::Set(d, det);
    auto vdst = hn::LoadU(d, dst);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst);

    dst[8] = dst[8] / det;
}

void matmat(double *dst, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_omp(double *dst, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int k = 0; k < BS; k++) {
            #pragma omp simd simdlen(3)
            for (int j = 0; j < BS; j++) {
                dst[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_avx256(double *dst, const double *A, const double *B) {
    const __mmask8 k = 0x7; // 0b00000111

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

void matmat_avx512(double *dst, const double *A, const double *B) {
    const __m512i  perm_idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);
    const __m512d  b_raw    = _mm512_loadu_pd(&B[idx(0,0)]);
    const __m512i  b_perm   = _mm512_set_epi64(5, 2, 7, 4, 1, 6, 3, 0);
    const __m512d  b        = _mm512_permutexvar_pd(b_perm, b_raw);

    for (int i = 0; i < BS; i++) {
        __m256d a_256 = _mm256_maskz_loadu_pd(0x7, &A[idx(i, 0)]);
        __m512d a_512 = _mm512_broadcast_f64x4(a_256);
        a_512 = _mm512_permutexvar_pd(perm_idx, a_512);

        __m512d prod = _mm512_mul_pd(a_512, b);

        dst[idx(i, 0)] = _mm512_mask_reduce_add_pd(MASK_SUM_FIRST_3, prod);
        dst[idx(i, 1)] = _mm512_mask_reduce_add_pd(MASK_SUM_MID_3,   prod);
        dst[idx(i, 2)] = _mm512_mask_reduce_add_pd(MASK_SUM_LAST_2,  prod) + A[idx(i, 2)] * B[idx(2, 2)];
    }
}

void matmat_hwy256(double *dst, const double *A, const double *B) {
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

void matmat_hwy512(double *dst, const double *A, const double *B) {
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

void matsub(double *dst, const double *A, const double *B) {
    for (int i = 0; i < BS * BS; i++) {
        dst[i] = A[i] - B[i];
    }
}

void matsub_omp(double *dst, const double *A, const double *B) {
    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst[i] = A[i] - B[i];
    }
}

void matsub_avx256(double *dst, const double *A, const double *B) {
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

void matsub_avx512(double *dst, const double *A, const double *B) {   
    __m512d va   = _mm512_loadu_pd(A);
    __m512d vb   = _mm512_loadu_pd(B);
    __m512d vdst = _mm512_sub_pd(va, vb);
    _mm512_storeu_pd(dst, vdst);
    dst[8] = A[8] - B[8];    
}

void matsub_hwy256(double *dst, const double *A, const double *B) {
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

void matsub_hwy512(double *dst, const double *A, const double *B) {
    const hn::FixedTag<double, 8> d;

    auto va   = hn::LoadU(d, A);
    auto vb   = hn::LoadU(d, B);
    auto vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst);

    dst[8] = A[8] - B[8];
}
