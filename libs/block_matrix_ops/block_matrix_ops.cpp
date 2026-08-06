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
    const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    const __m256d b0 = _mm256_maskload_pd(&B[idx(0,0)], mask);
    const __m256d b1 = _mm256_maskload_pd(&B[idx(1,0)], mask);
    const __m256d b2 = _mm256_maskload_pd(&B[idx(2,0)], mask);

    for (int i = 0; i < BS; i++) {
        __m256d sum = _mm256_setzero_pd();

        sum = _mm256_fmadd_pd(_mm256_set1_pd(A[idx(i,0)]), b0, sum);
        sum = _mm256_fmadd_pd(_mm256_set1_pd(A[idx(i,1)]), b1, sum);
        sum = _mm256_fmadd_pd(_mm256_set1_pd(A[idx(i,2)]), b2, sum);

        _mm256_maskstore_pd(&dst[idx(i,0)], mask, sum);
    }
}

void matmat_avx512(double *dst, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
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
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
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
