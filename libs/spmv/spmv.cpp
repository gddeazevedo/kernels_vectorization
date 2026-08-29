#include <spmv.h>

void spmv(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y)
{
    int bs = A.bs;

    for (int i = 0; i < A.nb * bs; i++) {
        y[i] = 0.0;
    }

    for (int row = 0; row < A.nb; row++) {
        int row_start = A.brptr[row];
        int row_end   = A.brptr[row + 1];

        double *yrow = &y[row * bs];

        for (int block_idx = row_start; block_idx < row_end; block_idx++) {
            int block_col = A.bcind[block_idx];

            const double *block = &A.bvals[block_idx * bs * bs];
            const double *xcol  = &x[block_col * bs];

            yrow[0] += block[0]*xcol[0] + block[1]*xcol[1] + block[2]*xcol[2];
            yrow[1] += block[3]*xcol[0] + block[4]*xcol[1] + block[5]*xcol[2];
            yrow[2] += block[6]*xcol[0] + block[7]*xcol[1] + block[8]*xcol[2];
        }
    }
}

void spmv_omp(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y)
{
    int bs = A.bs;

    for (int i = 0; i < A.nb * bs; i++) {
        y[i] = 0.0;
    }

    for (int row = 0; row < A.nb; row++) {
        int row_start = A.brptr[row];
        int row_end   = A.brptr[row + 1];

        double *yrow = &y[row * bs];

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        #pragma omp simd reduction(+:y0,y1,y2)
        for (int block_idx = row_start; block_idx < row_end; block_idx++) {
            int block_col = A.bcind[block_idx];

            const double *block = &A.bvals[block_idx * bs * bs];
            const double *xcol  = &x[block_col * bs];

            y0 += block[0]*xcol[0] + block[1]*xcol[1] + block[2]*xcol[2];
            y1 += block[3]*xcol[0] + block[4]*xcol[1] + block[5]*xcol[2];
            y2 += block[6]*xcol[0] + block[7]*xcol[1] + block[8]*xcol[2];
        }

        yrow[0] += y0;
        yrow[1] += y1;
        yrow[2] += y2;
    }
}

static double hadd_256(__m256d v)
{
    __m256d zero = _mm256_setzero_pd();

    // v_mod = [v0, v1, v2, 0.0]
    __m256d v_mod = _mm256_blend_pd(v, zero, MASK_ZERO_SLOT_3); 
    __m256d sum_half = _mm256_add_pd(
        v_mod, 
        _mm256_permute2f128_pd(v_mod, v_mod, 0x01) 
    );

    __m128d sum_128 = _mm256_castpd256_pd128(sum_half);
    __m128d final_sum = _mm_hadd_pd(sum_128, sum_128); 
    return _mm_cvtsd_f64(final_sum);
}

void spmv_avx256(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y)
{
    int bs = A.bs;

    for (int i = 0; i < A.nb * bs; i++) {
        y[i] = 0.0;
    }

    for (int row = 0; row < A.nb; row++) {
        int row_start = A.brptr[row];
        int row_end   = A.brptr[row + 1];

        double *yrow = &y[(size_t)row * bs];

        __m256d y0 = _mm256_setzero_pd();
        __m256d y1 = _mm256_setzero_pd();
        __m256d y2 = _mm256_setzero_pd();

        for (int block_idx = row_start; block_idx < row_end; block_idx++) {
            int block_col = A.bcind[block_idx];

            const double *block = &A.bvals[(size_t)block_idx * bs * bs];
            const double *xcol  = &x[(size_t)block_col * bs];

            __m256d vx = _mm256_loadu_pd(xcol);
            
            __m256d vb0 = _mm256_loadu_pd(block);
            y0  = _mm256_fmadd_pd(vb0, vx, y0);
            
            __m256d vb1 = _mm256_loadu_pd(&block[3]);
            y1 = _mm256_fmadd_pd(vb1, vx, y1);
            
            __m256d vb2  = _mm256_loadu_pd(&block[6]);
            y2 = _mm256_fmadd_pd(vb2, vx, y2);
        }

        yrow[0] = hadd_256(y0);
        yrow[1] = hadd_256(y1);
        yrow[2] = hadd_256(y2);   
    }
}


void spmv_avx512(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y)
{
    int bs = A.bs;

    for (int i = 0; i < A.nb * bs; i++) {
        y[i] = 0.0;
    }

    // Índices de permutação para replicar os elementos de x
    // [x0, x1, x2, x0, x1, x2, x0, x1]
    const __m512i perm_idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);

    for (int row = 0; row < A.nb; row++) {
        int row_start = A.brptr[row];
        int row_end   = A.brptr[row + 1];

        double *yrow = &y[(size_t)row * bs];

        __m512d acc = _mm512_setzero_pd();
        double y2_extra = 0.0;

        for (int block_idx = row_start; block_idx < row_end; block_idx++) {
            int block_col = A.bcind[block_idx];
            const double *block = &A.bvals[(size_t)block_idx * bs * bs];
            const double *xcol = &x[(size_t)block_col * bs];

            __m512d vb = _mm512_loadu_pd(block);  // Carrega 8 elementos do bloco

            __m256d vx_256 = _mm256_loadu_pd(xcol); // Carrega xcol em um vetor de 256 bits

            __m512d vx_512 = _mm512_broadcast_f64x4(vx_256); // Replica os elementos de xcol usando permutação
            vx_512 = _mm512_permutexvar_pd(perm_idx, vx_512);

            acc = _mm512_fmadd_pd(vb, vx_512, acc); // FMA: acc += vb * vx_512
            y2_extra += block[8] * xcol[2];
        }

        yrow[0] = _mm512_mask_reduce_add_pd(MASK_SUM_FIRST_3, acc);
        yrow[1] = _mm512_mask_reduce_add_pd(MASK_SUM_MID_3,   acc);
        yrow[2] = _mm512_mask_reduce_add_pd(MASK_SUM_LAST_2,  acc) + y2_extra;
    }
}

void spmv_hwy(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y)
{
    const int bs = A.bs;

    const hn::CappedTag<double, 4> d;
    const auto m3 = hn::FirstN(d, bs);

    for (int i = 0; i < A.nb * bs; i++) {
        y[i] = 0.0;
    }

    for (int row = 0; row < A.nb; row++) {
        const int row_start = A.brptr[row];
        const int row_end   = A.brptr[row + 1];

        double *yrow = &y[(size_t) row * bs];

        auto y0 = hn::Zero(d);
        auto y1 = hn::Zero(d);
        auto y2 = hn::Zero(d);

        for (int block_idx = row_start; block_idx < row_end; block_idx++) {
            const int block_col = A.bcind[block_idx];

            const double *block = &A.bvals[(size_t)block_idx * bs * bs];
            const double *xcol  = &x[(size_t)block_col * bs];

            const auto vx  = hn::MaskedLoad(m3, d, xcol);
            const auto vb0 = hn::MaskedLoad(m3, d, block);
            const auto vb1 = hn::MaskedLoad(m3, d, block + 3);
            const auto vb2 = hn::MaskedLoad(m3, d, block + 6);

            y0 = hn::MulAdd(vb0, vx, y0);
            y1 = hn::MulAdd(vb1, vx, y1);
            y2 = hn::MulAdd(vb2, vx, y2);
        }

        yrow[0] = hn::ReduceSum(d, y0);
        yrow[1] = hn::ReduceSum(d, y1);
        yrow[2] = hn::ReduceSum(d, y2);
    }
}