#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

#define MASK_ZERO_SLOT_3 0x8  // 0b1000
#define MASK_SUM_FIRST_3 0x07 // 0b00000111
#define MASK_SUM_MID_3   0x38 // 0b00111000
#define MASK_SUM_LAST_2  0xC0 // 0b11000000

#define BS 3
#define BS2 (BS * BS)
#define idx(i,j) ((i)*BS + (j))

inline void invert_common(double *dst, double &det, const double *M)
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

inline void transpose(double *dst, const double *M)
{
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            dst[idx(i,j)] = M[idx(j,i)];
        }
    }
}

inline void invert_3x3_matrix(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    for (int i = 0; i < BS * BS; i++) {
        dst[i] /= det;
    }
}

inline void matmat(double *dst, const double *A, const double *B)
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

inline void matsub(double *dst, const double *A, const double *B)
{
    for (int i = 0; i < BS * BS; i++) {
        dst[i] = A[i] - B[i];
    }
}

inline void invert_3x3_matrix_omp(double *dst, const double *M)
{
    double det;
    invert_common(dst, det, M);

    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst[i] /= det;
    }
}

inline void matmat_omp(double *dst, const double *A, const double *B)
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

inline void matsub_omp(double *dst, const double *A, const double *B)
{
    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst[i] = A[i] - B[i];
    }
}

inline void gather_blocks_avx256(
    __m256d dst[BS2],
    const double *blocks,
    __m256i offsets,
    __mmask8 mask
)
{
    const __m256d zeros = _mm256_setzero_pd();

    for (int reg = 0; reg < BS2; reg++) {
        __m256i indices = _mm256_add_epi64(offsets, _mm256_set1_epi64x(reg));
        dst[reg] = _mm256_mmask_i64gather_pd(zeros, mask, indices, blocks, sizeof(double));
    }
}

inline void scatter_blocks_avx256(
    double *blocks,
    const __m256d src[BS2],
    const int64_t *offsets,
    int n_blocks
)
{
    for (int reg = 0; reg < BS2; reg++) {
        double tmp[4];
        _mm256_storeu_pd(tmp, src[reg]);

        for (int l = 0; l < n_blocks; l++) {
            blocks[offsets[l] + reg] = tmp[l];
        }
    }
}

inline void matmat_avx256(__m256d dst[BS2], const __m256d A[BS2], const __m256d B[BS2])
{
    for (int row = 0; row < BS; row++) {
        for (int col = 0; col < BS; col++) {
            __m256d acc = _mm256_mul_pd  (A[idx(row, 0)], B[idx(0, col)]);
            acc         = _mm256_fmadd_pd(A[idx(row, 1)], B[idx(1, col)], acc);
            acc         = _mm256_fmadd_pd(A[idx(row, 2)], B[idx(2, col)], acc);
            dst[idx(row, col)] = acc;
        }
    }
}

inline void matsub_avx256(__m256d dst[BS2], const __m256d A[BS2], const __m256d B[BS2])
{
    for (int reg = 0; reg < BS2; reg++) {
        dst[reg] = _mm256_sub_pd(A[reg], B[reg]);
    }
}

inline void process_blocks_avx256(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
)
{
    __mmask8 mask = (__mmask8) ((1u << n_blocks) - 1);

    __m256i voffsets_i = _mm256_loadu_si256((const __m256i *) offsets_i);
    __m256i voffsets_k = _mm256_loadu_si256((const __m256i *) offsets_k);

    __m256d Bij[BS2];
    __m256d Bkj[BS2];
    __m256d Bik[BS2];
    __m256d prod[BS2];
    __m256d diff[BS2];

    gather_blocks_avx256(Bij, blocks, voffsets_i, mask);
    gather_blocks_avx256(Bkj, blocks, voffsets_k, mask);

    for (int reg = 0; reg < BS2; reg++) {
        Bik[reg] = _mm256_set1_pd(block_ik[reg]);
    }

    matmat_avx256(prod, Bik, Bkj);
    matsub_avx256(diff, Bij, prod);

    scatter_blocks_avx256(blocks, diff, offsets_i, n_blocks);
}

inline void gather_blocks_avx512(
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

inline void scatter_blocks_avx512(
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

inline void matmat_avx512(__m512d dst[BS2], const __m512d A[BS2], const __m512d B[BS2])
{
    for (int row = 0; row < BS; row++) {
        for (int col = 0; col < BS; col++) {
            __m512d acc = _mm512_mul_pd(A[idx(row, 0)], B[idx(0, col)]);
            acc         = _mm512_fmadd_pd(A[idx(row, 1)], B[idx(1, col)], acc);
            acc         = _mm512_fmadd_pd(A[idx(row, 2)], B[idx(2, col)], acc);
            dst[idx(row, col)] = acc;
        }
    }
}

inline void matsub_avx512(__m512d dst[BS2], const __m512d A[BS2], const __m512d B[BS2])
{
    for (int reg = 0; reg < BS2; reg++) {
        dst[reg] = _mm512_sub_pd(A[reg], B[reg]);
    }
}

inline void process_blocks_avx512(
    double *blocks,
    const double *block_ik,
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
    __m512d Bik[BS2];
    __m512d prod[BS2];
    __m512d diff[BS2];

    gather_blocks_avx512(Bij, blocks, voffsets_i, mask);
    gather_blocks_avx512(Bkj, blocks, voffsets_k, mask);

    for (int reg = 0; reg < BS2; reg++) {
        Bik[reg] = _mm512_set1_pd(block_ik[reg]);
    }

    matmat_avx512(prod, Bik, Bkj);
    matsub_avx512(diff, Bij, prod);

    scatter_blocks_avx512(blocks, diff, voffsets_i, mask);
}

inline void gather_blocks_hwy256(
    hn::Vec<hn::FixedTag<double, 4>> dst[BS2],
    const double *blocks,
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 4>>> offsets,
    hn::Mask<hn::FixedTag<double, 4>> mask
)
{
    const hn::FixedTag<double, 4> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    const auto zeros = hn::Zero(d);

    for (int reg = 0; reg < BS2; reg++) {
        auto indices = hn::Add(offsets, hn::Set(di, reg));
        dst[reg] = hn::MaskedGatherIndexOr(zeros, mask, d, blocks, indices);
    }
}

inline void scatter_blocks_hwy256(
    double *blocks,
    const hn::Vec<hn::FixedTag<double, 4>> src[BS2],
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 4>>> offsets,
    hn::Mask<hn::FixedTag<double, 4>> mask
)
{
    const hn::FixedTag<double, 4> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    for (int reg = 0; reg < BS2; reg++) {
        auto indices = hn::Add(offsets, hn::Set(di, reg));
        hn::MaskedScatterIndex(src[reg], mask, d, blocks, indices);
    }
}

inline void matmat_hwy256(
    hn::Vec<hn::FixedTag<double, 4>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> B[BS2]
)
{
    for (int row = 0; row < BS; row++) {
        for (int col = 0; col < BS; col++) {
            auto acc = hn::Mul(A[idx(row, 0)], B[idx(0, col)]);
            acc      = hn::MulAdd(A[idx(row, 1)], B[idx(1, col)], acc);
            acc      = hn::MulAdd(A[idx(row, 2)], B[idx(2, col)], acc);
            dst[idx(row, col)] = acc;
        }
    }
}

inline void matsub_hwy256(
    hn::Vec<hn::FixedTag<double, 4>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> B[BS2]
)
{
    for (int reg = 0; reg < BS2; reg++) {
        dst[reg] = hn::Sub(A[reg], B[reg]);
    }
}

inline void process_blocks_hwy256(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
)
{
    const hn::FixedTag<double, 4> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    auto mask = hn::FirstN(d, n_blocks);

    auto voffsets_i = hn::LoadU(di, offsets_i);
    auto voffsets_k = hn::LoadU(di, offsets_k);

    hn::Vec<decltype(d)> Bij[BS2];
    hn::Vec<decltype(d)> Bkj[BS2];
    hn::Vec<decltype(d)> Bik[BS2];
    hn::Vec<decltype(d)> prod[BS2];
    hn::Vec<decltype(d)> diff[BS2];

    gather_blocks_hwy256(Bij, blocks, voffsets_i, mask);
    gather_blocks_hwy256(Bkj, blocks, voffsets_k, mask);

    for (int reg = 0; reg < BS2; reg++) {
        Bik[reg] = hn::Set(d, block_ik[reg]);
    }

    matmat_hwy256(prod, Bik, Bkj);
    matsub_hwy256(diff, Bij, prod);

    scatter_blocks_hwy256(blocks, diff, voffsets_i, mask);
}

inline void gather_blocks_hwy512(
    hn::Vec<hn::FixedTag<double, 8>> dst[BS2],
    const double *blocks,
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 8>>> offsets,
    hn::Mask<hn::FixedTag<double, 8>> mask
)
{
    const hn::FixedTag<double, 8> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    const auto zeros = hn::Zero(d);

    for (int reg = 0; reg < BS2; reg++) {
        auto indices = hn::Add(offsets, hn::Set(di, reg));
        dst[reg] = hn::MaskedGatherIndexOr(zeros, mask, d, blocks, indices);
    }
}

inline void scatter_blocks_hwy512(
    double *blocks,
    const hn::Vec<hn::FixedTag<double, 8>> src[BS2],
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 8>>> offsets,
    hn::Mask<hn::FixedTag<double, 8>> mask
)
{
    const hn::FixedTag<double, 8> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    for (int reg = 0; reg < BS2; reg++) {
        auto indices = hn::Add(offsets, hn::Set(di, reg));
        hn::MaskedScatterIndex(src[reg], mask, d, blocks, indices);
    }
}

inline void matmat_hwy512(
    hn::Vec<hn::FixedTag<double, 8>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> B[BS2]
)
{
    for (int row = 0; row < BS; row++) {
        for (int col = 0; col < BS; col++) {
            auto acc = hn::Mul(A[idx(row, 0)], B[idx(0, col)]);
            acc      = hn::MulAdd(A[idx(row, 1)], B[idx(1, col)], acc);
            acc      = hn::MulAdd(A[idx(row, 2)], B[idx(2, col)], acc);
            dst[idx(row, col)] = acc;
        }
    }
}

inline void matsub_hwy512(
    hn::Vec<hn::FixedTag<double, 8>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> B[BS2]
)
{
    for (int reg = 0; reg < BS2; reg++) {
        dst[reg] = hn::Sub(A[reg], B[reg]);
    }
}

inline void process_blocks_hwy512(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
)
{
    const hn::FixedTag<double, 8> d;
    const hn::Rebind<int64_t, decltype(d)> di;

    auto mask = hn::FirstN(d, n_blocks);

    auto voffsets_i = hn::LoadU(di, offsets_i);
    auto voffsets_k = hn::LoadU(di, offsets_k);

    hn::Vec<decltype(d)> Bij[BS2];
    hn::Vec<decltype(d)> Bkj[BS2];
    hn::Vec<decltype(d)> Bik[BS2];
    hn::Vec<decltype(d)> prod[BS2];
    hn::Vec<decltype(d)> diff[BS2];

    gather_blocks_hwy512(Bij, blocks, voffsets_i, mask);
    gather_blocks_hwy512(Bkj, blocks, voffsets_k, mask);

    for (int reg = 0; reg < BS2; reg++) {
        Bik[reg] = hn::Set(d, block_ik[reg]);
    }

    matmat_hwy512(prod, Bik, Bkj);
    matsub_hwy512(diff, Bij, prod);

    scatter_blocks_hwy512(blocks, diff, voffsets_i, mask);
}