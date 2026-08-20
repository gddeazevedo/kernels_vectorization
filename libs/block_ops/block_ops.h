#pragma once
// block_ops.h
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

void invert_common(double *dst, double &det, const double *M);

void transpose(double *dst, const double *M);

void invert_3x3_matrix(double *dst, const double *M);

void matmat(double *dst, const double *A, const double *B);

void matsub(double *dst, const double *A, const double *B);

void invert_3x3_matrix_omp(double *dst, const double *M);

void matmat_omp(double *dst, const double *A, const double *B);

void matsub_omp(double *dst, const double *A, const double *B);

void gather_blocks_avx256(
    __m256d dst[BS2],
    const double *blocks,
    __m256i offsets,
    __mmask8 mask
);

void scatter_blocks_avx256(
    double *blocks,
    const __m256d src[BS2],
    const int64_t *offsets,
    int n_blocks
);

void matmat_avx256(__m256d dst[BS2], const __m256d A[BS2], const __m256d B[BS2]);

void matsub_avx256(__m256d dst[BS2], const __m256d A[BS2], const __m256d B[BS2]);

void process_blocks_avx256(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
);

void gather_blocks_avx512(
    __m512d dst[BS2],
    const double *blocks,
    __m512i offsets,
    __mmask8 mask
);

void scatter_blocks_avx512(
    double *blocks,
    const __m512d src[BS2],
    __m512i offsets,
    __mmask8 mask
);

void matmat_avx512(__m512d dst[BS2], const __m512d A[BS2], const __m512d B[BS2]);

void matsub_avx512(__m512d dst[BS2], const __m512d A[BS2], const __m512d B[BS2]);

void process_blocks_avx512(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
);

void gather_blocks_hwy256(
    hn::Vec<hn::FixedTag<double, 4>> dst[BS2],
    const double *blocks,
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 4>>> offsets,
    hn::Mask<hn::FixedTag<double, 4>> mask
);

void scatter_blocks_hwy256(
    double *blocks,
    const hn::Vec<hn::FixedTag<double, 4>> src[BS2],
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 4>>> offsets,
    hn::Mask<hn::FixedTag<double, 4>> mask
);

void matmat_hwy256(
    hn::Vec<hn::FixedTag<double, 4>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> B[BS2]
);

void matsub_hwy256(
    hn::Vec<hn::FixedTag<double, 4>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 4>> B[BS2]
);

void process_blocks_hwy256(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
);

void gather_blocks_hwy512(
    hn::Vec<hn::FixedTag<double, 8>> dst[BS2],
    const double *blocks,
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 8>>> offsets,
    hn::Mask<hn::FixedTag<double, 8>> mask
);

void scatter_blocks_hwy512(
    double *blocks,
    const hn::Vec<hn::FixedTag<double, 8>> src[BS2],
    hn::Vec<hn::Rebind<int64_t, hn::FixedTag<double, 8>>> offsets,
    hn::Mask<hn::FixedTag<double, 8>> mask
);

void matmat_hwy512(
    hn::Vec<hn::FixedTag<double, 8>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> B[BS2]
);

void matsub_hwy512(
    hn::Vec<hn::FixedTag<double, 8>> dst[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> A[BS2],
    const hn::Vec<hn::FixedTag<double, 8>> B[BS2]
);

void process_blocks_hwy512(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
);