#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

#define BATCH 8
#define BS 3
#define BS2 (BS * BS)
#define idx(i,j) ((i)*BS + (j))

using HwyTag  = hn::ScalableTag<double>;
using HwyTagI = hn::Rebind<int64_t, HwyTag>;

#define HWY_BATCH_MAX HWY_MAX_LANES_D(HwyTag)

void transpose(double *dst, const double *M);
void invert_3x3_matrix(double *dst, const double *M);
void matmat(double *dst, const double *A, const double *B);
void matsub(double *dst, const double *A, const double *B);

void gather_blocks_omp(
    double dst[BS2][BATCH],
    const double *blocks,
    const int64_t *offsets,
    int n_blocks
);
void scatter_blocks_omp(
    double *blocks,
    const double src[BS2][BATCH],
    const int64_t *offsets,
    int n_blocks
);
void matmat_batch_omp(
    double dst[BS2][BATCH],
    const double A[BS2][BATCH],
    const double B[BS2][BATCH]
);
void matsub_batch_omp(
    double dst[BS2][BATCH],
    const double A[BS2][BATCH],
    const double B[BS2][BATCH]
);
void process_blocks_omp(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
);

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

void gather_blocks_hwy(
    hn::Vec<HwyTag> dst[BS2],
    const double *blocks,
    hn::Vec<HwyTagI> offsets,
    hn::Mask<HwyTag> mask
);
void scatter_blocks_hwy(
    double *blocks,
    const hn::Vec<HwyTag> src[BS2],
    hn::Vec<HwyTagI> offsets,
    hn::Mask<HwyTag> mask
);
void matmat_hwy(
    hn::Vec<HwyTag> dst[BS2],
    const hn::Vec<HwyTag> A[BS2],
    const hn::Vec<HwyTag> B[BS2]
);
void matsub_hwy(
    hn::Vec<HwyTag> dst[BS2],
    const hn::Vec<HwyTag> A[BS2],
    const hn::Vec<HwyTag> B[BS2]
);
void process_blocks_hwy(
    double *blocks,
    const double *block_ik,
    const int64_t *offsets_i,
    const int64_t *offsets_k,
    int n_blocks
);