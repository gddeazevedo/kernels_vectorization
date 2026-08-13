#pragma once

#include <stdlib.h>
#include <immintrin.h>
#include <hwy/highway.h>
#include <bcsr.h>

#define MASK_ZERO_SLOT_3 0x8  // 0b1000
#define MASK_SUM_FIRST_3 0x07 // 0b00000111
#define MASK_SUM_MID_3   0x38 // 0b00111000
#define MASK_SUM_LAST_2  0xC0 // 0b11000000

namespace hn = hwy::HWY_NAMESPACE;

using spmv_func_t = void (*)(const BlockedCSR &, const double * __restrict__, double * __restrict__);

static double hadd_256(__m256d v);

void spmv(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y);
void spmv_omp(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y);
void spmv_avx256(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y);
void spmv_avx512(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y);
void spmv_hwy256(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y);
void spmv_hwy512(const BlockedCSR &A, const double * __restrict__ x, double * __restrict__ y);
