#pragma once

#include <bcsr.h>
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

using ilu0_func_t = void (*)(BlockedCSR &);

static void invert_3x3_matrix(double *dst, const double *M);
static void matmat(double *dst, const double *A, const double *B);
static void matsub(double *dst, const double *A, const double *B);

static void invert_3x3_matrix_omp(double *dst, const double *M);
static void matmat_omp(double *dst, const double *A, const double *B);
static void matsub_omp(double *dst, const double *A, const double *B);

void ilu0_decomposition(BlockedCSR &A);
void ilu0_decomposition_omp(BlockedCSR &A);
void ilu0_decomposition_avx256(BlockedCSR &A);
void ilu0_decomposition_avx512(BlockedCSR &A);
void ilu0_decomposition_hwy256(BlockedCSR &A);
void ilu0_decomposition_hwy512(BlockedCSR &A);