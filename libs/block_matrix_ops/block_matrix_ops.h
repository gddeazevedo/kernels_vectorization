#pragma once

#include <stdlib.h>
#include <immintrin.h>
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

#define MASK_ZERO_SLOT_3 0x8  // 0b1000
#define MASK_SUM_FIRST_3 0x07 // 0b00000111
#define MASK_SUM_MID_3   0x38 // 0b00111000
#define MASK_SUM_LAST_2  0xC0 // 0b11000000

#define BS 3
#define idx(i,j) ((i)*BS + (j))

static void invert_common(double *dst, const double *M, int *det);
static void transpose(double *dst, const double *M);

void invert_3x3_matrix(double *dst, const double *M);
void invert_3x3_matrix_omp(double *dst, const double *M);
void invert_3x3_matrix_avx256(double *dst, const double *M);
void invert_3x3_matrix_avx512(double *dst, const double *M);
void invert_3x3_matrix_hwy256(double *dst, const double *M);
void invert_3x3_matrix_hwy512(double *dst, const double *M);

void matmat(double *dst, const double *A, const double *B);
void matmat_omp(double *dst, const double *A, const double *B);
void matmat_avx256(double *dst, const double *A, const double *B);
void matmat_avx512(double *dst, const double *A, const double *B);
void matmat_hwy256(double *dst, const double *A, const double *B);
void matmat_hwy512(double *dst, const double *A, const double *B);

void matsub(double *dst, const double *A, const double *B);
void matsub_omp(double *dst, const double *A, const double *B);
void matsub_avx256(double *dst, const double *A, const double *B);
void matsub_avx512(double *dst, const double *A, const double *B);
void matsub_hwy256(double *dst, const double *A, const double *B);
void matsub_hwy512(double *dst, const double *A, const double *B);
