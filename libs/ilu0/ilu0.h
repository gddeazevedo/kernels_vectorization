#pragma once

#include <bcsr.h>
#include <block_matrix_ops.h>

using ilu0_func_t = void (*)(BlockedCSR &);

void ilu0_decomposition(BlockedCSR &A);
void ilu0_decomposition_omp(BlockedCSR &A);
void ilu0_decomposition_avx256(BlockedCSR &A);
void ilu0_decomposition_avx512(BlockedCSR &A);
void ilu0_decomposition_hwy256(BlockedCSR &A);
void ilu0_decomposition_hwy512(BlockedCSR &A);
