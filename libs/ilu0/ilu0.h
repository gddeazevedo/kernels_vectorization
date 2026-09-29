#pragma once

#include <bcsr.h>
#include <block_ops.h>

using ilu0_func_t = void (*)(BlockCSR &);

void ilu0_decomposition(BlockCSR &A);
void ilu0_decomposition_omp(BlockCSR &A);
void ilu0_decomposition_avx256(BlockCSR &A);
void ilu0_decomposition_avx512(BlockCSR &A);
void ilu0_decomposition_hwy(BlockCSR &A);