#include <immintrin.h>
#include <iostream>

// g++ -O3 -march=native -o fma fma.cc -> otimização máxima
// g++ -mavx -mfma -o fma_test fma_test.cpp

int main() {
    // __m256 represents a vector of 8 floats
    __m256 a = _mm256_set1_ps(2.0f);
    __m256 b = _mm256_set1_ps(2.0f);
    // Cria um vetor SIMD de 8 floats, onde todos os elementos são inicializados com o valor 4.0f.
    // Utiliza a instrução AVX _mm256_set1_ps para preencher o registrador __m256.
    // Isso permite operações vetoriais eficientes com o mesmo valor em todos os elementos.
    __m256 c = _mm256_set1_ps(6.0f);

    // Result = a * b + c
    __m256 result = _mm256_fmadd_ps(a, b, c);

    float res[8];
    _mm256_storeu_ps(res, result);

    std::cout << "Result: " << res[0] << std::endl;
    return 0;
}
