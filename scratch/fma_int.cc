#include <immintrin.h>
#include <iostream>

int main() {
    __m512i a = _mm512_set1_epi32(4);
    __m512i b = _mm512_set1_epi32(3);
    __m512i c = _mm512_set1_epi32(10);

    // Result = a * b + c
    __m512i result = _mm512_add_epi32(_mm512_mullo_epi32(a, b), c);

    int res[16];
    _mm512_storeu_epi32(res, result);

    for (int i = 0; i < sizeof(res) / sizeof(int); i++) {
        std::cout << "Result: " << res[i] << std::endl;
    }
    return 0;
}
