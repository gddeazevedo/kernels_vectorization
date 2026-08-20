#include <immintrin.h>
#include <iostream>

int main() {
    __m512 a = _mm512_set1_ps(3.0f);
    __m512 b = _mm512_set1_ps(3.0f);
    __m512 c = _mm512_set1_ps(0.0f);

    // Result = a * b + c
    __m512 result = _mm512_fmadd_ps(a, b, c);

    float res[16];
    _mm512_storeu_ps(res, result);

    for (int i = 0; i < sizeof(res) / sizeof(float); i++) {
        std::cout << "Result: " << res[i] << std::endl;
    }
    return 0;
}
