#include <immintrin.h>
#include <iostream>

int main() {
    __m512d a = _mm512_set1_pd(3.0);
    __m512d b = _mm512_set1_pd(3.0);
    __m512d c = _mm512_set1_pd(0.0);

    // Result = a * b + c
    __m512d result = _mm512_fmadd_pd(a, b, c);

    double res[8];
    _mm512_storeu_pd(res, result);

    for (int i = 0; i < sizeof(res) / sizeof(double); i++) {
        std::cout << "Result: " << res[i] << std::endl;
    }
    return 0;
}
