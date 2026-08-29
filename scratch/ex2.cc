#include <immintrin.h>
#include <iostream>

int main() {
    double a[8] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    double b[8] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    double result[8];

    __m512d vec_a = _mm512_loadu_pd(a);
    __m512d vec_b = _mm512_loadu_pd(b);
    __m512d vec_result = _mm512_add_pd(vec_a, vec_b);
    _mm512_storeu_pd(result, vec_result);

    std::cout << "Result: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}