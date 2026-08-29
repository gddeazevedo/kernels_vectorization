#include <immintrin.h>
#include <iostream>


#define SIZE 2

int main() {
    double a[SIZE] = {1.0, 2.0};
    double b[SIZE] = {3.0, 4.0};
    double result[SIZE];

    __m128d vec_a = _mm_loadu_pd(a);
    __m128d vec_b = _mm_loadu_pd(b);
    __m128d vec_result = _mm_add_pd(vec_a, vec_b);
    _mm_storeu_pd(result, vec_result);

    std::cout << "Result: ";
    for (int i = 0; i < SIZE; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}