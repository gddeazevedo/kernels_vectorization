#include <iostream>
#include <immintrin.h>

int main() {
    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4] = {5.0, 10.0, 15.0, 20.0};
    unsigned char mask = 0b0100;

    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d vc = _mm256_blend_pd(va, vb, mask);

    for (int i = 0; i < 4; i++) {
        double val = vc[i];
        printf("Element %d: %f\n", i, val);
    }

    return 0;
}