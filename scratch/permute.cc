#include <iostream>
#include <immintrin.h>

int main() {
    double x[3] = {1.0, 2.0, 3.0};

    __m512i perm_idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);
    __m256d vx256 = _mm256_loadu_pd(x);
    __m512d vx512 = _mm512_broadcast_f64x4(vx256);
    vx512 = _mm512_permutexvar_pd(perm_idx, vx512);

    for (int i = 0; i < 8; i++) {
        double val = vx512[i];
        printf("Element %d: %f\n", i, val);
    }

    return 0;
}