#include <iostream>
#include <immintrin.h>

int main() {
    double x[3] = {1.0, 2.0, 3.0};

    __m512i idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);
    __m256d vx = _mm256_load_pd(x);
    __m512d vx_broadcast = _mm512_broadcast_f64x4(vx);
    __m512d vx_permuted = _mm512_permutexvar_pd(idx, vx_broadcast);

    for (int i = 0; i < 8; i++) {
        double val = vx_permuted[i];
        std::cout << "Element " << i << ": " << val << std::endl;
    }

    return 0;
}
