#include <immintrin.h>
#include <cstdio>


void AddArrays(
    const double * __restrict__ a,
    const double * __restrict__ b,
    double * __restrict__ out,
    const size_t n
) {
    auto chunks = sizeof(__m512d) / sizeof(double);

    size_t i = 0;

    for (; i + chunks <= n; i += chunks) {
        __m512d va   = _mm512_loadu_pd(a + i);
        __m512d vb   = _mm512_loadu_pd(b + i);
        __m512d vsum = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(out +i, vsum);
    }

    for (; i < n; i++) {
        std::printf("Adding elements %zu (a: %.1f, b: %.1f) (out: %.1f)\n", i, a[i], b[i], out[i]);
        out[i] = a[i] + b[i];
    }
}

int main() {
    constexpr size_t n = 10;
    double a[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double b[n] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    double out[n];

    AddArrays(a, b, out, n);

    for (size_t i = 0; i < n; ++i) {
        std::printf("%.1f ", out[i]);
    }

    std::printf("\n");
    return 0;
}