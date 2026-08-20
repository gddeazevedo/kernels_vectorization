#include <cstdio>
#include <hwy/highway.h>
#include <hwy/print-inl.h>

namespace hn = hwy::HWY_NAMESPACE;

void AddArrays(
    const float *HWY_RESTRICT a,
    const float *HWY_RESTRICT b,
    float *HWY_RESTRICT out,
    const size_t n
) {
    const hn::ScalableTag<float> d;
    const size_t N_LANES = hn::Lanes(d); // é a quantidade que um registrador pode conter, por exemplo, 8 para AVX2 e 16 para AVX-512 no caso de floats

    std::printf("FLOAT: Vector size: %zu, Lanes: %zu\n", n, N_LANES);

    size_t i = 0;
    for (; i + N_LANES <= n; i += N_LANES) {
        auto va   = hn::Load(d, a + i);
        auto vb   = hn::Load(d, b + i);
        auto vsum = va + vb;
        hn::Store(vsum, d, out + i);
    }

    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void AddArrays(
    const double *HWY_RESTRICT a,
    const double *HWY_RESTRICT b,
    double *HWY_RESTRICT out,
    const size_t n
) {
    const hn::ScalableTag<double> d;
    const size_t N_LANES = hn::Lanes(d); // é a quantidade que um registrador pode conter, por exemplo, 4 para AVX2 e 8 para AVX-512 no caso de doubles

    std::printf("DOUBLE: Vector size: %zu, Lanes: %zu\n", n, N_LANES);

    size_t i = 0;
    for (; i + N_LANES <= n; i += N_LANES) {
        auto va   = hn::LoadU(d, a + i);
        auto vb   = hn::LoadU(d, b + i);
        auto vsum = hn::Add(va, vb);
        hn::StoreU(vsum, d, out + i);
    }

    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    constexpr size_t n = 10;
    float a[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[n] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    float out[n];

    AddArrays(a, b, out, n);

    for (size_t i = 0; i < n; ++i) {
        std::printf("%.1f ", out[i]);
    }

    std::printf("\n");
    return 0;
}