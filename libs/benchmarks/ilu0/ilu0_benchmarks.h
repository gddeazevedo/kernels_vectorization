#pragma once

#include <benchmark_base.h>
#include <ilu0.h>

using ilu0_func_t = void (*)(BlockedCSR &);

struct Ilu0Variant {
    const std::string name;
    ilu0_func_t func;
};

class Ilu0Benchmark : public BenchmarkBase {
    private:
        const std::vector<Ilu0Variant> variants = {
            {"Base",        ilu0_decomposition},
            {"OpenMP",      ilu0_decomposition_omp},
            {"AVX256",      ilu0_decomposition_avx256},
            {"AVX512",      ilu0_decomposition_avx512},
            {"Highway256",  ilu0_decomposition_hwy256},
            {"Highway512",  ilu0_decomposition_hwy512}
        };

        void evaluate(int nx, int ny, int nz, FILE *runs_csv) override;
        const char *benchmark_name() const override;
        const char *csv_prefix() const override;
        int variant_count() const override;
        const std::string &variant_name(int v) const override;

    public:
        Ilu0Benchmark(int ini, int fim, int inc, int K, const std::string &compiler);
};