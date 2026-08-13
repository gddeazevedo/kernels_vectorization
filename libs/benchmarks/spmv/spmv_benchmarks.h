#pragma once

#include <benchmark_base.h>
#include <spmv.h>

struct MatvecVariant {
    const std::string name;
    spmv_func_t func;
};

class SpmvBenchmark : public BenchmarkBase {
    private:
        const std::vector<MatvecVariant> variants = {
            {"Base",        spmv},
            {"OpenMP",      spmv_omp},
            {"AVX256",      spmv_avx256},
            {"AVX512",      spmv_avx512},
            {"Highway256",  spmv_hwy256},
            {"Highway512",  spmv_hwy512}
        };

        void evaluate(int nx, int ny, int nz, FILE *runs_csv) override;
        const char *benchmark_name() const override;
        const char *csv_prefix() const override;
        int variant_count() const override;
        const std::string &variant_name(int v) const override;

    public:
        SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler);
};