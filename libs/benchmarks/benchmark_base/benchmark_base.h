#pragma once

#include <utils.h>
#include <bcsr.h>
#include <vector>

class BenchmarkBase {
    protected:
        int ini, fim, inc, K;
        std::string compiler;
        std::vector<double> gs_mean;
        std::vector<double> gs_median;
        int gs_count = 0;

        virtual void evaluate(int nx, int ny, int nz, FILE *runs_csv) = 0;
        virtual const char *benchmark_name() const = 0;
        virtual const char *csv_prefix() const = 0;
        virtual int variant_count() const = 0;
        virtual const std::string &variant_name(int v) const = 0;

    public:
        BenchmarkBase(int ini, int fim, int inc, int K, const std::string &compiler);
        virtual ~BenchmarkBase() = default;
        int run();
};