#include <spmv_benchmarks.h>

SpmvBenchmark::SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler)
    : BenchmarkBase(ini, fim, inc, K, compiler) {}

const char *SpmvBenchmark::benchmark_name() const
{
    return "SpMV Benchmark";
}

const char *SpmvBenchmark::csv_prefix() const
{
    return "spmv";
}

int SpmvBenchmark::variant_count() const
{
    return (int)variants.size();
}

const std::string &SpmvBenchmark::variant_name(int v) const
{
    return variants[v].name;
}

void SpmvBenchmark::evaluate(int nx, int ny, int nz, FILE *runs_csv)
{
    int N = nx * ny * nz;

    constexpr int TABLE_WIDTH = 92;
    constexpr useconds_t COOLDOWN_US = 50000;

    printf("\n");
    print_separator('=', TABLE_WIDTH);
    printf("  Malha: %d x %d x %d   |   N = %d   |   3N = %d   |   K = %d iterações\n",
           nx, ny, nz, N, 3*N, K);
    print_separator('=', TABLE_WIDTH);

    BlockedCSR A = BlockedCSR::generate_blocked27_3x3(nx, ny, nz);

    double *x      = (double *)malloc((size_t)3 * N * sizeof(double));
    double *y_ref  = (double *)malloc((size_t)3 * N * sizeof(double));
    double *y_test = (double *)malloc((size_t)3 * N * sizeof(double));

    for (int i = 0; i < 3 * N; i++) {
        x[i] = (double)i;
    }

    spmv(A, x, y_ref);

    std::vector<double> means;
    std::vector<double> medians;
    std::vector<double> errors(variants.size());

    auto prepare = [](int) {};

    auto kernel = [&](int v) {
        variants[v].func(A, x, y_test);
    };

    measure_interleaved(prepare, kernel, COOLDOWN_US, means, medians);

    for (int v = 0; v < (int)variants.size(); v++) {
        variants[v].func(A, x, y_test);

        double max_err = 0.0;
        for (int i = 0; i < 3 * N; i++) {
            double diff = fabs(y_ref[i] - y_test[i]) / fabs(y_ref[i]);
            if (diff > max_err) max_err = diff;
        }
        errors[v] = max_err;
    }

    double mean_ref   = means[0];
    double median_ref = medians[0];

    printf("  %-18s %14s %12s %14s %12s %12s\n",
           "Variante", "Média (s)", "Speedup", "Mediana (s)", "Speedup", "Erro Máx");
    print_separator('-', TABLE_WIDTH);

    for (int v = 0; v < (int)variants.size(); v++) {
        double speedup_mean   = mean_ref   / means[v];
        double speedup_median = median_ref / medians[v];

        gs_mean[v]   += log(speedup_mean);
        gs_median[v] += log(speedup_median);

        printf("  %-18s %14.6f %11.2fx %14.6f %11.2fx %12.2e\n",
               variants[v].name.c_str(),
               means[v],   speedup_mean,
               medians[v], speedup_median,
               errors[v]);

        fprintf(runs_csv, "%d,%d,%d,%d,%s,%.6f,%.4f,%.6f,%.4f,%.2e\n",
            N, nx, ny, nz,
            variants[v].name.c_str(),
            means[v],
            speedup_mean,
            medians[v],
            speedup_median,
            errors[v]);
    }

    print_separator('=', TABLE_WIDTH);

    gs_count++;

    free(x);
    free(y_ref);
    free(y_test);
}