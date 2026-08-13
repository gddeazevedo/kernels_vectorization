#include <ilu0_benchmarks.h>

Ilu0Benchmark::Ilu0Benchmark(int ini, int fim, int inc, int K, const std::string &compiler)
    : BenchmarkBase(ini, fim, inc, K, compiler) {}

const char *Ilu0Benchmark::benchmark_name() const {
    return "ILU0 Benchmark";
}

const char *Ilu0Benchmark::csv_prefix() const {
    return "ilu0";
}

int Ilu0Benchmark::variant_count() const {
    return (int)variants.size();
}

const std::string &Ilu0Benchmark::variant_name(int v) const {
    return variants[v].name;
}

void Ilu0Benchmark::evaluate(int nx, int ny, int nz, FILE *runs_csv) {
    int N = nx * ny * nz;

    constexpr int TABLE_WIDTH = 92;

    printf("\n");
    print_separator('=', TABLE_WIDTH);
    printf("  Malha: %d x %d x %d   |   N = %d   |   3N = %d   |   K = %d iterações\n",
           nx, ny, nz, N, 3*N, K);
    print_separator('=', TABLE_WIDTH);

    BlockedCSR A = BlockedCSR::generate_blocked27_3x3(nx, ny, nz);

    size_t vals_size = (size_t)A.nnzb * A.bs * A.bs * sizeof(double);

    double *orig_vals = (double *)malloc(vals_size);
    double *ref_vals  = (double *)malloc(vals_size);
    memcpy(orig_vals, A.vals, vals_size);

    // Resultado de referência (variante base)
    ilu0_decomposition(A);
    memcpy(ref_vals, A.vals, vals_size);

    double *sample = (double *)malloc(K * sizeof(double));

    std::vector<double> means(variants.size());
    std::vector<double> medians(variants.size());
    std::vector<double> errors(variants.size());

    for (int v = 0; v < (int)variants.size(); v++) {
        double sum = 0.0;

        for (int i = 0; i < 5; i++) {
            memcpy(A.vals, orig_vals, vals_size);
            variants[v].func(A);
        }

        for (int k = 0; k < K; k++) {
            memcpy(A.vals, orig_vals, vals_size);
            double t0 = wtime();
            variants[v].func(A);
            sample[k] = wtime() - t0;
            sum += sample[k];
        }

        means[v]   = sum / K;
        medians[v] = median(sample, K);

        double max_err = 0.0;
        int total_vals = A.nnzb * A.bs * A.bs;
        for (int i = 0; i < total_vals; i++) {
            double ref = fabs(ref_vals[i]);
            if (ref > 0.0) {
                double diff = fabs(ref_vals[i] - A.vals[i]) / ref;
                if (diff > max_err) max_err = diff;
            }
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

        gs_mean[v]   += 1.0 / speedup_mean;
        gs_median[v] += 1.0 / speedup_median;

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

    free(sample);
    free(orig_vals);
    free(ref_vals);
}