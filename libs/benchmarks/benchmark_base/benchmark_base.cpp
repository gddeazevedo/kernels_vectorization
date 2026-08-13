#include <benchmark_base.h>

BenchmarkBase::BenchmarkBase(int ini, int fim, int inc, int K, const std::string &compiler) {
    this->ini = ini;
    this->fim = fim;
    this->inc = inc;
    this->K   = K;
    this->compiler = compiler;
}

int BenchmarkBase::run() {
    if (ini <= 0 || fim < ini || inc <= 0 || K <= 0) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    this->gs_mean   = std::vector<double>(variant_count(), 0.0);
    this->gs_median = std::vector<double>(variant_count(), 0.0);

    constexpr int SUMMARY_WIDTH = 64;

    std::string compiler_dir;

    ensure_experiment_dirs(csv_prefix(), compiler, compiler_dir);

    std::string runs_path = build_path(compiler_dir, std::string(csv_prefix()) + "_runs.csv");
    FILE *runs_csv = fopen(runs_path.c_str(), "w");
    fprintf(runs_csv, "N,nx,ny,nz,variante,media_s,speedup_mean,mediana_s,speedup_median,erro_max\n");

    printf("\n");
    print_separator('#', SUMMARY_WIDTH);
    printf("#%*s%*s#\n",
           (SUMMARY_WIDTH - 2) / 2 + 13, benchmark_name(),
           (SUMMARY_WIDTH - 2) / 2 - 13, "");
    print_separator('#', SUMMARY_WIDTH);
    printf("  Compilador : %s\n", compiler.c_str());
    printf("  Malhas     : %d → %d (passo %d)\n", ini, fim, inc);
    printf("  Iterações  : %d por variante\n", K);
    printf("  Variantes  : %d\n", variant_count());
    print_separator('#', SUMMARY_WIDTH);

    for (int nx = ini; nx <= fim; nx += inc) {
        evaluate(nx, nx, nx, runs_csv);
    }

    fclose(runs_csv);

    printf("\n");
    print_separator('=', SUMMARY_WIDTH);
    printf("  Speedup Geral (média harmônica sobre %d malhas)\n", gs_count);
    print_separator('=', SUMMARY_WIDTH);
    printf("  %-18s %18s %18s\n", "Variante", "Speedup (Mean)", "Speedup (Median)");
    print_separator('-', SUMMARY_WIDTH);

    ensure_experiment_dirs(csv_prefix(), compiler, compiler_dir);
    std::string general_path = build_path(compiler_dir, std::string(csv_prefix()) + "_general.csv");

    FILE *speedup_csv = fopen(general_path.c_str(), "w");
    fprintf(speedup_csv, "variante,speedup_geral_mean,speedup_geral_median\n");

    for (int v = 0; v < variant_count(); v++) {
        double speedup_mean   = gs_count / gs_mean[v];
        double speedup_median = gs_count / gs_median[v];

        printf("  %-18s %17.2fx %17.2fx\n",
               variant_name(v).c_str(),
               speedup_mean,
               speedup_median);

        fprintf(speedup_csv, "%s,%.4f,%.4f\n",
                variant_name(v).c_str(),
                speedup_mean,
                speedup_median);
    }

    print_separator('=', SUMMARY_WIDTH);
    printf("\n");

    fclose(speedup_csv);

    return 0;
}