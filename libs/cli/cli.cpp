#include <cli.h>

CLI::CLI(int argc, char **argv) : argc(argc), argv(argv) {}

void CLI::print_usage(const char *prog)
{
    printf("Uso: %s <operacao> <compilador>\n\n", prog);
    printf("Operacoes disponiveis:\n");
    printf("  spmv\n");
    printf("  ilu0\n");
}

int CLI::run()
{
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    operation = argv[1];
    compiler  = argv[2];

    if (operation == "spmv") {
        return run_spmv();
    } else if (operation == "ilu0") {
        return run_ilu0();
    } else {
        fprintf(stderr, "Operacao desconhecida: %s\n\n", operation.c_str());
        print_usage(argv[0]);
        return 1;
    }
}

int CLI::run_spmv()
{
    constexpr int ini = 20;
    constexpr int fim = 200;
    constexpr int inc = 20;
    constexpr int K   = 100;

    SpmvBenchmark benchmark(ini, fim, inc, K, compiler);
    return benchmark.run();
}

int CLI::run_ilu0()
{
    constexpr int ini = 20;
    constexpr int fim = 160;
    constexpr int inc = 20;

    constexpr int K   = 100;

    Ilu0Benchmark benchmark(ini, fim, inc, K, compiler);
    return benchmark.run();
}