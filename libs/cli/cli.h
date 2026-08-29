#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <spmv_benchmarks.h>
#include <ilu0_benchmarks.h>

class CLI
{
    public:
        CLI(int argc, char **argv);
        int run();

    private:
        int argc;
        char **argv;
        std::string operation;
        std::string compiler;

        void print_usage(const char *prog);
        int run_spmv();
    int run_ilu0();
};