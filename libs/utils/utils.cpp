#include <utils.h>

double wtime() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

double median(double *arr, int n) {
    double *tmp = (double *)malloc(n * sizeof(double));
    memcpy(tmp, arr, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double);
    double m = (n % 2 == 0) ? (tmp[n/2-1] + tmp[n/2]) / 2.0 : tmp[n/2];
    free(tmp);
    return m;
}

void ensure_dir(const char *path) {
    if (access(path, F_OK) == -1) {
        mkdir(path, 0755);
    }
}

void ensure_experiment_dirs(const std::string &prefix, const std::string &compiler, std::string &compiler_dir) {
    ensure_dir("experiments");
    std::string bench_dir = "experiments/" + prefix;
    ensure_dir(bench_dir.c_str());
    compiler_dir = bench_dir + "/" + compiler;
    ensure_dir(compiler_dir.c_str());
}

std::string build_path(const std::string &dir, const std::string &file) {
    return dir + "/" + file;
}

void print_separator(char ch, int width) {
    for (int i = 0; i < width; i++) {
        putchar(ch);
    }
    putchar('\n');
}