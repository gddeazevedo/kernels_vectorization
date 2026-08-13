#pragma once

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <climits>
#include <unistd.h>

double wtime();
int cmp_double(const void *a, const void *b);
double median(double *arr, int n);
void ensure_dir(const char *path);
void ensure_experiment_dirs(const std::string &prefix, const std::string &compiler, std::string &compiler_dir);
std::string build_path(const std::string &dir, const std::string &file);
void print_separator(char ch = '=', int width = 60);