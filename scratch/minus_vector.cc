#include <vector>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>

int main() {
    const std::vector<int32_t> v = {10, 20, 30, 40, 50};
    int32_t result = 0;

    omp_set_num_threads(4);
    #pragma omp parallel for reduction(-:result)
    for (size_t i = 0; i < v.size(); ++i) {
        result -= v[i];
    }

    printf("Result of subtracting vector elements: %d\n", result);

    return 0;
}