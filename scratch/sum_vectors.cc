#include <stdio.h>
#include <vector>
#include <omp.h>

#define VECTOR_SIZE 10000000

int main() {
    std::vector<int> v(VECTOR_SIZE);
    std::vector<int> u(VECTOR_SIZE);
    std::vector<int> w(VECTOR_SIZE);

    omp_set_num_threads(1);

    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        v[i] = i;
        u[i] = i * 2;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); ++i) {
        w[i] = v[i] + u[i];
        // printf("Thread %d adding elements %zu (v: %d, u: %d) (w: %d)\n", omp_get_thread_num(), i, v[i], u[i], w[i]);
    }

    printf("%d\n", w[10000]);

    // printf("Resulting vector w elements: ");
    // for (const auto& val : w) {
    //     printf("%d ", val);
    // }
    // printf("\n");

    return 0;
}