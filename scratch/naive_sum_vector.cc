#include <stdio.h>
#include <vector>

#define VECTOR_SIZE 10000000

int main() {
    std::vector<int> v(VECTOR_SIZE);
    int sum = 0;

    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        v[i] = i;
    }

    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
    }

    printf("Sum of vector elements: %d\n", sum);

    return 0;
}