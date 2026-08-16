#include <immintrin.h>
#include <stdio.h>

int main() {
    __m512d vec = _mm512_set_pd(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    for (int i = 0; i < 8; i++) {
        double val = vec[i];
        printf("Element %d: %f\n", i, val);
    }

    double sum = _mm512_reduce_add_pd(vec);

    printf("The sum of elements is: %f\n", sum);

    // 0x0f = 0b00001111
    double half_sum = _mm512_mask_reduce_add_pd(0x0f, vec); // sum first four elements

    printf("The sum of the first four elements is: %f\n", half_sum);

    return 0;
}
