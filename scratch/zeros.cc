#include <iostream>
#include <immintrin.h>


int main() {
    __m256d zeros = _mm256_setzero_pd();
    __m256d ones = _mm256_set1_pd(1.0);

    for (int i = 0; i < 4; i++) {
        double zero = ((double*)&zeros)[i];
        double one  = ((double*)&ones)[i];
        std::cout << "zeros[" << i << "] = " << zero << std::endl;
        std::cout << "ones[" << i << "] = " << one << std::endl;  
    }
}