#include <immintrin.h>
#include <iostream>

#define SIZE 24

void print_array(const std::string message, const double *array, const unsigned int size);

int main() {
    double even_numbers[SIZE];
    double odd_numbers[SIZE];
    double result[SIZE];

    for (int i = 0; i < SIZE; ++i) {
        double pair = static_cast<double>(i * 2);
        double odd  = static_cast<double>(i * 2 + 1); 
        even_numbers[i] = pair;
        odd_numbers[i]  = odd;
    }

    print_array("Even numbers: ", even_numbers, SIZE);
    print_array("Odd numbers:  ", odd_numbers, SIZE);

    auto chunks = sizeof(__m512d) / sizeof(double);

    for (int i = 0; i + chunks <= SIZE; i += chunks) {
        __m512d vec_a = _mm512_loadu_pd(even_numbers + i);
        __m512d vec_b = _mm512_loadu_pd(odd_numbers + i);
        __m512d vec_result = _mm512_add_pd(vec_a, vec_b);
        _mm512_storeu_pd(result + i, vec_result);
    }

    print_array("Result:       ", result, SIZE);

    return 0;
}

void print_array(const std::string message, const double *array, const unsigned int size) {
    std::cout << message;
    for (int i = 0; i < SIZE; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}