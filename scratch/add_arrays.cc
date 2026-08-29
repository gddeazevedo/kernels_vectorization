#include <iostream>
#include <immintrin.h>
#include <string.h>

#define SIZE 1000000000
#define OPTION_1 "naive"
#define OPTION_2 "intrin"

void naive_add_arrays();
void intrinsics_add_arrays();

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <" << OPTION_1 << "|" << OPTION_2 << ">" << std::endl;
        return 1;
    }

    std::cout << "Selected option: " << argv[1] << std::endl;

    char* option = argv[1];

    if (strcmp(option, OPTION_1) == 0) {
        naive_add_arrays();
    } else if (strcmp(option, OPTION_2) == 0) {
        intrinsics_add_arrays();
    } else {
        std::cerr << "Invalid option. Use '" << OPTION_1 << "' or '" << OPTION_2 << "'." << std::endl;
        return 1;
    }

    return 0;
}

void naive_add_arrays() {
    double *even_numbers = new double[SIZE];
    double *odd_numbers = new double[SIZE];

    for (int i = 0; i < SIZE; i++) {
        double pair = static_cast<double>(i * 2);
        double odd  = static_cast<double>(i * 2 + 1); 
        even_numbers[i] = pair;
        odd_numbers[i]  = odd;
    }

    for (int i = 0; i < SIZE; i++) {
        even_numbers[i] = even_numbers[i] + odd_numbers[i];
    }

    delete[] even_numbers;
    delete[] odd_numbers;

    std::cout << "Naive addition completed." << std::endl;
}

void intrinsics_add_arrays() {
    double *even_numbers = new double[SIZE];
    double *odd_numbers = new double[SIZE];

    for (int i = 0; i < SIZE; i++) {
        double pair = static_cast<double>(i * 2);
        double odd  = static_cast<double>(i * 2 + 1); 
        even_numbers[i] = pair;
        odd_numbers[i]  = odd;
    }

    auto chunks = sizeof(__m512d) / sizeof(double);

    int i = 0;
    for (; i + chunks <= SIZE; i += chunks) {
        __m512d vec_even = _mm512_loadu_pd(even_numbers + i);
        __m512d vec_odd = _mm512_loadu_pd(odd_numbers + i);
        __m512d vec_result = _mm512_add_pd(vec_even, vec_odd);
        _mm512_storeu_pd(even_numbers + i, vec_result);
    }

    for (; i < SIZE; i++) {
        even_numbers[i] = even_numbers[i] + odd_numbers[i];
    }

    delete[] even_numbers;
    delete[] odd_numbers;

    std::cout << "Intrinsics addition completed." << std::endl;
}
