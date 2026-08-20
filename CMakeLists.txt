cmake_minimum_required(VERSION 3.10)
project(vectorization)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)
find_package(hwy 1.3.0 REQUIRED)

add_subdirectory(libs)

add_executable(vectorization src/main.cpp)

target_link_libraries(vectorization
    PUBLIC
    spmv_benchmarks
    ilu0_benchmarks
    spmv
    bcsr
    cli
    ilu0
    block_ops
    utils
    benchmark_base
    OpenMP::OpenMP_CXX
    hwy
)