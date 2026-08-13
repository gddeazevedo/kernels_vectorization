cmake_minimum_required(VERSION 3.10)
project(vectorization)

set(CMAKE_CXX_STANDARD 23)

# Find a package (usually provided by something external to the project), and load its package-specific details.
# Calls to this command can also be intercepted by dependency providers.
find_package(OpenMP REQUIRED)
find_package(hwy 1.3.0 REQUIRED)

add_subdirectory(libs)

add_executable(vectorization src/main.cpp)

# Specify libraries or flags to use when linking a given target and/or its dependents. 
# Usage requirements from linked library targets will be propagated. 
# Usage requirements of a target's dependencies affect compilation of its own sources
target_link_libraries(vectorization
    PUBLIC
    spmv_benchmarks
    ilu0_benchmarks
    spmv
    bcsr
    cli
    ilu0
    utils
    block_matrix_ops
    benchmark_base
    OpenMP::OpenMP_CXX
    hwy
) # where are the dependencies for the main executable