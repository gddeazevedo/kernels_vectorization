# bcsr
add_library(
    bcsr
    bcsr/bcsr.cpp
)
target_include_directories(bcsr PUBLIC bcsr) # where are the headers for bcsr

# Add a library to the project using the specified source files
# spmv
add_library(
    spmv
    spmv/spmv.cpp
)
target_include_directories(spmv PUBLIC spmv) # where are the headers for spmv
target_link_libraries(spmv PUBLIC bcsr) # where are the dependencies for spmv


add_library(
    utils
    utils/utils.cpp
)
target_include_directories(utils PUBLIC utils) # where are the headers for utils

add_library(
    benchmark_base
    benchmarks/benchmark_base/benchmark_base.cpp
)
target_include_directories(benchmark_base PUBLIC benchmarks/benchmark_base)
target_link_libraries(benchmark_base PUBLIC bcsr utils)

# benchmarks/spmv
add_library(
    spmv_benchmarks
    benchmarks/spmv/spmv_benchmarks.cpp
)
target_include_directories(spmv_benchmarks PUBLIC benchmarks/spmv) # where are the headers for spmv_benchmarks
target_link_libraries(spmv_benchmarks PUBLIC bcsr spmv utils benchmark_base) # where are the dependencies for spmv_benchmarks


add_library(
    block_matrix_ops
    block_matrix_ops/block_matrix_ops.cpp
)
target_include_directories(block_matrix_ops PUBLIC block_matrix_ops) # where are the headers for helpers

add_library(ilu0 ilu0/ilu0.cpp)
target_include_directories(ilu0 PUBLIC ilu0) # where are the headers for ilu0
target_link_libraries(ilu0 PUBLIC bcsr block_matrix_ops) # where are the dependencies for ilu0

# benchmarks/ilu0
add_library(
    ilu0_benchmarks
    benchmarks/ilu0/ilu0_benchmarks.cpp
)
target_include_directories(ilu0_benchmarks PUBLIC benchmarks/ilu0) # where are the headers for ilu0_benchmarks
target_link_libraries(ilu0_benchmarks PUBLIC bcsr ilu0 utils benchmark_base) # where are the dependencies for ilu0_benchmarks


add_library(cli cli/cli.cpp)
target_include_directories(cli PUBLIC cli) # where are the headers for cli
target_link_libraries(cli PUBLIC spmv_benchmarks ilu0_benchmarks) # where are the dependencies for cli
