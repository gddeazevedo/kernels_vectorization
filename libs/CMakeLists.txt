# bcsr
add_library(
    bcsr
    bcsr/bcsr.cpp
)
target_include_directories(bcsr PUBLIC bcsr)

# block_ops
add_library(
    block_ops
    block_ops/block_ops.cpp
)
target_include_directories(block_ops PUBLIC block_ops)
target_link_libraries(block_ops PUBLIC hwy OpenMP::OpenMP_CXX)

# spmv
add_library(
    spmv
    spmv/spmv.cpp
)
target_include_directories(spmv PUBLIC spmv)
target_link_libraries(spmv PUBLIC bcsr)

add_library(
    utils
    utils/utils.cpp
)
target_include_directories(utils PUBLIC utils)

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
target_include_directories(spmv_benchmarks PUBLIC benchmarks/spmv)
target_link_libraries(spmv_benchmarks PUBLIC bcsr spmv utils benchmark_base)

add_library(ilu0 ilu0/ilu0.cpp)
target_include_directories(ilu0 PUBLIC ilu0)
target_link_libraries(ilu0 PUBLIC bcsr block_ops)

# benchmarks/ilu0
add_library(
    ilu0_benchmarks
    benchmarks/ilu0/ilu0_benchmarks.cpp
)
target_include_directories(ilu0_benchmarks PUBLIC benchmarks/ilu0)
target_link_libraries(ilu0_benchmarks PUBLIC bcsr ilu0 utils benchmark_base)

add_library(cli cli/cli.cpp)
target_include_directories(cli PUBLIC cli)
target_link_libraries(cli PUBLIC spmv_benchmarks ilu0_benchmarks)