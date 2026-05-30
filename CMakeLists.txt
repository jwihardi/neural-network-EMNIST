cmake_minimum_required(VERSION 3.16)

project(neural_network LANGUAGES CXX)

# Put built executables in the project root, not inside build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# OpenMP for the parallel-omp target (adds -fopenmp to compile + link)
find_package(OpenMP REQUIRED)

# Shared flags applied to every target via add_nn_target() below
set(NN_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2

    -O3                     # aggressive optimization (vectorization, unrolling)
    -pipe
    -march=native           # Zen 4: unlock AVX-512/FMA for this CPU
    -funroll-loops          # unroll hot loops
    -floop-unroll-and-jam   # reuse loaded data across unrolled outer iterations (the matmul win)
    #    -ffast-math             # allow FP reassociation -> vectorized reduction loops (dot products)
)

# Helper: add_nn_target(<name> <sources...>)
# Defines an executable with the project's standard flags/features.
function(add_nn_target name)
    add_executable(${name} ${ARGN})
    target_compile_features(${name} PRIVATE cxx_std_20)
    target_compile_options(${name} PRIVATE ${NN_COMPILE_OPTIONS})
endfunction()

add_nn_target(unoptimized-serial-nn
    serial-unoptimized/neural_network.cpp
    serial-unoptimized/matrix.cpp
)

add_nn_target(optimized-serial-nn
    serial-optimized/neural_network.cpp
    serial-optimized/matrix.cpp
)

# Same code + flags as optimized-serial, plus OpenMP parallelism.
add_nn_target(parallel-omp-nn
    parallel-omp/neural_network.cpp
    parallel-omp/matrix.cpp
)
target_link_libraries(parallel-omp-nn PRIVATE OpenMP::OpenMP_CXX)

# Same code + flags as optimized-serial, with the hot GEMM/GER/AXPY ops
# offloaded to an external CBLAS (OpenBLAS recommended; it is internally threaded).
find_package(BLAS)
if(BLAS_FOUND)
    # FindBLAS provides the link target but not the cblas.h header location.
    find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(NOT CBLAS_INCLUDE_DIR)
        message(WARNING "BLAS library found but cblas.h was not; "
                        "parallel-cblas-nn may fail to compile. "
                        "Install the CBLAS headers (e.g. sci-libs/openblas).")
    endif()
    add_nn_target(parallel-cblas-nn
        parallel-cblas/neural_network.cpp
        parallel-cblas/matrix.cpp
    )
    target_link_libraries(parallel-cblas-nn PRIVATE BLAS::BLAS)
    if(CBLAS_INCLUDE_DIR)
        target_include_directories(parallel-cblas-nn PRIVATE ${CBLAS_INCLUDE_DIR})
    endif()
else()
    message(WARNING "BLAS not found; skipping parallel-cblas-nn target. "
                    "Install a CBLAS (e.g. 'emerge sci-libs/openblas') and re-run cmake.")
endif()
