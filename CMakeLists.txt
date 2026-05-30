cmake_minimum_required(VERSION 3.16)

project(neural_network LANGUAGES CXX)

# Put built executables in the project root, not inside build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

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
