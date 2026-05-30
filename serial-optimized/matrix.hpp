#pragma once

#include <vector>
#include <random>
#include <cstdint>

#include "data_loader.hpp"

struct Matrix{
    std::vector<float> data;
    int rows, cols;

    Matrix(int, int);

    void multiply_into(const Matrix&, Matrix&) const;

    void transpose_multiply_into(const Matrix&, Matrix&) const;

    void hadamard_into(const Matrix&, Matrix&) const;
    
    static void load_image_into(const Dataset&, int, int, Matrix&);

    void subtract_outer_product(const Matrix&, const Matrix&, float);

    void add(const Matrix&);

    static Matrix init_he(int, int, std::mt19937&);

    int argmax(int) const;
    
    void subtract_one_hot(const std::vector<uint8_t>&, int);

    void subtract_scaled(const Matrix&, float);
};
