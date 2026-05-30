#pragma once

#include <vector>
#include <random>

struct Dataset;

struct Matrix{
    std::vector<float> data;
    int rows, cols;

    Matrix(int, int);
    static Matrix load_image_mat(const Dataset&, int, int);
    static Matrix init_he(int, int, std::mt19937&);

    Matrix multiply(const Matrix&) const;
    Matrix transpose_multiply(const Matrix&) const;
    Matrix hadamard(const Matrix&) const;

    void add(const Matrix&);
    void subtract_scaled(const Matrix&, float);
    void subtract_outer_product(const Matrix&, const Matrix&, float);
    void subtract_one_hot(int);

    int argmax() const;
};
