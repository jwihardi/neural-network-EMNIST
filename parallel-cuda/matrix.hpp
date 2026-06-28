#pragma once

#include <vector>
#include <random>
#include <cstdint>

struct Dataset;

struct Matrix{
    float *data = nullptr; // device pointer
    int rows, cols;

    Matrix(int, int);
    ~Matrix();
    Matrix(Matrix&&) noexcept;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    static Matrix init_he(int, int, std::mt19937&);
    static void load_image_into(const Dataset&, int, int, Matrix&);

    void multiply_into(const Matrix&, Matrix&) const;
    void transpose_multiply_into(const Matrix&, Matrix&) const;
    void hadamard_into(const Matrix&, Matrix&) const;
    

    void add(const Matrix&);
    void subtract_scaled(const Matrix&, float);
    void subtract_outer_product(const Matrix&, const Matrix&, float);
    void subtract_one_hot(const std::vector<uint8_t>&, int);

    int argmax(int) const;
};
