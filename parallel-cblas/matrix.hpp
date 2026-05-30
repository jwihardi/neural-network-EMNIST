#pragma once

#include <vector>
#include <random>
#include <cstdint>

struct Dataset;

struct Matrix{
    std::vector<float> data;
    int rows, cols;

    Matrix(int, int);
    static Matrix init_he(int, int, std::mt19937&);
    static void load_image_into(const Dataset&, int, int, Matrix&);

    void hadamard_into(const Matrix&, Matrix&) const;
    

    void add(const Matrix&);
    void subtract_scaled(const Matrix&, float);
    void subtract_one_hot(const std::vector<uint8_t>&, int);

    int argmax(int) const;
};
