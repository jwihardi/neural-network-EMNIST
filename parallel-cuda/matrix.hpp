#pragma once

#include <vector>
#include <random>
#include <cstdint>

struct Dataset;

struct DeviceDataset{
    float *images = nullptr;
    uint8_t *labels = nullptr;
    int num_samples, image_size;

    explicit DeviceDataset(const Dataset&);
    ~DeviceDataset();
    DeviceDataset(const DeviceDataset&) = delete;
    DeviceDataset& operator=(const DeviceDataset&) = delete;
};

struct Metrics{
    float *loss = nullptr;
    int *correct = nullptr;

    Metrics();
    ~Metrics();
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;

    void reset();
    void read(float&, int&) const;
};

struct Matrix{
    float *data = nullptr; // device pointer
    int rows, cols;

    Matrix(int, int);
    ~Matrix();
    Matrix(Matrix&&) noexcept;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    static Matrix init_he(int, int, std::mt19937&);
    static void load_image_into(const DeviceDataset&, int, int, Matrix&);

    void multiply_into(const Matrix&, Matrix&) const;
    void transpose_multiply_into(const Matrix&, Matrix&) const;
    void hadamard_into(const Matrix&, Matrix&) const;

    void relu_into(Matrix&) const;
    void relu_derivative_into(Matrix&) const;
    void softmax_into(Matrix&) const;

    void add(const Matrix&);
    void subtract_scaled(const Matrix&, float);
    void subtract_outer_product(const Matrix&, const Matrix&, float);
    void subtract_one_hot(const DeviceDataset&, int);

    void accumulate_metrics(const DeviceDataset&, int, Metrics&) const;
};
