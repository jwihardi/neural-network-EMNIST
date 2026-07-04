#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

struct Dataset;

#define CUDA_CHECK(call)                                                                            \
    do{                                                                                             \
        cudaError_t err_ = (call);                                                                  \
        if(err_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(err_));             \
    }while(0)

// single stream all work goes on, so an epoch can be captured into a graph
cudaStream_t gpu_stream();

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

// schedule state lives on the device so graph replays see fresh values
struct OneCycle{
    float *scale = nullptr; // current lr / batch size, the update kernels read this
    int *step = nullptr;
    int total_steps;
    float max_lr, inv_batch;

    OneCycle(int, float, int);
    ~OneCycle();
    OneCycle(const OneCycle&) = delete;
    OneCycle& operator=(const OneCycle&) = delete;

    void tick();
};

struct Matrix{
    float *data = nullptr; // device pointer
    int rows, cols;
    int ld; // row stride, lets a matrix be a column slice of a bigger one
    bool owns = true;

    Matrix(int, int);
    Matrix(float*, int, int, int); // non owning view
    ~Matrix();
    Matrix(Matrix&&) noexcept;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    static Matrix init_he(int, int, std::mt19937&);
    static Matrix batch_view(const DeviceDataset&, int, int);

    void multiply_into(const Matrix&, Matrix&) const;
    void transpose_multiply_into(const Matrix&, Matrix&) const;

    void bias_relu(const Matrix&);
    void softmax_bias_into(const Matrix&, Matrix&) const;
    void relu_backward(const Matrix&);

    void subtract_scaled(const Matrix&, Matrix&, float, const float*);
    void accumulate_outer_product(const Matrix&, const Matrix&, float);
    void subtract_velocity(const Matrix&, const float*);

    void accumulate_metrics(const DeviceDataset&, int, Metrics&);
};
