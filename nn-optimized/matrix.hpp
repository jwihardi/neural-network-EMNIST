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

// single stream all work goes on
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

struct Matrix;

struct StagedChunk{
    uint8_t *host = nullptr; // pinned
    uint8_t *raw = nullptr;
    float *pixels = nullptr;
    int image_size, max_samples;
    cudaEvent_t done;

    StagedChunk(int, int);
    ~StagedChunk();
    StagedChunk(const StagedChunk&) = delete;
    StagedChunk& operator=(const StagedChunk&) = delete;

    void wait();
    Matrix upload(int);
    void record();
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
    static Matrix init_uniform(int, int, float, std::mt19937&);
    static Matrix batch_view(const DeviceDataset&, int, int);
    Matrix first_cols(int) const;

    void multiply_into(const Matrix&, Matrix&) const;
    void bias_relu(const Matrix&);

    void one_hot(const uint8_t*, int);
    void gram_accumulate(const Matrix&);
    void target_accumulate(const Matrix&, const Matrix&);
    void add_diagonal(float);
    void cholesky_solve(Matrix&); // factors this in place, solution lands in the rhs

    void accumulate_metrics(const DeviceDataset&, int, Metrics&) const;
};
