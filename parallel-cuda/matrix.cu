#include <stdexcept>
#include <random>
#include <cstddef>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matrix.hpp"
#include "data_loader.hpp"
#include "losses.hpp"

#define CUDA_CHECK(call)                                                                            \
    do{                                                                                             \
        cudaError_t err_ = (call);                                                                  \
        if(err_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(err_));             \
    }while(0)

#define CUBLAS_CHECK(call)                                                                          \
    do{                                                                                             \
        if((call) != CUBLAS_STATUS_SUCCESS)                                                         \
            throw std::runtime_error("cublas call failed");                                         \
    }while(0)

static constexpr int BLOCK = 256;

static inline int blocks_for(int n){ return (n + BLOCK - 1) / BLOCK; }

static cublasHandle_t cublas_handle(){
    static cublasHandle_t handle = []{
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
        return h;
    }();
    return handle;
}

/* pixel major so a batch is just a column range of the full dataset */
__global__ void normalize_transpose_kernel(const uint8_t *in, float *out, int num_samples, int image_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_samples * image_size) return;

    int pixel = idx / num_samples;
    int sample = idx % num_samples;
    out[idx] = in[sample * image_size + pixel] / 255.0f;
}

__global__ void bias_relu_kernel(float *a, const float *bias, int cols, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) a[idx] = fmaxf(0.0f, a[idx] + bias[idx / cols]);
}

__global__ void softmax_bias_kernel(const float *in, const float *bias, float *out, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols) return;

    float max_val = in[col] + bias[0];
    for(int u = 0; u < rows; u++)
        max_val = fmaxf(max_val, in[u * cols + col] + bias[u]);

    float sum = 0.0f;
    for(int u = 0; u < rows; u++){
        float exponent = expf(in[u * cols + col] + bias[u] - max_val);
        out[u * cols + col] = exponent;
        sum += exponent;
    }
    for(int u = 0; u < rows; u++)
        out[u * cols + col] /= sum;
}

__global__ void relu_backward_kernel(float *grad, const float *activation, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n && activation[idx] <= 0.0f) grad[idx] = 0.0f;
}

__global__ void subtract_scaled_kernel(float *bias, const float *mat, int rows, int batch_size, float scale){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows) return;

    float curr_sum = 0.0f;
    for(int u = 0; u < batch_size; u++)
        curr_sum += mat[row * batch_size + u];
    bias[row] -= scale * curr_sum;
}

__global__ void metrics_kernel(float *predictions, const uint8_t *labels, int start,
                               int rows, int cols, float *loss, int *correct){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols) return;

    int best_idx = 0;
    float best_val = predictions[col];
    for(int u = 1; u < rows; u++){
        float val = predictions[u * cols + col];
        if(val > best_val){
            best_val = val;
            best_idx = u;
        }
    }

    int actual = labels[start + col];
    if(best_idx == actual) atomicAdd(correct, 1);

    float p = fmaxf(predictions[actual * cols + col], Losses::EPSILON);
    atomicAdd(loss, -logf(p));

    // subtract the one hot while we're here, saves backprop a launch
    predictions[actual * cols + col] -= 1.0f;
}

DeviceDataset::DeviceDataset(const Dataset& dataset) :
    num_samples(static_cast<int>(dataset.num_samples)),
    image_size(static_cast<int>(dataset.height * dataset.width))
{
    uint8_t *staging = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, dataset.images.size()));
    CUDA_CHECK(cudaMemcpy(staging, dataset.images.data(), dataset.images.size(), cudaMemcpyHostToDevice));

    int total = num_samples * image_size;
    CUDA_CHECK(cudaMalloc(&images, sizeof(float) * dataset.images.size()));
    normalize_transpose_kernel<<<blocks_for(total), BLOCK>>>(staging, images, num_samples, image_size);

    CUDA_CHECK(cudaMalloc(&labels, dataset.labels.size()));
    CUDA_CHECK(cudaMemcpy(labels, dataset.labels.data(), dataset.labels.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(staging));
}

DeviceDataset::~DeviceDataset(){
    if(images) cudaFree(images);
    if(labels) cudaFree(labels);
}

Metrics::Metrics(){
    CUDA_CHECK(cudaMalloc(&loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&correct, sizeof(int)));
    reset();
}

Metrics::~Metrics(){
    if(loss) cudaFree(loss);
    if(correct) cudaFree(correct);
}

void Metrics::reset(){
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(correct, 0, sizeof(int)));
}

void Metrics::read(float& loss_out, int& correct_out) const{
    CUDA_CHECK(cudaMemcpy(&loss_out, loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&correct_out, correct, sizeof(int), cudaMemcpyDeviceToHost));
}

Matrix::Matrix(int rows_, int cols_) : rows(rows_), cols(cols_), ld(cols_){
    std::size_t bytes = sizeof(float) * static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
    CUDA_CHECK(cudaMalloc(&data, bytes));
    CUDA_CHECK(cudaMemset(data, 0, bytes));
}

Matrix::Matrix(float *data_, int rows_, int cols_, int ld_) :
    data(data_), rows(rows_), cols(cols_), ld(ld_), owns(false) {}

Matrix::~Matrix(){
    if(data && owns) cudaFree(data);
}

Matrix::Matrix(Matrix&& other) noexcept :
    data(other.data), rows(other.rows), cols(other.cols), ld(other.ld), owns(other.owns){
    other.data = nullptr;
}

Matrix Matrix::init_he(int rows_, int cols_, std::mt19937& rand){
    Matrix matrix(rows_, cols_);
    float stddev = std::sqrt(2.0f / static_cast<float>(cols_));
    std::normal_distribution<float> dist(0.0f, stddev);

    std::vector<float> host(static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_));
    for(std::size_t i = 0; i < host.size(); i++)
        host[i] = dist(rand);

    CUDA_CHECK(cudaMemcpy(matrix.data, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice));
    return matrix;
}

Matrix Matrix::batch_view(const DeviceDataset& dataset, int start_idx, int batch_size){
    return Matrix(dataset.images + start_idx, dataset.image_size, batch_size, dataset.num_samples);
}

void Matrix::multiply_into(const Matrix& other, Matrix& out) const{
    if(cols != other.rows)
        throw std::runtime_error("multiply_into: Invalid matrix dimensions (1)");
    if(out.rows != rows || out.cols != other.cols)
        throw std::runtime_error("multiply_into: Invalid matrix dimensions (2)");

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        other.cols, rows, cols,
        &alpha,
        other.data, other.ld,
        data, ld,
        &beta,
        out.data, out.ld));
}

void Matrix::transpose_multiply_into(const Matrix& other, Matrix& out) const{
    if(out.rows != cols || out.cols != other.cols || other.rows != rows)
        throw std::runtime_error("transpose_multiply_into: Invalid matrix dimensions");

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        other.cols, cols, rows,
        &alpha,
        other.data, other.ld,
        data, ld,
        &beta,
        out.data, out.ld));
}

void Matrix::bias_relu(const Matrix& bias){
    if(bias.rows != rows || bias.cols != 1)
        throw std::runtime_error("bias_relu: Invalid matrix dimensions");

    int n = rows * cols;
    bias_relu_kernel<<<blocks_for(n), BLOCK>>>(data, bias.data, cols, n);
}

void Matrix::softmax_bias_into(const Matrix& bias, Matrix& out) const{
    if(bias.rows != rows || bias.cols != 1)
        throw std::runtime_error("softmax_bias_into: Invalid matrix dimensions");

    softmax_bias_kernel<<<blocks_for(cols), BLOCK>>>(data, bias.data, out.data, rows, cols);
}

void Matrix::relu_backward(const Matrix& activation){
    if(activation.rows != rows || activation.cols != cols)
        throw std::runtime_error("relu_backward: Invalid matrix dimensions");

    int n = rows * cols;
    relu_backward_kernel<<<blocks_for(n), BLOCK>>>(data, activation.data, n);
}

void Matrix::subtract_scaled(const Matrix& mat, float scale){
    subtract_scaled_kernel<<<blocks_for(rows), BLOCK>>>(data, mat.data, rows, mat.cols, scale);
}

void Matrix::subtract_outer_product(const Matrix& col, const Matrix& row, float scale){
    if(col.rows != rows || row.rows != cols || row.cols != col.cols) // lol weird
        throw std::runtime_error("subtract_outer_product: Invalid matrix dimensions");

    const int batch_size = col.cols;
    const float alpha = -scale, beta = 1.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        cols, rows, batch_size,
        &alpha,
        row.data, row.ld,
        col.data, col.ld,
        &beta,
        data, ld));
}

void Matrix::accumulate_metrics(const DeviceDataset& dataset, int start, Metrics& metrics){
    metrics_kernel<<<blocks_for(cols), BLOCK>>>(data, dataset.labels, start, rows, cols, metrics.loss, metrics.correct);
}
