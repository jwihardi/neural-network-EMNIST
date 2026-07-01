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
        return h;
    }();
    return handle;
}

__global__ void normalize_kernel(const uint8_t *in, float *out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) out[idx] = in[idx] / 255.0f;
}

__global__ void gather_batch_kernel(const float *images, float *out, int start, int batch_size, int image_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= image_size * batch_size) return;

    int pixel = idx / batch_size;
    int b = idx % batch_size;
    out[idx] = images[(start + b) * image_size + pixel];
}

__global__ void add_kernel(float *a, const float *b, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) a[idx] += b[idx];
}

__global__ void bias_add_kernel(float *a, const float *bias, int cols, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) a[idx] += bias[idx / cols];
}

__global__ void hadamard_kernel(const float *a, const float *b, float *out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void relu_kernel(const float *in, float *out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void relu_derivative_kernel(const float *in, float *out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) out[idx] = (in[idx] > 0.0f) ? 1.0f : 0.0f;
}

__global__ void softmax_kernel(const float *in, float *out, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols) return;

    float max_val = in[col];
    for(int u = 0; u < rows; u++)
        max_val = fmaxf(max_val, in[u * cols + col]);

    float sum = 0.0f;
    for(int u = 0; u < rows; u++){
        float exponent = expf(in[u * cols + col] - max_val);
        out[u * cols + col] = exponent;
        sum += exponent;
    }
    for(int u = 0; u < rows; u++)
        out[u * cols + col] /= sum;
}

__global__ void subtract_scaled_kernel(float *bias, const float *mat, int rows, int batch_size, float scale){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows) return;

    float curr_sum = 0.0f;
    for(int u = 0; u < batch_size; u++)
        curr_sum += mat[row * batch_size + u];
    bias[row] -= scale * curr_sum;
}

__global__ void subtract_one_hot_kernel(float *data, const uint8_t *labels, int start, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols) return;

    int label = labels[start + col];
    data[label * cols + col] -= 1.0f;
}

__global__ void metrics_kernel(const float *predictions, const uint8_t *labels, int start,
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
    normalize_kernel<<<blocks_for(total), BLOCK>>>(staging, images, total);

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

Matrix::Matrix(int rows_, int cols_) : rows(rows_), cols(cols_){
    std::size_t bytes = sizeof(float) * static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
    CUDA_CHECK(cudaMalloc(&data, bytes));
    CUDA_CHECK(cudaMemset(data, 0, bytes));
}

Matrix::~Matrix(){
    if(data) cudaFree(data);
}

Matrix::Matrix(Matrix&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols){
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

void Matrix::load_image_into(const DeviceDataset& dataset, int start_idx, int batch_size, Matrix& out){
    int total = dataset.image_size * batch_size;
    gather_batch_kernel<<<blocks_for(total), BLOCK>>>(dataset.images, out.data, start_idx, batch_size, dataset.image_size);
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
        other.data, other.cols,
        data, cols,
        &beta,
        out.data, other.cols));
}

void Matrix::transpose_multiply_into(const Matrix& other, Matrix& out) const{
    if(out.rows != cols || out.cols != other.cols || other.rows != rows)
        throw std::runtime_error("transpose_multiply_into: Invalid matrix dimensions");

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        other.cols, cols, rows,
        &alpha,
        other.data, other.cols,
        data, cols,
        &beta,
        out.data, other.cols));
}

void Matrix::hadamard_into(const Matrix& other, Matrix& out) const{
    if(cols != other.cols || rows != other.rows)
        throw std::runtime_error("hadamard_into: Invalid matrix dimensions (1)");
    if(out.rows != rows || out.cols != cols)
        throw std::runtime_error("hadamard_into: Invalid matrix dimensions (2)");

    int n = rows * cols;
    hadamard_kernel<<<blocks_for(n), BLOCK>>>(data, other.data, out.data, n);
}

void Matrix::relu_into(Matrix& out) const{
    int n = rows * cols;
    relu_kernel<<<blocks_for(n), BLOCK>>>(data, out.data, n);
}

void Matrix::relu_derivative_into(Matrix& out) const{
    int n = rows * cols;
    relu_derivative_kernel<<<blocks_for(n), BLOCK>>>(data, out.data, n);
}

void Matrix::softmax_into(Matrix& out) const{
    softmax_kernel<<<blocks_for(cols), BLOCK>>>(data, out.data, rows, cols);
}

void Matrix::add(const Matrix& other){
    if(rows != other.rows)
        throw std::runtime_error("add: Invalid dimensions for matrix addition (1)");

    int n = rows * cols;
    if(cols == other.cols) // normal matrix addition
        add_kernel<<<blocks_for(n), BLOCK>>>(data, other.data, n);
    else if(other.cols == 1) // broadcast addition over all columns
        bias_add_kernel<<<blocks_for(n), BLOCK>>>(data, other.data, cols, n);
    else
        throw std::runtime_error("add: Invalid dimensions for matrix addition (2)");
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
        row.data, batch_size,
        col.data, batch_size,
        &beta,
        data, cols));
}

void Matrix::subtract_one_hot(const DeviceDataset& dataset, int start){
    subtract_one_hot_kernel<<<blocks_for(cols), BLOCK>>>(data, dataset.labels, start, cols);
}

void Matrix::accumulate_metrics(const DeviceDataset& dataset, int start, Metrics& metrics) const{
    metrics_kernel<<<blocks_for(cols), BLOCK>>>(data, dataset.labels, start, rows, cols, metrics.loss, metrics.correct);
}
