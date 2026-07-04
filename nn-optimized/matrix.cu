#include <stdexcept>
#include <random>
#include <cstddef>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "matrix.hpp"
#include "data_loader.hpp"

#define CUBLAS_CHECK(call)                                                                          \
    do{                                                                                             \
        if((call) != CUBLAS_STATUS_SUCCESS)                                                         \
            throw std::runtime_error("cublas call failed");                                         \
    }while(0)

#define CUSOLVER_CHECK(call)                                                                        \
    do{                                                                                             \
        if((call) != CUSOLVER_STATUS_SUCCESS)                                                       \
            throw std::runtime_error("cusolver call failed");                                       \
    }while(0)

static constexpr int BLOCK = 256;
static constexpr float EPSILON = 1e-7f;

static inline int blocks_for(int n){ return (n + BLOCK - 1) / BLOCK; }

cudaStream_t gpu_stream(){
    static cudaStream_t stream = []{
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        return s;
    }();
    return stream;
}

static cublasHandle_t cublas_handle(){
    static cublasHandle_t handle = []{
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
        CUBLAS_CHECK(cublasSetStream(h, gpu_stream()));
        return h;
    }();
    return handle;
}

static cusolverDnHandle_t cusolver_handle(){
    static cusolverDnHandle_t handle = []{
        cusolverDnHandle_t h;
        CUSOLVER_CHECK(cusolverDnCreate(&h));
        CUSOLVER_CHECK(cusolverDnSetStream(h, gpu_stream()));
        return h;
    }();
    return handle;
}

/* pixel major so a chunk is just a column range of the full dataset */
__global__ void normalize_transpose_kernel(const uint8_t *in, float *out, int num_samples, int image_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_samples * image_size) return;

    int pixel = idx / num_samples;
    int sample = idx % num_samples;
    out[idx] = in[sample * image_size + pixel] / 255.0f;
}

__global__ void bias_relu_kernel(float *a, const float *bias, int cols, int ld, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    int row = idx / cols, col = idx % cols;
    a[row * ld + col] = fmaxf(0.0f, a[row * ld + col] + bias[row]);
}

__global__ void one_hot_kernel(float *y, const uint8_t *labels, int start, int cols, int ld, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    int row = idx / cols, col = idx % cols;
    y[row * ld + col] = (labels[start + col] == row) ? 1.0f : 0.0f;
}

__global__ void add_diagonal_kernel(float *a, int rows, int ld, float value){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows) a[row * ld + row] += value;
}

__global__ void metrics_kernel(const float *scores, const uint8_t *labels, int start,
                               int rows, int cols, int ld, float *loss, int *correct){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols) return;

    int best_idx = 0;
    float best_val = scores[col];
    for(int u = 1; u < rows; u++){
        float val = scores[u * ld + col];
        if(val > best_val){
            best_val = val;
            best_idx = u;
        }
    }

    int actual = labels[start + col];
    if(best_idx == actual) atomicAdd(correct, 1);

    /* scores are regression outputs not logits, softmax here just gives a loss comparable to the other versions */
    float sum = 0.0f;
    for(int u = 0; u < rows; u++)
        sum += expf(scores[u * ld + col] - best_val);
    float p = expf(scores[actual * ld + col] - best_val) / sum;
    atomicAdd(loss, -logf(fmaxf(p, EPSILON)));
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
    normalize_transpose_kernel<<<blocks_for(total), BLOCK, 0, gpu_stream()>>>(staging, images, num_samples, image_size);

    CUDA_CHECK(cudaMalloc(&labels, dataset.labels.size()));
    CUDA_CHECK(cudaMemcpy(labels, dataset.labels.data(), dataset.labels.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamSynchronize(gpu_stream()));
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
    CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), gpu_stream()));
    CUDA_CHECK(cudaMemsetAsync(correct, 0, sizeof(int), gpu_stream()));
}

void Metrics::read(float& loss_out, int& correct_out) const{
    CUDA_CHECK(cudaMemcpyAsync(&loss_out, loss, sizeof(float), cudaMemcpyDeviceToHost, gpu_stream()));
    CUDA_CHECK(cudaMemcpyAsync(&correct_out, correct, sizeof(int), cudaMemcpyDeviceToHost, gpu_stream()));
    CUDA_CHECK(cudaStreamSynchronize(gpu_stream()));
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

Matrix Matrix::init_uniform(int rows_, int cols_, float range, std::mt19937& rand){
    Matrix matrix(rows_, cols_);
    std::uniform_real_distribution<float> dist(-range, range);

    std::vector<float> host(static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_));
    for(std::size_t i = 0; i < host.size(); i++)
        host[i] = dist(rand);

    CUDA_CHECK(cudaMemcpy(matrix.data, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice));
    return matrix;
}

Matrix Matrix::batch_view(const DeviceDataset& dataset, int start_idx, int batch_size){
    return Matrix(dataset.images + start_idx, dataset.image_size, batch_size, dataset.num_samples);
}

Matrix Matrix::first_cols(int n) const{
    return Matrix(data, rows, n, ld);
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

void Matrix::bias_relu(const Matrix& bias){
    if(bias.rows != rows || bias.cols != 1)
        throw std::runtime_error("bias_relu: Invalid matrix dimensions");

    int n = rows * cols;
    bias_relu_kernel<<<blocks_for(n), BLOCK, 0, gpu_stream()>>>(data, bias.data, cols, ld, n);
}

void Matrix::one_hot(const DeviceDataset& dataset, int start){
    int n = rows * cols;
    one_hot_kernel<<<blocks_for(n), BLOCK, 0, gpu_stream()>>>(data, dataset.labels, start, cols, ld, n);
}

/* A += H Hᵀ, one triangle only, potrf below reads the same one */
void Matrix::gram_accumulate(const Matrix& h){
    if(h.rows != rows || rows != cols)
        throw std::runtime_error("gram_accumulate: Invalid matrix dimensions");

    const float alpha = 1.0f, beta = 1.0f;
    CUBLAS_CHECK(cublasSsyrk(cublas_handle(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
        rows, h.cols,
        &alpha,
        h.data, h.ld,
        &beta,
        data, ld));
}

/* B += H Yᵀ, accumulated column major so cusolver can take it directly */
void Matrix::target_accumulate(const Matrix& h, const Matrix& y){
    if(rows != y.rows || cols != h.rows || h.cols != y.cols)
        throw std::runtime_error("target_accumulate: Invalid matrix dimensions");

    const float alpha = 1.0f, beta = 1.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        cols, rows, h.cols,
        &alpha,
        h.data, h.ld,
        y.data, y.ld,
        &beta,
        data, ld));
}

void Matrix::add_diagonal(float value){
    add_diagonal_kernel<<<blocks_for(rows), BLOCK, 0, gpu_stream()>>>(data, rows, ld, value);
}

void Matrix::cholesky_solve(Matrix& rhs){
    const int n = rows;
    const int nrhs = rhs.rows;

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver_handle(), CUBLAS_FILL_MODE_LOWER, n, data, ld, &lwork));

    float *workspace = nullptr;
    int *info = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(float) * static_cast<std::size_t>(lwork)));
    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnSpotrf(cusolver_handle(), CUBLAS_FILL_MODE_LOWER, n, data, ld, workspace, lwork, info));
    CUSOLVER_CHECK(cusolverDnSpotrs(cusolver_handle(), CUBLAS_FILL_MODE_LOWER, n, nrhs, data, ld, rhs.data, n, info));

    int host_info = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost, gpu_stream()));
    CUDA_CHECK(cudaStreamSynchronize(gpu_stream()));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info));

    if(host_info != 0)
        throw std::runtime_error("cholesky_solve: factorization failed, matrix not positive definite");
}

void Matrix::accumulate_metrics(const DeviceDataset& dataset, int start, Metrics& metrics) const{
    metrics_kernel<<<blocks_for(cols), BLOCK, 0, gpu_stream()>>>(data, dataset.labels, start, rows, cols, ld, metrics.loss, metrics.correct);
}
