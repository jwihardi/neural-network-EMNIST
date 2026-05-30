#include <stdexcept>
#include <random>
#include <cstddef>
#include "matrix.hpp"
#include "data_loader.hpp"

Matrix::Matrix(int rows_, int cols_) : data(rows_ * cols_), rows(rows_), cols(cols_) {}

void Matrix::multiply_into(const Matrix& other, Matrix& out) const{
    if(cols != other.rows)
        throw std::runtime_error("multiply_into: Invalid matrix dimensions (1)");
    if(out.rows != rows || out.cols != other.cols)
        throw std::runtime_error("multiply_into: Invalid matrix dimensions (2)");
    
    for(int i = 0; i < rows; i++){
        for(int u = 0; u < other.cols; u++){
            float curr_sum = 0.0f;
            for(int k = 0; k < cols; k++)
                curr_sum += data[i * cols + k] * other.data[k * other.cols + u];
            out.data[i * out.cols + u] = curr_sum;
        }
    }
}

void Matrix::transpose_multiply_into(const Matrix& other, Matrix& out) const{
    if(out.rows != cols || out.cols != 1)
        throw std::runtime_error("transpose_multiply_into: Invalid matrix dimensions");

    for(int i = 0; i < cols; i++){
        float curr_sum = 0.0f;
        for(int u = 0; u < rows; u++)
            curr_sum += data[u * cols + i] * other.data[u];
        out.data[i] = curr_sum;
    }
}

void Matrix::load_image_into(const Dataset& dataset, int start_idx, int batch_size, Matrix& out){
    int image_size = dataset.height * dataset.width;

    for(int b = 0; b < batch_size; b++){
        int curr_image_idx = start_idx + b;
        for(int pixel = 0; pixel < image_size; pixel++)
            out.data[pixel * out.cols + b] = dataset.images[image_size * curr_image_idx + pixel];
    }
}

void Matrix::hadamard_into(const Matrix& other, Matrix& out) const{
    if(cols != other.cols || rows != other.rows)
        throw std::runtime_error("hadamard_into: Invalid matrix dimensions (1)");
    if(out.rows != rows || out.cols != cols)
        throw std::runtime_error("hadamard_into: Invalid matrix dimensions (2)");
    for(std::size_t i = 0; i < data.size(); i++)
        out.data[i] = data[i] * other.data[i];
}

void Matrix::subtract_outer_product(const Matrix& col, const Matrix& row, float scale){
    for(int i = 0; i < rows; i++)
        for(int u = 0; u < cols; u++){
            data[i * cols + u] -= scale * col.data[i] * row.data[u];
        }
}

void Matrix::add(const Matrix& other){
    if(rows != other.rows || cols != other.cols)
        throw std::runtime_error("Invalid dimensions for matrix addition");

    for(int i = 0; i < rows; i++){
        for(int u = 0; u < cols; u++)
            data[i * cols + u] += other.data[i * cols + u];
    }
}

Matrix Matrix::init_he(int rows_, int cols_, std::mt19937& rand){
    Matrix matrix(rows_, cols_);
    float stddev = std::sqrt(2.0f / static_cast<float>(cols_));
    std::normal_distribution<float> dist(0.0f, stddev);
    for(std::size_t i = 0; i < matrix.data.size(); i++)
        matrix.data[i] = dist(rand);
    return matrix;
}

int Matrix::argmax() const{
    int best_idx = 0;
    float best_val = data[0];

    for(std::size_t i = 0; i < data.size(); i++){
        if(data[i] <= best_val) continue;
        best_val = data[i];
        best_idx = static_cast<int>(i);
    }
    return best_idx;
}

void Matrix::subtract_one_hot(int label){
    data[label] -= 1.0f;
}

void Matrix::subtract_scaled(const Matrix& mat, float scale){
    for(std::size_t i = 0; i < data.size(); i++)
        data[i] -= scale * mat.data[i];
}
