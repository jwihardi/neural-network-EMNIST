#include <stdexcept>
#include <random>
#include <cstddef>
#include "matrix.hpp"
#include "data_loader.hpp"

Matrix::Matrix(int rows_, int cols_) : data(rows_ * cols_), rows(rows_), cols(cols_) {}

Matrix Matrix::multiply(const Matrix& other) const{
    if(cols != other.rows) 
        throw std::runtime_error("Invalid dimensions for matrix multiplication");

    Matrix res(rows, other.cols);

    for(int i = 0; i < rows; i++){
        for(int u = 0; u < other.cols; u++){
            float sum = 0.0f;
            for(int k = 0; k < cols; k++)
                sum += data[i * cols + k] * other.data[k * other.cols + u];
            res.data[i * res.cols + u] = sum;
        }
    }
    return res;
}

Matrix Matrix::transpose_multiply(const Matrix& other) const{
    Matrix res(cols, 1);

    for(int i = 0; i < cols; i++){
        float curr_sum = 0.0f;
        for(int u = 0; u < rows; u++)
            curr_sum += data[u * cols + i] * other.data[u];
        res.data[i] = curr_sum;
    }
    return res;
}

void Matrix::subtract_outer_product(const Matrix& col, const Matrix& row, float scale){
    for(int i = 0; i < rows; i++)
        for(int u = 0; u < cols; u++){
            data[i * cols + u] -= scale * col.data[i] * row.data[u];
        }
}

Matrix Matrix::hadamard(const Matrix& other) const{
    if(cols != other.cols || rows != other.rows) 
        throw std::runtime_error("Invalid dimensions for hadamard multiplication");

    Matrix res(rows, cols);

    for(int i = 0; i < rows; i++){  
        for(int u = 0; u < cols; u++){
            res.data[i * cols + u] = data[i * cols + u] * other.data[i * cols + u]; 
        }
    }
    return res;
}

Matrix Matrix::transpose() const{
    Matrix res(cols, rows);

    for(int i = 0; i < rows; i++){
        for(int u = 0; u < cols; u++){
            res.data[u * res.cols + i] = data[i * cols + u];
        }
    }
    return res;
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

Matrix Matrix::init_rand_mat(int rows_, int cols_, std::mt19937& rand, float min_val, float max_val){
    Matrix matrix(rows_, cols_);

    std::uniform_real_distribution<float> dist(min_val, max_val);

    for(std::size_t i = 0; i < matrix.data.size(); i++)
        matrix.data[i] = dist(rand);

    return matrix;
}

Matrix Matrix::load_image_mat(const Dataset& dataset, int start_idx, int batch_size){
    int image_size = dataset.height * dataset.width;

    Matrix x(image_size, batch_size); 

    for(int b = 0; b < batch_size; b++){
        int curr_image_idx = start_idx + b;

        for(int pixel = 0; pixel < image_size; pixel++){
            x.data[pixel * x.cols + b] = dataset.images[image_size * curr_image_idx + pixel];
        }
    }
    return x;
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

void Matrix::scale(float scale){
    for(float& val : data)
        val *= scale;
}

void Matrix::subtract_scaled(const Matrix& mat, float scale){
    for(std::size_t i = 0; i < data.size(); i++)
        data[i] -= scale * mat.data[i];
}
