#include <stdexcept>
#include <random>
#include <cstddef>
#include "matrix.hpp"
#include "data_loader.hpp"

Matrix::Matrix(int rows_, int cols_) : data(rows_ * cols_), rows(rows_), cols(cols_) {}

void Matrix::load_image_into(const Dataset& dataset, int start_idx, int batch_size, Matrix& out){
    int image_size = dataset.height * dataset.width;

    for(int b = 0; b < batch_size; b++){
        int curr_image_idx = start_idx + b;
        for(int pixel = 0; pixel < image_size; pixel++)
            out.data[pixel * out.cols + b] = dataset.images[image_size * curr_image_idx + pixel];
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

void Matrix::hadamard_into(const Matrix& other, Matrix& out) const{
    if(cols != other.cols || rows != other.rows)
        throw std::runtime_error("hadamard_into: Invalid matrix dimensions (1)");
    if(out.rows != rows || out.cols != cols)
        throw std::runtime_error("hadamard_into: Invalid matrix dimensions (2)");

    for(std::size_t i = 0; i < data.size(); i++)
        out.data[i] = data[i] * other.data[i];
}

void Matrix::add(const Matrix& other){
    if(rows != other.rows)
        throw std::runtime_error("add: Invalid dimensions for matrix addition (1)");
    
    if(cols == other.cols){ // normal matrix addition
        for(int i = 0; i < rows; i++)
        for(int u = 0; u < cols; u++)
            data[i * cols + u] += other.data[i * cols + u];
    }else if(other.cols == 1){ // broadcast addition over all columns
        for(int i = 0; i < rows; i++){
            const float b_i = other.data[i];
            for(int u = 0; u < cols; u++)
                data[i * cols + u] += b_i;
        }
    }else
        throw std::runtime_error("add: Invalid dimensions for matrix addition (2)");
}

void Matrix::subtract_scaled(const Matrix& mat, float scale){
    const int batch_size = mat.cols;

    for(int i = 0; i < rows; i++){
        float curr_sum = 0.0f;
        for(int u = 0; u < batch_size; u++)
            curr_sum += mat.data[i * batch_size + u];
        data[i] -= scale * curr_sum;
    }
}

void Matrix::subtract_one_hot(const std::vector<uint8_t>& labels, int start){
    for(int i = 0; i < cols; i++){
        int label = labels[start + i];
        data[label * cols + i] -= 1.0f;
    }
}

int Matrix::argmax(int col) const{
    int best_idx = 0;
    float best_val = data[col];

    for(int i = 1; i < rows; i++){
        float val = data[i * cols + col];
        if(val > best_val){
            best_val = val; 
            best_idx = i;
        }
    }
    return best_idx;
}
