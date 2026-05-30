#include <stdexcept>
#include <random>
#include <cstddef>
#include <omp.h>

#include "matrix.hpp"
#include "data_loader.hpp"

Matrix::Matrix(int rows_, int cols_) : data(rows_ * cols_), rows(rows_), cols(cols_) {}

void Matrix::multiply_into(const Matrix& other, Matrix& out) const{
    if(cols != other.rows)
        throw std::runtime_error("multiply_into: Invalid matrix dimensions (1)");
    if(out.rows != rows || out.cols != other.cols)
        throw std::runtime_error("multiply_into: Invalid matrix dimensions (2)");
    
    const int batch_size = other.cols;
    const float * __restrict x = other.data.data();
    float * __restrict o = out.data.data();

    #pragma omp for  
    for(int i = 0; i < rows; i++){
        float * __restrict out_row = o + i * batch_size;
        
        /* equivalent of the serial zero. Just zero curr batch once loaded in */
        for(int b = 0; b < batch_size; b++)
            out_row[b] = 0.0f;
       
        for(int u = 0; u < cols; u++){
            const float w = data[i * cols + u];
            const float * __restrict x_row = x + u * batch_size;
            for(int b = 0; b < batch_size; b++)
                out_row[b] += w * x_row[b];
        }
    }
}

void Matrix::transpose_multiply_into(const Matrix& other, Matrix& out) const{
    if(out.rows != cols || out.cols != other.cols || other.rows != rows)
        throw std::runtime_error("transpose_multiply_into: Invalid matrix dimensions");
    
    #pragma omp for
    for(int i = 0; i < cols; i++)
    for(int c = 0; c < other.cols; c++){
        float curr_sum = 0.0f;
        for(int u = 0; u < rows; u++)
            curr_sum += data[u * cols + i] * other.data[u * other.cols + c];
        out.data[i * out.cols + c] = curr_sum;
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

    #pragma omp for
    for(std::size_t i = 0; i < data.size(); i++)
        out.data[i] = data[i] * other.data[i];
}

void Matrix::subtract_outer_product(const Matrix& col, const Matrix& row, float scale){
    if(col.rows != rows || row.rows != cols || row.cols != col.cols) // lol weird
        throw std::runtime_error("subtract_outer_product: Invalid matrix dimensions");
    const int batch_size = col.cols;
    const float * __restrict col_data = col.data.data();
    const float * __restrict row_data = row.data.data();
    float * __restrict out_data = data.data();

    #pragma omp for
    for(int i = 0; i < rows; i++){
        const float * __restrict curr_col = col_data + i * batch_size;
        for(int u = 0; u < cols; u++){
            const float * __restrict curr_row = row_data + u * batch_size;
            float curr_sum = 0.0f;
            for(int b = 0; b < batch_size; b++)
                curr_sum += curr_col[b] * curr_row[b];
            out_data[i * cols + u] -= scale * curr_sum;
        }
    }
}

void Matrix::add(const Matrix& other){
    if(rows != other.rows)
        throw std::runtime_error("add: Invalid dimensions for matrix addition (1)");
    
    if(cols == other.cols){ // normal matrix addition
        #pragma omp for
        for(int iu = 0; iu < cols * rows; iu++)
            data[iu] += other.data[iu];
    }else if(other.cols == 1){ // broadcast addition over all columns
        #pragma omp for
        for(int i = 0; i < rows; i++){
            const float b_i = other.data[i];
            for(int u = 0; u < cols; u++)
                data[i * cols + u] += b_i;
        }
    }else
        throw std::runtime_error("add: Invalid dimensions for matrix addition (2)");
}

Matrix Matrix::init_he(int rows_, int cols_, std::mt19937& rand){
    Matrix matrix(rows_, cols_);
    float stddev = std::sqrt(2.0f / static_cast<float>(cols_));
    std::normal_distribution<float> dist(0.0f, stddev);
    for(std::size_t i = 0; i < matrix.data.size(); i++)
        matrix.data[i] = dist(rand);
    return matrix;
}

int Matrix::argmax(int col) const{
    int best_idx = 0;
    float best_val = data[col];

    for(int i = 1; i < rows; i++){
        float val = data[i * cols + col];
        if(val > best_val) { best_val = val; best_idx = i; }
    }
    return best_idx;
}

void Matrix::subtract_one_hot(const std::vector<uint8_t>& labels, int start){
    #pragma omp for
    for(int i = 0; i < cols; i++){
        int label = labels[start + i];
        data[label * cols + i] -= 1.0f;
    }
}

void Matrix::subtract_scaled(const Matrix& mat, float scale){
    const int batch_size = mat.cols;
    #pragma omp for
    for(int i = 0; i < rows; i++){
        float curr_sum = 0.0f;
        for(int u = 0; u < batch_size; u++)
            curr_sum += mat.data[i * batch_size + u];
        data[i] -= scale * curr_sum;
    }
}
