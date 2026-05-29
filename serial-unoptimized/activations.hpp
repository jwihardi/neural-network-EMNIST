#pragma once

#include "matrix.hpp"

struct Activations{
    /* softmax = e^{current score} / sum of all e^{scores} */
    inline static Matrix softmax(const Matrix& mat){
        Matrix res(mat.rows, mat.cols);
        float max_val = mat.data[0];    
        
        for(int i = 0; i < mat.rows; i++){
            if(mat.data[i] > max_val)
                max_val = mat.data[i];
        }

        float sum = 0.0f;

        for(int i = 0; i < mat.rows; i++){
            res.data[i] = std::exp(mat.data[i] - max_val); // prevent e from exploding
            sum += res.data[i];
        }
        
        for(int i = 0; i < mat.rows; i++){
            res.data[i] /= sum;
        }
        return res;
    }

    inline static Matrix relu(const Matrix& mat){
        Matrix res(mat.rows, mat.cols);
        
        for(std::size_t i = 0; i < mat.data.size(); i++)
            res.data[i] = std::max(0.0f, mat.data[i]);
        
        return res;
    }

    inline static Matrix relu_derivative(const Matrix& mat){
        Matrix res(mat.rows, mat.cols);

        for(std::size_t i = 0; i < mat.data.size(); i++)
            res.data[i] = (mat.data[i] > 0.0f) ? 1.0f : 0.0f;
            
        return res;
    }

};
