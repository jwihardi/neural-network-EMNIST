#pragma once

#include <algorithm>
#include <cmath>

#include "matrix.hpp"

struct Activations{
    inline static void softmax_into(const Matrix& in, Matrix& out){
        #pragma omp for
        for(int i = 0; i < in.cols; i++){
            float max_val = in.data[i];
            for(int u = 0; u < in.rows; u++){
                float val = in.data[u * in.cols + i];
                if(val > max_val) max_val = val;
            }
            float sum = 0.0f;
            for(int u = 0; u < in.rows; u++){
                float exponent = std::exp(in.data[u * in.cols + i] - max_val);
                out.data[u * in.cols + i] = exponent;
                sum += exponent;
            }
            for(int u = 0; u < in.rows; u++)
                out.data[u * in.cols + i] /= sum;
        }
    }

    inline static void relu_into(const Matrix& in, Matrix& out){
        #pragma omp for
        for(std::size_t i = 0; i < in.data.size(); i++)
            out.data[i] = std::max(0.0f, in.data[i]);
    }

    inline static void relu_derivative_into(const Matrix& in, Matrix& out){
        #pragma omp for
        for(std::size_t i = 0; i < in.data.size(); i++)
            out.data[i] = (in.data[i] > 0.0f) ? 1.0f : 0.0f;
    }
};
