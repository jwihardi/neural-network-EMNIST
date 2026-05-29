#pragma once

#include "matrix.hpp"

struct Activations{
    inline static void softmax_into(const Matrix& in, Matrix& out){
        float max_val = in.data[0];

        for(int i = 1; i < in.rows; i++)
            if(in.data[i] > max_val) max_val = in.data[i];

        float sum = 0.0f;
        for(int i = 0; i < in.rows; i++){
            out.data[i] = std::exp(in.data[i] - max_val);
            sum += out.data[i];
        }

        for(int i = 0; i < in.rows; i++)
            out.data[i] /= sum;
    }

    inline static void relu_into(const Matrix& in, Matrix& out){
        for(std::size_t i = 0; i < in.data.size(); i++)
            out.data[i] = std::max(0.0f, in.data[i]);
    }

    inline static void relu_derivative_into(const Matrix& in, Matrix& out){
        for(std::size_t i = 0; i < in.data.size(); i++)
            out.data[i] = (in.data[i] > 0.0f) ? 1.0f : 0.0f;
    }
};
