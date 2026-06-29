#pragma once

#include "matrix.hpp"

struct Activations{
    inline static void relu_into(const Matrix& in, Matrix& out){
        in.relu_into(out);
    }

    inline static void relu_derivative_into(const Matrix& in, Matrix& out){
        in.relu_derivative_into(out);
    }

    inline static void softmax_into(const Matrix& in, Matrix& out){
        in.softmax_into(out);
    }
};
