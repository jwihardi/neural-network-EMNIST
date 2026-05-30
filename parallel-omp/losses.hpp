#pragma once

struct Losses{
    static constexpr float EPSILON = 1e-7f; // the standard, using MIN_FLT would be too small

    static inline float cross_entropy(const Matrix& predictions, int col, int label){
        float p = predictions.data[label * predictions.cols + col];
        p = std::max(p, EPSILON);
        return -std::log(p);
    }

};

