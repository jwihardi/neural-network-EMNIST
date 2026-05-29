#pragma once

#include <string>

struct Shared {
    inline static const std::string IMAGE_FILE_PATH = "mnist/train-images-idx3-ubyte";
    inline static const std::string LABEL_FILE_PATH = "mnist/train-labels-idx1-ubyte";
    inline static const std::string TEST_IMAGE_FILE_PATH = "mnist/t10k-images-idx3-ubyte";
    inline static const std::string TEST_LABEL_FILE_PATH = "mnist/t10k-labels-idx1-ubyte";
};
