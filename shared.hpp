#pragma once

#include <string>
#include <stdexcept>

struct DatasetPaths {
    std::string train_images, train_labels, test_images, test_labels;
    int num_classes;

};

struct Shared {
    static DatasetPaths dataset(const std::string& name){
        if(name == "mnist"){
            return {
                "mnist/train-images-idx3-ubyte",
                "mnist/train-labels-idx1-ubyte",
                "mnist/t10k-images-idx3-ubyte",
                "mnist/t10k-labels-idx1-ubyte",
                10
            };
        }
        if(name == "emnist"){
            return {
                "emnist/emnist-byclass-train-images-idx3-ubyte",
                "emnist/emnist-byclass-train-labels-idx1-ubyte",
                "emnist/emnist-byclass-test-images-idx3-ubyte",
                "emnist/emnist-byclass-test-labels-idx1-ubyte",
                62
            };
        }
        throw std::runtime_error("Shared: Invalid dataset name");
    }
};
