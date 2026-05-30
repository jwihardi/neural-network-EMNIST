#pragma once

#include <string>
#include <stdexcept>

struct DatasetPaths {
    std::string train_images, train_labels, test_images, test_labels;
    int num_classes;

};
/*
 *
 *
 *󰡯 emnist-byclass-test-images-idx3-ubyte   󰡯 emnist-digits-test-images-idx3-ubyte   󰡯 emnist-letters-test-images-idx3-ubyte
󰡯 emnist-byclass-test-labels-idx1-ubyte   󰡯 emnist-digits-test-labels-idx1-ubyte   󰡯 emnist-letters-test-labels-idx1-ubyte
󰡯 emnist-byclass-train-images-idx3-ubyte  󰡯 emnist-digits-train-images-idx3-ubyte  󰡯 emnist-letters-train-images-idx3-ubyte
󰡯 emnist-byclass-train-labels-idx1-ubyte  󰡯 emnist-digits-train-labels-idx1-ubyte  󰡯 emnist-letters-train-labels-idx1-ubyte
 *
 *
 *
 *
 *
 * */

struct Shared {
    static DatasetPaths dataset(const std::string& name){
        if(name == "digits"){
            return {
                "emnist/emnist-digits-train-images-idx3-ubyte",
                "emnist/emnist-digits-train-labels-idx1-ubyte",
                "emnist/emnist-digits-test-images-idx3-ubyte",
                "emnist/emnist-digits-test-labels-idx1-ubyte",
                10
            };
        }
        if(name == "letters"){
            return {
                "emnist/emnist-letters-train-images-idx3-ubyte",
                "emnist/emnist-letters-train-labels-idx1-ubyte",
                "emnist/emnist-letters-test-images-idx3-ubyte",
                "emnist/emnist-letters-test-labels-idx1-ubyte",
                27
            };
        }
        if(name == "byclass"){
            return {
                "emnist-byclass-train-images-idx3-ubyte",
                "emnist-byclass-train-labels-idx1-ubyte",
                "emnist-byclass-test-images-idx3-ubyte",
                "emnist-byclass-test-labels-idx1-ubyte",
                62
            };
        }
        throw std::runtime_error("Shared: Invalid dataset name");
    }
};
