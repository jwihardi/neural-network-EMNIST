#pragma once 

#include <iostream>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int IMAGE_MAGIC = 2051;
static constexpr int LABEL_MAGIC = 2049;

struct Dataset{
    std::vector<float> images;
    std::vector<uint8_t> labels;

    unsigned int num_samples, height, width, num_classes;

    Dataset(unsigned int height_, unsigned int width_, unsigned int num_classes_, unsigned int num_samples_) :
        images(height_ * width_ * num_samples_),
        labels(num_samples_),
        num_samples(num_samples_), 
        height(height_),
        width(width_),
        num_classes(num_classes_) 
    {}
};

inline uint32_t read_u32_line(std::ifstream& file){
    uint8_t bytes[4]; 

    file.read(reinterpret_cast<char *>(bytes), 4);

    if(!file) 
        throw std::runtime_error("Failed to read 4 bytes\n");

    /* combined all 4 bytes into one using big endian order*/
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) | (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

inline Dataset load_images(const std::string& image_file_path, int num_classes){
    std::ifstream file(image_file_path, std::ios::binary);
    if(!file) 
        throw std::runtime_error("Failed to open image file: " + image_file_path);

    uint32_t magic = read_u32_line(file);
    
    /* IMAGE_MAGIC determines if the IDX file contains files */
    if(magic != IMAGE_MAGIC) throw std::runtime_error("Invalid MNIST/EMNIST image file");
    
    int num_images = static_cast<int>(read_u32_line(file));
    int height = static_cast<int>(read_u32_line(file));
    int width = static_cast<int>(read_u32_line(file));
    int image_size = height * width;

    Dataset dataset(height, width, num_classes, num_images);

    for(unsigned int image = 0; image < dataset.num_samples; image++){
        for(int pixel = 0; pixel < image_size; pixel++){
            uint8_t pixel_val;
            file.read((char*)(&pixel_val), 1);
            if(!file) 
                throw std::runtime_error("Failed to read image pixel (image: " + 
                        std::to_string(image) + ", pixel: " + std::to_string(pixel) + ")");

            int idx = image * image_size + pixel;
            dataset.images[idx] = pixel_val / 255.0f; // 255.0f to normalize
        }
    }
    return dataset;
}

inline void load_labels(const std::string& label_file_path, Dataset* dataset){
    std::ifstream file(label_file_path, std::ios::binary);
    if(!file) 
        throw std::runtime_error("Failed to open label file: " + label_file_path);

    uint32_t magic = read_u32_line(file);
    if(magic != LABEL_MAGIC) 
        throw std::runtime_error("Invalid MNIST/EMNIST label file");

    uint32_t num_labels = read_u32_line(file);
    
    if(num_labels != dataset->num_samples) 
        throw std::runtime_error("Number of labels don't match number of images");

    for(uint32_t i = 0; i < num_labels; i++){
        uint8_t label;
        file.read(reinterpret_cast<char *>(&label), 1);
        if(!file) 
            throw std::runtime_error("Failed to read label at index: " + std::to_string(i));
        dataset->labels[i] = static_cast<int>(label);
    }
}

