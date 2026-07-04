#include <iostream>
#include <string>
#include <algorithm>

#include "data_loader.hpp"
#include "matrix.hpp"
#include "../shared.hpp"

void solve(ImageStream&, const std::vector<uint8_t>&, const Matrix&, const Matrix&, Matrix&, float);
void evaluate(const DeviceDataset&, const Matrix&, const Matrix&, const Matrix&);

/* big enough to keep the gemms fat, small enough that H fits comfortably */
static constexpr int CHUNK = 16384;

int main(int argc, char *argv[]){
    std::cout << "Neural Network\n";

    if(argc < 4){
        std::cerr << "Usage: " << argv[0]
            << " <digits/letters/byclass> <hidden size> <ridge lambda>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int hidden_size = std::stoi(argv[2]);
    float lambda = std::stof(argv[3]);

    DatasetPaths paths = Shared::dataset(dataset_name);
    int num_labels = paths.num_classes;

    ImageStream stream(paths.train_images);
    std::vector<uint8_t> labels = load_label_file(paths.train_labels, stream.num_samples);

    std::mt19937 rand(Shared::SEED);

    int input_size = stream.height * stream.width;

    /* the hidden layer is never trained, random projection + relu is the whole feature map */
    Matrix W1 = Matrix::init_he(hidden_size, input_size, rand);
    Matrix b1 = Matrix::init_uniform(hidden_size, 1, 0.5f, rand);
    Matrix W2(num_labels, hidden_size);

    std::cout << "************\tTRAINING\t*************" << "\n";
    solve(stream, labels, W1, b1, W2, lambda);

    Dataset test_set = load_images(paths.test_images, num_labels);
    load_labels(paths.test_labels, &test_set);
    DeviceDataset device_test_set(test_set);

    std::cout << "************\tEVALUATION\t*************\n";
    evaluate(device_test_set, W1, b1, W2);

    return 0;
}

void solve(ImageStream& stream, const std::vector<uint8_t>& labels, const Matrix& W1, const Matrix& b1, Matrix& W2, float lambda){
    int hidden_size = W1.rows;
    int num_labels = W2.rows;
    int image_size = W1.cols;
    int num_samples = static_cast<int>(stream.num_samples);

    Matrix A(hidden_size, hidden_size);
    Matrix H(hidden_size, CHUNK);
    Matrix Y(num_labels, CHUNK);

    uint8_t *device_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&device_labels, labels.size()));
    CUDA_CHECK(cudaMemcpyAsync(device_labels, labels.data(), labels.size(), cudaMemcpyHostToDevice, gpu_stream()));

    StagedChunk staging[2] = {{image_size, CHUNK}, {image_size, CHUNK}};

    /* one pass over the data builds the normal equations, no epochs */
    for(int start = 0; start < num_samples; start += CHUNK){
        int width = std::min(CHUNK, num_samples - start);
        StagedChunk& buf = staging[(start / CHUNK) % 2];

        buf.wait();
        stream.read_slab(buf.host, width);
        Matrix X = buf.upload(width);

        Matrix Hv = H.first_cols(width);
        Matrix Yv = Y.first_cols(width);

        W1.multiply_into(X, Hv);
        Hv.bias_relu(b1);

        A.gram_accumulate(Hv);
        Yv.one_hot(device_labels, start);
        W2.target_accumulate(Hv, Yv);

        buf.record();
    }

    /* lambda scales with the sample count so the same value works on every set */
    A.add_diagonal(lambda * static_cast<float>(num_samples));
    A.cholesky_solve(W2);

    CUDA_CHECK(cudaFree(device_labels));
}

void evaluate(const DeviceDataset& dataset, const Matrix& W1, const Matrix& b1, const Matrix& W2){
    int hidden_size = W1.rows;
    int num_labels = W2.rows;

    Matrix H(hidden_size, CHUNK);
    Matrix Z(num_labels, CHUNK);

    Metrics metrics;

    for(int start = 0; start < dataset.num_samples; start += CHUNK){
        int width = std::min(CHUNK, dataset.num_samples - start);
        Matrix X = Matrix::batch_view(dataset, start, width);
        Matrix Hv = H.first_cols(width);
        Matrix Zv = Z.first_cols(width);

        W1.multiply_into(X, Hv);
        Hv.bias_relu(b1);
        W2.multiply_into(Hv, Zv);

        Zv.accumulate_metrics(dataset, start, metrics);
    }

    float tot_loss = 0.0f;
    int correct = 0;
    metrics.read(tot_loss, correct);

    const float processed = static_cast<float>(dataset.num_samples);
    float avg_loss = tot_loss / processed;
    float accuracy = static_cast<float>(correct) / processed;
    std::cout << "test loss: " << avg_loss << " | test accuracy: " << accuracy << "\n";
}
