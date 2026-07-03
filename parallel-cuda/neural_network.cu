#include <iostream>
#include <string>

#include "data_loader.hpp"
#include "matrix.hpp"
#include "../shared.hpp"

void train(const DeviceDataset&, Matrix&, Matrix&, Matrix&, Matrix&, int, float, int);
void evaluate(const DeviceDataset&, const Matrix&, const Matrix&, const Matrix&, const Matrix&, int);

int main(int argc, char *argv[]){
    std::cout << "Neural Network\n";

    if(argc < 7){
        std::cerr << "Usage: " << argv[0]
            << " <digits/letters/byclass> <epochs> <learning rate> <hidden size> <training_batch_size> <eval_batch_size>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int epochs = std::stoi(argv[2]);
    float learning_rate = std::stof(argv[3]);
    int hidden_size = std::stoi(argv[4]);
    int training_batch_size = std::stoi(argv[5]);
    int eval_batch_size = std::stoi(argv[6]);

    DatasetPaths paths = Shared::dataset(dataset_name);
    int num_labels = paths.num_classes;

    Dataset dataset = load_images(paths.train_images, num_labels);
    load_labels(paths.train_labels, &dataset);

    DeviceDataset device_dataset(dataset);

    std::mt19937 rand(Shared::SEED);

    int input_size = dataset.height * dataset.width;

    Matrix W1 = Matrix::init_he(hidden_size, input_size, rand);
    Matrix b1(hidden_size, 1);
    Matrix W2 = Matrix::init_he(num_labels, hidden_size, rand);
    Matrix b2(num_labels, 1);

    std::cout << "************\tTRAINING\t*************" << "\n";
    train(device_dataset, W1, b1, W2, b2, epochs, learning_rate, training_batch_size);

    Dataset test_set = load_images(paths.test_images, num_labels);
    load_labels(paths.test_labels, &test_set);
    DeviceDataset device_test_set(test_set);

    std::cout << "************\tEVALUATION\t*************\n";
    evaluate(device_test_set, W1, b1, W2, b2, eval_batch_size);

    return 0;
}

void train(const DeviceDataset& dataset, Matrix& W1, Matrix& b1, Matrix& W2, Matrix& b2, int epochs, float learning_rate, int batch_size){
    int hidden_size = W1.rows;
    int num_labels = W2.rows;
    constexpr float MOMENTUM = 0.9f;
    const float gradient_scale = learning_rate / static_cast<float>(batch_size);

    Matrix A1(hidden_size, batch_size);
    Matrix Z2(num_labels, batch_size);
    Matrix predictions(num_labels, batch_size);
    Matrix dA1(hidden_size, batch_size);

    Matrix vW1(hidden_size, W1.cols);
    Matrix vW2(num_labels, hidden_size);
    Matrix vb1(hidden_size, 1);
    Matrix vb2(num_labels, 1);

    Metrics metrics;

    /* dropping the last batch if it's smaller, makes most sense with the current architecture */
    const int num_batches = dataset.num_samples / batch_size; // want int division

    auto run_epoch = [&](){
        for(int batch = 0; batch < num_batches; batch++){
            int start = batch * batch_size;
            Matrix X = Matrix::batch_view(dataset, start, batch_size);

            /*      Forward     */
            W1.multiply_into(X, A1);
            A1.bias_relu(b1);

            W2.multiply_into(A1, Z2);
            Z2.softmax_bias_into(b2, predictions);

            /* intermediate metrics */
            predictions.accumulate_metrics(dataset, start, metrics);

            /*      Backpropagation     */
            W2.transpose_multiply_into(predictions, dA1);
            dA1.relu_backward(A1);

            vW2.accumulate_outer_product(predictions, A1, MOMENTUM);
            vW1.accumulate_outer_product(dA1, X, MOMENTUM);
            W2.subtract_velocity(vW2, gradient_scale);
            W1.subtract_velocity(vW1, gradient_scale);
            b2.subtract_scaled(predictions, vb2, MOMENTUM, gradient_scale);
            b1.subtract_scaled(dA1, vb1, MOMENTUM, gradient_scale);
        }
    };

    cudaGraphExec_t epoch_graph = nullptr;

    for(int epoch = 0; epoch < epochs; epoch++){
        metrics.reset();

        if(epoch == 0){
            /* first epoch runs eagerly, warms cublas up so the capture below is clean */
            run_epoch();
        }else{
            if(!epoch_graph){
                /* record the whole epoch once, every launch after this is a single replay */
                cudaGraph_t graph;
                CUDA_CHECK(cudaStreamBeginCapture(gpu_stream(), cudaStreamCaptureModeGlobal));
                run_epoch();
                CUDA_CHECK(cudaStreamEndCapture(gpu_stream(), &graph));
                CUDA_CHECK(cudaGraphInstantiate(&epoch_graph, graph, 0));
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
            CUDA_CHECK(cudaGraphLaunch(epoch_graph, gpu_stream()));
        }

        float tot_loss = 0.0f;
        int correct = 0;
        metrics.read(tot_loss, correct);

        const float processed = static_cast<float>(num_batches * batch_size);
        float avg_loss = tot_loss / processed;
        float accuracy = static_cast<float>(correct) / processed;
        std::cout << "Epoch: " << epoch + 1 << " | loss: " << avg_loss << " | accuracy: " << accuracy << "\n";
    }

    if(epoch_graph) CUDA_CHECK(cudaGraphExecDestroy(epoch_graph));
}

void evaluate(const DeviceDataset& dataset, const Matrix& W1, const Matrix& b1, const Matrix& W2, const Matrix& b2, int batch_size){
    int hidden_size = W1.rows;
    int num_labels = W2.rows;

    Matrix A1(hidden_size, batch_size);
    Matrix Z2(num_labels, batch_size);
    Matrix predictions(num_labels, batch_size);

    Metrics metrics;

    // assuming divide evenly
    const int num_batches = dataset.num_samples / batch_size;

    for(int batch = 0; batch < num_batches; batch++){
        int start = batch * batch_size;
        Matrix X = Matrix::batch_view(dataset, start, batch_size);

        W1.multiply_into(X, A1);
        A1.bias_relu(b1);

        W2.multiply_into(A1, Z2);
        Z2.softmax_bias_into(b2, predictions);

        predictions.accumulate_metrics(dataset, start, metrics);
    }
    float tot_loss = 0.0f;
    int correct = 0;
    metrics.read(tot_loss, correct);

    // safegaurd incase the assumed isn't true
    const float processed = static_cast<float>(num_batches * batch_size);
    float avg_loss = tot_loss / processed;
    float accuracy = static_cast<float>(correct) / processed;
    std::cout << "test loss: " << avg_loss << " | test accuracy: " << accuracy << "\n";
}
