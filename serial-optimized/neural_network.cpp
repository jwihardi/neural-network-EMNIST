#include <iostream>
#include <string>

#include "data_loader.hpp"
#include "matrix.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include "../shared.hpp"

void train(const Dataset&, Matrix&, Matrix&, Matrix&, Matrix&, int, float);
void evaluate(const Dataset&, const Matrix&, const Matrix&, const Matrix&, const Matrix&);

int main(int argc, char *argv[]){
    std::cout << "Neural Network\n"; 

    if(argc < 4){
        std::cerr << "Usage: " << argv[0] 
            << " <epochs> <learning rate> <hidden size>\n";
        return 1;
    }

    int epochs = std::stoi(argv[1]);
    float learning_rate = std::stof(argv[2]);
    int hidden_size = std::stoi(argv[3]);

    int num_labels = 10;
    Dataset dataset = load_images(Shared::IMAGE_FILE_PATH, num_labels);
    load_labels(Shared::LABEL_FILE_PATH, &dataset);

    std::mt19937 rand(0);
    
    int input_size = dataset.height * dataset.width;

    Matrix W1 = Matrix::init_he(hidden_size, input_size, rand);
    Matrix b1(hidden_size, 1);
    Matrix W2 = Matrix::init_he(num_labels, hidden_size, rand);
    Matrix b2(num_labels, 1);

    std::cout << "************\tTRAINING\t*************" << "\n";
    train(dataset, W1, b1, W2, b2, epochs, learning_rate);
   
    Dataset test_set = load_images(Shared::TEST_IMAGE_FILE_PATH, num_labels);
    load_labels(Shared::TEST_LABEL_FILE_PATH, &test_set);

    std::cout << "************\tEVALUATION\t*************\n";
    evaluate(test_set, W1, b1, W2, b2);

    return 0;
}

void train(const Dataset& dataset, Matrix& W1, Matrix& b1, Matrix& W2, Matrix& b2, int epochs, float learning_rate){
    int input_size = dataset.height * dataset.width;
    int hidden_size = W1.rows;
    int num_labels = W2.rows;

    Matrix x(input_size, 1);
    Matrix z1(hidden_size, 1);
    Matrix a1(hidden_size, 1);
    Matrix z2(num_labels, 1);
    Matrix predictions(num_labels, 1);
    Matrix da1(hidden_size, 1);
    Matrix relu_deriv(hidden_size, 1);
    Matrix dz1(hidden_size, 1);

    for(int epoch = 0; epoch < epochs; epoch++){
        float tot_loss = 0.0f;
        int correct = 0;

        for(unsigned int sample = 0; sample < dataset.num_samples; sample++){
            Matrix::load_image_into(dataset, sample, 1, x);
            
            /*      Forward     */
            W1.multiply_into(x, z1);
            z1.add(b1);
            Activations::relu_into(z1, a1);

            W2.multiply_into(a1, z2);
            z2.add(b2);

            Activations::softmax_into(z2, predictions);

            int actual = dataset.labels[sample];
            if(predictions.argmax() == actual) correct++;
            tot_loss += Losses::cross_entropy(predictions, actual);

            /*      Backpropagation     */
            predictions.subtract_one_hot(actual);

            W2.transpose_multiply_into(predictions, da1);
            Activations::relu_derivative_into(z1, relu_deriv);
            da1.hadamard_into(relu_deriv, dz1);

            W2.subtract_outer_product(predictions, a1, learning_rate);
            W1.subtract_outer_product(dz1, x, learning_rate);
            b2.subtract_scaled(predictions, learning_rate);
            b1.subtract_scaled(dz1, learning_rate);
        }
        float avg_loss = tot_loss / static_cast<float>(dataset.num_samples);
        float accuracy = static_cast<float>(correct) / static_cast<float>(dataset.num_samples);
        std::cout << "Epoch: " << epoch + 1 << " | loss: " << avg_loss << " | accuracy: " << accuracy << "\n";
    }
}

void evaluate(const Dataset& dataset, const Matrix& W1, const Matrix& b1, const Matrix& W2, const Matrix& b2){
    int input_size = dataset.height * dataset.width;
    int hidden_size = W1.rows;
    int num_labels = W2.rows;

    Matrix x(input_size, 1);
    Matrix z1(hidden_size, 1);
    Matrix a1(hidden_size, 1);
    Matrix z2(num_labels, 1);
    Matrix predictions(num_labels, 1);

    float tot_loss = 0.0f;
    int correct = 0;

    for(unsigned int sample = 0; sample < dataset.num_samples; sample++){
        Matrix::load_image_into(dataset, sample, 1, x);

        W1.multiply_into(x, z1);
        z1.add(b1);
        Activations::relu_into(z1, a1);

        W2.multiply_into(a1, z2);
        z2.add(b2);

        Activations::softmax_into(z2, predictions);

        int actual = dataset.labels[sample];
        if(predictions.argmax() == actual) ++correct;
        tot_loss += Losses::cross_entropy(predictions, actual);
    }
    float avg_loss = tot_loss / static_cast<float>(dataset.num_samples);
    float accuracy = static_cast<float>(correct) / static_cast<float>(dataset.num_samples);
    std::cout << "test loss: " << avg_loss << " | test accuracy: " << accuracy << "\n";
}
