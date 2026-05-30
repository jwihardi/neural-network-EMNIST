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

    if(argc < 5){
        std::cerr << "Usage: " << argv[0] 
            << " <digits/letters/byclass> <epochs> <learning rate> <hidden size>\n";
        return 1;
    }
    std::string dataset_name = argv[1];
    int epochs = std::stoi(argv[2]);
    float learning_rate = std::stof(argv[3]);
    int hidden_size = std::stoi(argv[4]);
    
    DatasetPaths paths = Shared::dataset(dataset_name);
    int num_labels = paths.num_classes;

    Dataset dataset = load_images(paths.train_images, num_labels);
    load_labels(paths.train_labels, &dataset);

    std::mt19937 rand(0);
    
    int input_size = dataset.height * dataset.width;

    Matrix W1 = Matrix::init_he(hidden_size, input_size, rand);
    Matrix b1(hidden_size, 1);
    Matrix W2 = Matrix::init_he(num_labels, hidden_size, rand);
    Matrix b2(num_labels, 1);

    std::cout << "************\tTRAINING\t*************" << "\n";
    train(dataset, W1, b1, W2, b2, epochs, learning_rate);
   
    Dataset test_set = load_images(paths.test_images, num_labels);
    load_labels(paths.test_labels, &test_set);

    std::cout << "************\tEVALUATION\t*************\n";
    evaluate(test_set, W1, b1, W2, b2);

    return 0;
}

void train(const Dataset& dataset, Matrix& W1, Matrix& b1, Matrix& W2, Matrix& b2, int epochs, float learning_rate){
    for(int epoch = 0; epoch < epochs; epoch++){
        float tot_loss = 0.0f;
        int correct = 0;

        for(unsigned int sample = 0; sample < dataset.num_samples; sample++){
            Matrix x = Matrix::load_image_mat(dataset, sample, 1); // batch size 1 for now
            
            /*      Forward     */
            Matrix z1 = W1.multiply(x);
            z1.add(b1);
            Matrix a1 = Activations::relu(z1);

            Matrix z2 = W2.multiply(a1);
            z2.add(b2);

            Matrix predictions = Activations::softmax(z2);

            int actual = dataset.labels[sample];
            int prediction = predictions.argmax();

            if(prediction == actual) correct++;
            tot_loss += Losses::cross_entropy(predictions, actual);

            /*      Backpropagation     */
            predictions.subtract_one_hot(actual);

            Matrix da1 = W2.transpose_multiply(predictions);
            Matrix dz1 = da1.hadamard(Activations::relu_derivative(z1));

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
    float tot_loss = 0.0f;
    int correct = 0;

    for(unsigned int sample = 0; sample < dataset.num_samples; sample++){
        Matrix x = Matrix::load_image_mat(dataset, sample, 1);

        Matrix z1 = W1.multiply(x);
        z1.add(b1);
        Matrix a1 = Activations::relu(z1);

        Matrix z2 = W2.multiply(a1);
        z2.add(b2);

        Matrix predictions = Activations::softmax(z2);

        int actual = dataset.labels[sample];
        if(predictions.argmax() == actual) ++correct;
        tot_loss += Losses::cross_entropy(predictions, actual);
    }
    float avg_loss = tot_loss / static_cast<float>(dataset.num_samples);
    float accuracy = static_cast<float>(correct) / static_cast<float>(dataset.num_samples);
    std::cout << "test loss: " << avg_loss << " | test accuracy: " << accuracy << "\n";
}
