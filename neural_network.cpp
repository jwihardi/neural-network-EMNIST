#include <iostream>
#include <string>
#include "data_loader.hpp"
#include "matrix.hpp"
#include "activations.hpp"
#include "losses.hpp"

void train(const Dataset&, Matrix&, Matrix&, Matrix&, Matrix&, int, float);

int main(int argc, char *argv[]){
    std::cout << "Neural Network\n"; 

    if(argc < 6){
        std::cerr << "Usage: " << argv[0] 
            << " <images_file> <labels_file> <epochs> <learning_rate> <hidden_layer>\n";
        return 1;
    }

    std::string image_file_path = argv[1];
    std::string label_file_path = argv[2];
    int epochs = std::stoi(argv[3]);
    float learning_rate = std::stof(argv[4]);
    int hidden_size = std::stoi(argv[5]);

    int num_labels = 10;
    Dataset dataset = load_images(image_file_path, num_labels);
    load_labels(label_file_path, &dataset);

    int batch_size = 1;

    std::mt19937 rand(0);
    
    int input_size = dataset.height * dataset.width;

    Matrix W1 = Matrix::init_he(hidden_size, input_size, rand);
    Matrix b1(hidden_size, 1);
    Matrix W2 = Matrix::init_he(num_labels, hidden_size, rand);
    Matrix b2(num_labels, 1);

    std::cout << "************\tTRAINING\t*************" << "\n";
    train(dataset, W1, b1, W2, b2, epochs, learning_rate);

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

            Matrix dW2 = predictions.multiply(a1.transpose());

            Matrix da1 = W2.transpose().multiply(predictions);
            Matrix dz1 = da1.hadamard(Activations::relu_derivative(z1));

            Matrix dW1 = dz1.multiply(x.transpose());

            W2.subtract_scaled(dW2, learning_rate);
            b2.subtract_scaled(predictions, learning_rate);
            W1.subtract_scaled(dW1, learning_rate);
            b1.subtract_scaled(dz1, learning_rate);
        }
        float avg_loss = tot_loss / static_cast<float>(dataset.num_samples);
        float accuracy = static_cast<float>(correct) / static_cast<float>(dataset.num_samples);
        std::cout << "Epoch: " << epoch + 1 << " | loss: " << avg_loss << " | accuracy: " << accuracy << "\n";
    }
}

