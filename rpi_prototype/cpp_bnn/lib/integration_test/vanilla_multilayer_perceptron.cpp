#include <iostream>
#include <chrono>
#include <memory>
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/base_layer.h"
#include "../utils/csv_reader.h"
#include "../naive_layers/vanilla_layers/dense/vanilla_dense.h"
#include "../naive_layers/vanilla_layers/batchnorm/vanilla_batchnorm.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"
#include "../optimizers/optimizers.h"


int main(int argc, char * argv[]) {
    const int BATCH_SIZE = atoi(argv[1]);
    const int EPOCH = atoi(argv[2]);
    
    // Matrix<float> test_input({BATCH_SIZE, 784});
    // Matrix<float> test_label({BATCH_SIZE, 10});
    // glorot_normal_initializer<float>(test_input);
    
    //for (size_t i=0; i<test_label.shape()[0]; i++) {
    //    test_label.set(1, i, i%test_label.shape()[1]);
    //}
    
    std::string input_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_X_train.csv";
    std::string label_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_y_train.csv";
    
    // Model Architecture 
    const float BATCH_NORM_MOMENTUM = 0.9;
    
    if (EPOCH != 0) {
        std::vector<std::unique_ptr<BaseLayer<Matrix>>> layer_seq;
        layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(256, true, 0));
        layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix>>());

        layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(256, false, 1));
        layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix>>());

        layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(256, false, 2));
        layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix>>());

        layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(256, false, 3));
        layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix>>());

        layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(10, false, 4));
        layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<Softmax<float, Matrix>>());

        CrossEntropy<float, Matrix> cross_entropy_1;
        Adam Adam_opt(1e-3, 0.9, 0.999, 10);
        // ----------------------------------------
        for (int i=0; i<EPOCH; i++) {
            {
                printf("Epoch %d: \n", i);
                printf("start forward --------------------------------------\n");
                
                auto start = std::chrono::high_resolution_clock::now();
                Matrix<float> mnist_in = read_csv<float>(input_filename, {BATCH_SIZE,784}, i*BATCH_SIZE);
                // Matrix<float> test_output = test_input;
                for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                    mnist_in = layer_seq[_lay]->forward(mnist_in, true);
                    /*if (_lay==12) {
                        //printf("layer %d: %f\n", _lay, mnist_in[9]);
                        //print_mat(layer_seq[_lay]->get_weight());
                        //print_mat(mnist_in);
                    }
                    if (_lay==4) {
                        //printf("layer %d: %f\n", _lay, mnist_in[9]);
                        //print_mat(mnist_in);
                    }
                    if (std::isinf(mnist_in[9])) {
                         throw std::runtime_error("infinity");
                    }
                    if (std::isnan(mnist_in[9])) {
                         throw std::runtime_error("nan");
                    }*/
                }
                Matrix<float> mnist_label = read_csv<float>(label_filename, {BATCH_SIZE,10}, i*BATCH_SIZE);
                mnist_in = cross_entropy_1.forward(mnist_in, mnist_label, true);
                
                auto stop = std::chrono::high_resolution_clock::now();
                printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
                float loss = average_loss(mnist_in);
                printf("Average loss: %f \n", loss);
            }

            {
                printf("start backprop --------------------------------------\n");
                auto start_2 = std::chrono::high_resolution_clock::now();
                Matrix<float> mnist_label = read_csv<float>(label_filename, {BATCH_SIZE,10}, i*BATCH_SIZE);

                mnist_label = cross_entropy_1.backprop(mnist_label);
                for (int _lay=layer_seq.size()-1; _lay>0; _lay--) {
                    mnist_label = layer_seq[_lay]->backprop(mnist_label);
                    /*//if (_lay==1) {
                    //    print_mat(test_output_back);
                    //}
                    if (_lay==4) {
                        //printf("layer %d: %f\n", _lay, mnist_label[9]);
                        //print_mat(mnist_label);
                    }
                    if (_lay==5) {
                        //printf("layer %d: %f\n", _lay, mnist_label[9]);
                        //print_mat(mnist_label);
                    }*/
                }
                Matrix<float> mnist_in = read_csv<float>(input_filename, {BATCH_SIZE,784}, i*BATCH_SIZE);
                layer_seq[0]->backprop(mnist_label, mnist_in);
                
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            {
                printf("start Adam --------------------------------------\n");
                auto start = std::chrono::high_resolution_clock::now();
                Adam_opt.update(layer_seq);
                auto stop = std::chrono::high_resolution_clock::now();
                printf("Update (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            }
        }
    }
    
    return 0;
}