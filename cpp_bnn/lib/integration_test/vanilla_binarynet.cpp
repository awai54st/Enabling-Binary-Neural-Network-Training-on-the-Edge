#include <iostream>
#include <chrono>
#include <memory>
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../naive_layers/vanilla_layers/dense/vanilla_dense.h"
#include "../naive_layers/vanilla_layers/convolution/vanilla_convolution.h"
#include "../naive_layers/vanilla_layers/batchnorm/vanilla_batchnorm.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"
#include "../naive_layers/common_layers/pooling_layers/pooling_layers.h"
#include "../naive_layers/common_layers/flatten_layer/flatten_layer.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"
#include "../utils/base_layer.h"


int main(int argc, char * argv[]) {
    const int BATCH_SIZE = atoi(argv[1]);
    const int EPOCH = atoi(argv[2]);
    
    Matrix<float> test_input({BATCH_SIZE, 32, 32, 3});
    Matrix<float> test_label({BATCH_SIZE, 10});
    glorot_normal_initializer<float>(test_input);
    
    for (size_t i=0; i<test_label.shape()[0]; i++) {
            test_label.set(1, i, i%test_label.shape()[1]);
    }
    
    // Model Architecture 
    const float BATCH_NORM_MOMENTUM = 0.9;
    
    std::vector<std::unique_ptr<BaseLayer>> layer_seq;
    layer_seq.push_back(std::make_unique<Vanilla_Convolution2D>(128, 3, "same", true, 0));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormConv>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    layer_seq.push_back(std::make_unique<Vanilla_Convolution2D>(128, 3, "same", false, 1));
    layer_seq.push_back(std::make_unique<MaxPooling<float>>(2, 2));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormConv>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_Convolution2D>(256, 3, "same", false, 2));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormConv>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    layer_seq.push_back(std::make_unique<Vanilla_Convolution2D>(256, 3, "same", false, 3));
    layer_seq.push_back(std::make_unique<MaxPooling<float>>(2, 2));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormConv>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_Convolution2D>(512, 3, "same", false, 4));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormConv>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    layer_seq.push_back(std::make_unique<Vanilla_Convolution2D>(512, 3, "same", false, 5));
    layer_seq.push_back(std::make_unique<MaxPooling<float>>(2, 2));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormConv>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    
    layer_seq.push_back(std::make_unique<Flatten>());
    
    layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(1024, false, 0));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(1024, false, 1));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_BNNDense>(10, false, 4));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<Softmax>());
    
    CrossEntropy cross_entropy_1;
    
    // ----------------------------------------
    
    if (EPOCH != 0) {
        for (int i=0; i<EPOCH; i++) {
            {
                printf("Epoch %d: \n", EPOCH);
                auto start = std::chrono::high_resolution_clock::now();
                
                Matrix<float> test_output = test_input;
                for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                    test_output = layer_seq[_lay]->forward(test_output, true);
                }
                test_output = cross_entropy_1.forward(test_output, test_label, true);
                
                auto stop = std::chrono::high_resolution_clock::now();
                printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
                //print_mat(test_output);
            }

            {
                printf("start backprop --------------------------------------\n");
                auto start_2 = std::chrono::high_resolution_clock::now();

                Matrix<float> test_output_back = cross_entropy_1.backprop(test_label);
                for (int _lay=layer_seq.size()-1; _lay>-1; _lay--) {
                    test_output_back = layer_seq[_lay]->backprop(test_output_back);
                }
                
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
        }
    } else {
        Matrix<float> test_output = test_input;
    }
    
    return 0;
}