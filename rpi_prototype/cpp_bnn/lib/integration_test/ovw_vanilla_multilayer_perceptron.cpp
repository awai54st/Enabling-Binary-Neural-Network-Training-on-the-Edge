#include <iostream>
#include <chrono>
#include <memory>
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../naive_layers_ov/vanilla_layers/dense/vanilla_dense_ovw.h"
#include "../naive_layers/vanilla_layers/batchnorm/vanilla_batchnorm.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"
#include "../utils/base_layer.h"

// type alias
//using BaseLayerClass = BaseLayer<Matrix2D>

int main(int argc, char * argv[]) {
    printf("---------------------------------------\n");
    printf("Train ovw_vanilla_multilayer_perceptron\n");
    const int BATCH_SIZE = atoi(argv[1]);
    const int EPOCH = atoi(argv[2]);
    
    Matrix2D<float> test_input({BATCH_SIZE, 784});
    Matrix2D<float> test_label({BATCH_SIZE, 10});
    glorot_normal_initializer<float>(test_input);
    
    for (size_t i=0; i<test_label.shape()[0]; i++) {
            test_label.set(1, i, i%test_label.shape()[1]);
    }
    
    // Model Architecture 
    const float BATCH_NORM_MOMENTUM = 0.9;
    
    
    std::vector<std::unique_ptr<BaseLayer<Matrix2D>> > layer_seq;
    layer_seq.push_back(std::make_unique<Vanilla_Dense_OVW>(256, true, 0));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix2D>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix2D>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_Dense_OVW>(256, false, 1));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix2D>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix2D>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_Dense_OVW>(256, false, 2));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix2D>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix2D>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_Dense_OVW>(256, false, 3));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix2D>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<float, Matrix2D>>());
    
    layer_seq.push_back(std::make_unique<Vanilla_Dense_OVW>(10, false, 4));
    layer_seq.push_back(std::make_unique<Vanilla_BatchNormDense<Matrix2D>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<Softmax<Matrix2D>>());
    
    CrossEntropy<Matrix2D> cross_entropy_1;
    // ----------------------------------------
    
    if (EPOCH != 0) {
        for (int i=0; i<EPOCH; i++) {
            {
                printf("Epoch %d: \n", i);
                printf("start forward --------------------------------------\n");
                auto start = std::chrono::high_resolution_clock::now();
                
                Matrix2D<float> test_output = test_input;
                for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                    test_output = layer_seq[_lay]->forward(test_output, true);
                }
                test_output = cross_entropy_1.forward(test_output, test_label, true);
                
                auto stop = std::chrono::high_resolution_clock::now();
                printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            }

            {
                printf("start backprop --------------------------------------\n");
                auto start_2 = std::chrono::high_resolution_clock::now();

                Matrix2D<float> test_output_back = cross_entropy_1.backprop(test_label);
                for (int _lay=layer_seq.size()-1; _lay>0; _lay--) {
                    test_output_back = layer_seq[_lay]->backprop(test_output_back);
                    //if (_lay==1) {
                    //    print_mat(test_output_back);
                    //}
                }
                layer_seq[0]->backprop(test_output_back, test_input);
                
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
        }
    } else {
        Matrix2D<float> test_output = test_input;
        Matrix2D<float> test_output_back = cross_entropy_1.backprop(test_label);
    }
    
    return 0;
}