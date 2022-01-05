#include <iostream>
#include <chrono>
#include <memory>
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/base_layer.h"
#include "../utils/csv_reader.h"
#include "../optimizers/optimizers.h"
#include "../optimized_layers/xnor_layers/dense/opt_xnor_dense.h"
#include "../optimized_layers/xnor_layers/convolution/opt_xnor_convolution.h"
#include "../naive_layers/xnor_layers/batchnorm/xnor_batchnorm.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"
#include "../naive_layers/common_layers/pooling_layers/pooling_layers.h"
#include "../naive_layers/common_layers/flatten_layer/flatten_layer.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"
#include "../optimizers/optimizers.h"


int main(int argc, char * argv[]) {
    const int BATCH_SIZE = atoi(argv[1]);
    const int EPOCH = atoi(argv[2]);
    
    std::string input_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/cifar10/cifar10_x.csv";
    std::string label_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/cifar10/cifar10_y.csv";
    //print_mat(test_label);
    
    // Model Architecture 
    const float BATCH_NORM_MOMENTUM = 0.9;
    
    std::vector<std::unique_ptr<BaseLayer<Matrix, float16_t>>> layer_seq;
    
    layer_seq.push_back(std::make_unique<OPT_XNor_Convolution2D>(128, 3, "same", true, 0));
    layer_seq.push_back(std::make_unique<XNor_BatchNormConv<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    layer_seq.push_back(std::make_unique<OPT_XNor_Convolution2D>(128, 3, "same", false, 0));
    layer_seq.push_back(std::make_unique<MaxPooling<bool, Matrix, float16_t>>(2, 2));
    layer_seq.push_back(std::make_unique<XNor_BatchNormConv<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    
    layer_seq.push_back(std::make_unique<OPT_XNor_Convolution2D>(256, 3, "same", false, 0));
    layer_seq.push_back(std::make_unique<XNor_BatchNormConv<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    layer_seq.push_back(std::make_unique<OPT_XNor_Convolution2D>(256, 3, "same", false, 0));
    layer_seq.push_back(std::make_unique<MaxPooling<bool, Matrix, float16_t>>(2, 2));
    layer_seq.push_back(std::make_unique<XNor_BatchNormConv<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    
    layer_seq.push_back(std::make_unique<OPT_XNor_Convolution2D>(512, 3, "same", false, 0));
    layer_seq.push_back(std::make_unique<XNor_BatchNormConv<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    layer_seq.push_back(std::make_unique<OPT_XNor_Convolution2D>(512, 3, "same", false, 0));
    layer_seq.push_back(std::make_unique<MaxPooling<bool, Matrix, float16_t>>(2, 2));
    layer_seq.push_back(std::make_unique<XNor_BatchNormConv<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());


    layer_seq.push_back(std::make_unique<Flatten<Matrix, float16_t>>());
    
    layer_seq.push_back(std::make_unique<OPT_XNor_Dense>(1024, false, 0));
    layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    
    layer_seq.push_back(std::make_unique<OPT_XNor_Dense>(1024, false, 1));
    layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());
    
    layer_seq.push_back(std::make_unique<OPT_XNor_Dense>(10, false, 2));
    layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
    layer_seq.push_back(std::make_unique<Softmax<float16_t, Matrix>>());
    
    CrossEntropy<float16_t, Matrix> cross_entropy_1;
    Adam<float16_t> Adam_opt(1e-3, 0.9, 0.999, 10);
    
    // ----------------------------------------
    for (int i=0; i<EPOCH; i++) {
        {
            printf("Epoch %d: \n", i);
            printf("start forward --------------------------------------\n");
            auto start = std::chrono::high_resolution_clock::now();
            Matrix<float16_t> mnist_in = read_csv<float16_t>(input_filename, {BATCH_SIZE,32*32*3}, 0, BATCH_SIZE*32*32*3);
            mnist_in.reshape(BATCH_SIZE, 32, 32, 3);
            for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                mnist_in = layer_seq[_lay]->forward(mnist_in, true);
                //print_shape(test_output);
            }
            Matrix<float16_t> mnist_label = read_csv<float16_t>(label_filename, {BATCH_SIZE,10}, 0);
            mnist_in = cross_entropy_1.forward(mnist_in, mnist_label, true);

            auto stop = std::chrono::high_resolution_clock::now();
            printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            float loss = average_loss(mnist_in);
            printf("Average loss: %f \n", loss);
            //print_mat(test_output);
        }

        {
            printf("start backprop --------------------------------------\n");
            auto start_2 = std::chrono::high_resolution_clock::now();

            Matrix<float16_t> mnist_label = read_csv<float16_t>(label_filename, {BATCH_SIZE,10}, 0, BATCH_SIZE*10);
            mnist_label = cross_entropy_1.backprop(mnist_label);
            for (int _lay=layer_seq.size()-1; _lay>0; _lay--) {
                mnist_label = layer_seq[_lay]->backprop(mnist_label);
                //print_shape(test_output_back);
            }
            Matrix<float16_t> mnist_in = read_csv<float16_t>(input_filename, {BATCH_SIZE,32*32*3}, 0);
            mnist_in.reshape(BATCH_SIZE, 32, 32, 3);
            layer_seq[0]->backprop(mnist_label, mnist_in);

            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            //print_mat(test_output_back);
        }
        {
            printf("start Adam --------------------------------------\n");
            auto start = std::chrono::high_resolution_clock::now();
            Adam_opt.update(layer_seq);
            auto stop = std::chrono::high_resolution_clock::now();
            printf("Update (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
        }
    }
    
    return 0;
}