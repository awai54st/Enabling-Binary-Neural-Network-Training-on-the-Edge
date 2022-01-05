#include <iostream>
#include <chrono>
#include <memory>
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/base_layer.h"
#include "../utils/csv_reader.h"
#include "../optimizers/optimizers.h"
#include "../naive_layers_ovw/xnor_layers/dense/xnor_dense_ovw.h"
#include "../naive_layers_ovw/xnor_layers/convolution/xnor_convolution_ovw.h"
#include "../naive_layers/xnor_layers/batchnorm/xnor_batchnorm.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"
#include "../naive_layers/common_layers/pooling_layers/pooling_layers.h"
#include "../naive_layers/common_layers/flatten_layer/flatten_layer.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"


int main(int argc, char * argv[]) {
    const int BATCH_SIZE = atoi(argv[1]);
    const int EPOCH = atoi(argv[2]);
    
    std::string input_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_X_train.csv";
    std::string label_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_y_train.csv";
    
    // Model Architecture 
    const float BATCH_NORM_MOMENTUM = 0.9;
    
    if (EPOCH != 0) {
    
        std::vector<std::unique_ptr<BaseLayer<Matrix, float16_t, Matrix<float16_t>&>>> layer_seq(17);
        layer_seq[0] = std::make_unique<XNor_Convolution2D_OWH1D>(32, 3, "same", true, 0);
        layer_seq[1] = std::make_unique<XNor_BatchNormConv<Matrix, Matrix<float16_t>&>>(BATCH_NORM_MOMENTUM);
        layer_seq[2] = std::make_unique<BinaryActivation<bool, Matrix, float16_t, Matrix<float16_t>&>>();

        layer_seq[3] = std::make_unique<XNor_Convolution2D_OWH1D>(32, 3, "same", false, 0);
        layer_seq[4] = std::make_unique<XNor_BatchNormConv<Matrix, Matrix<float16_t>&>>(BATCH_NORM_MOMENTUM);
        layer_seq[5] = std::make_unique<BinaryActivation<bool, Matrix, float16_t, Matrix<float16_t>&>>();
        layer_seq[6] = std::make_unique<MaxPooling<bool, Matrix, float16_t, Matrix<float16_t>&>>(2, 2);

        layer_seq[7] = std::make_unique<Flatten<Matrix, float16_t, Matrix<float16_t>&>>();

        layer_seq[8] = std::make_unique<XNor_Dense_OVW1D>(64, false, 1);
        layer_seq[9] = std::make_unique<XNor_BatchNormDense<Matrix, Matrix<float16_t>&>>(BATCH_NORM_MOMENTUM);
        layer_seq[10] = std::make_unique<BinaryActivation<bool, Matrix, float16_t, Matrix<float16_t>&>>();

        layer_seq[11] = std::make_unique<XNor_Dense_OVW1D>(64, false, 1);
        layer_seq[12] = std::make_unique<XNor_BatchNormDense<Matrix, Matrix<float16_t>&>>(BATCH_NORM_MOMENTUM);
        layer_seq[13] = std::make_unique<BinaryActivation<bool, Matrix, float16_t, Matrix<float16_t>&>>();

        layer_seq[14] = std::make_unique<XNor_Dense_OVW1D>(10, false, 1);
        layer_seq[15] = std::make_unique<XNor_BatchNormDense<Matrix, Matrix<float16_t>&>>(BATCH_NORM_MOMENTUM);
        layer_seq[16] = std::make_unique<Softmax<float16_t, Matrix, Matrix<float16_t>&>>();

        CrossEntropy<float16_t, Matrix> cross_entropy_1;
        Adam<float16_t, Matrix<float16_t>&> Adam_opt(1e-3, 0.9, 0.999, 10);

        // ----------------------------------------
        for (int i=0; i<EPOCH; i++) {
            {
                printf("Epoch %d: \n", i);
                printf("start forward --------------------------------------\n");
                auto start = std::chrono::high_resolution_clock::now();
                Matrix<float16_t> mnist_in;
                //mnist_in.m_data.reserve({BATCH_SIZE*784*17});
                mnist_in.resize({BATCH_SIZE,784});
                read_csv<float16_t>(input_filename, mnist_in, {BATCH_SIZE,784}, 0);
                printf("mnist_in: "); print_shape(mnist_in);
                mnist_in.reshape(BATCH_SIZE, 28, 28, 1);
                //layer_seq[0]->forward(mnist_in, true);
                for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                    //mnist_in = layer_seq[_lay]->forward(mnist_in, true);
                    layer_seq[_lay]->forward(mnist_in, true);
                    printf("layer [%d]: ", _lay); print_shape(mnist_in);
                    //print_shape(test_output);
                }
                Matrix<float16_t> mnist_label;
                mnist_label.resize({BATCH_SIZE,10});
                read_csv<float16_t>(label_filename, mnist_label, {BATCH_SIZE,10}, 0);
                cross_entropy_1.forward(mnist_in, mnist_label, true);
                
                auto stop = std::chrono::high_resolution_clock::now();
                printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
                //float loss = average_loss(mnist_in);
                //printf("Average loss: %f \n", loss);
                //print_mat(test_output);
            }
            {
                printf("start backprop --------------------------------------\n");
                auto start_2 = std::chrono::high_resolution_clock::now();

                Matrix<float16_t> mnist_label = read_csv<float16_t>(label_filename, {BATCH_SIZE,10}, 0);
                mnist_label = cross_entropy_1.backprop(mnist_label);
                for (int _lay=layer_seq.size()-1; _lay>0; _lay--) {
                    mnist_label = layer_seq[_lay]->backprop(mnist_label);
                    //print_shape(test_output_back);
                }
                Matrix<float16_t> mnist_in = read_csv<float16_t>(input_filename, {BATCH_SIZE,784}, 0);
                mnist_in.reshape(BATCH_SIZE, 28, 28, 1);
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
    }
    
    return 0;
}