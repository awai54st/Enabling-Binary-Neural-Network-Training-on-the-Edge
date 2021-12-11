#include <iostream>
#include <chrono>
#include <memory>
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/base_layer.h"
#include "../utils/csv_reader.h"
#include "../naive_layers/xnor_layers/dense/xnor_dense.h"
#include "../naive_layers/xnor_layers/batchnorm/xnor_batchnorm.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"
#include "../naive_layers/common_layers/softmax_layer/softmax.h"
#include "../naive_layers/loss/loss.h"
#include "../optimizers/optimizers.h"
//#include "../utils/check_utils.h"


int main(int argc, char * argv[]) {
    const int BATCH_SIZE = atoi(argv[1]);
    const int EPOCH = atoi(argv[2]);
    
    // Matrix<float> test_input({BATCH_SIZE, 784});
    // Matrix<float> test_label({BATCH_SIZE, 10});
    // glorot_normal_initializer<float>(test_input);
    // glorot_normal_initializer<float>(test_label);
    
    std::string input_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_X_train.csv";
    std::string label_filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_y_train.csv";
    
    // Model Architecture 
    const float BATCH_NORM_MOMENTUM = 0.9;
    
    if (EPOCH != 0) {
        std::vector<std::unique_ptr<BaseLayer<Matrix, float16_t>>> layer_seq;
        layer_seq.push_back(std::make_unique<XNor_Dense>(256, true, 0)); //0
        layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM)); //1
        layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>()); //2

        layer_seq.push_back(std::make_unique<XNor_Dense>(256, false, 1));//3
        layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));//4
        layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());//5

        layer_seq.push_back(std::make_unique<XNor_Dense>(256, false, 2));
        layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());//8

        layer_seq.push_back(std::make_unique<XNor_Dense>(256, false, 3));
        layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));
        layer_seq.push_back(std::make_unique<BinaryActivation<bool, Matrix, float16_t>>());//11

        layer_seq.push_back(std::make_unique<XNor_Dense>(10, false, 4));
        layer_seq.push_back(std::make_unique<XNor_BatchNormDense<Matrix>>(BATCH_NORM_MOMENTUM));//13
        layer_seq.push_back(std::make_unique<Softmax<float16_t, Matrix>>());//14

        CrossEntropy<float16_t, Matrix> cross_entropy_1;
        Adam<float16_t> Adam_opt(1e-3, 0.9, 0.999, 10);
        // ----------------------------------------
        for (int i=0; i<EPOCH; i++) {
            {
                printf("Epoch %d: \n", i);
                printf("start forward --------------------------------------\n");
                
                auto start = std::chrono::high_resolution_clock::now();
                //Matrix<float> mnist_in = read_csv<float>(input_filename, {BATCH_SIZE,784}, i*BATCH_SIZE);
                //Matrix<float> mnist_label = read_csv<float>(label_filename, {BATCH_SIZE,10}, i*BATCH_SIZE);
                Matrix<float16_t> mnist_in = read_csv<float16_t>(input_filename, {BATCH_SIZE,784}, 0);
                // Matrix<float> test_output = test_input;
                for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                    mnist_in = layer_seq[_lay]->forward(mnist_in, true);
                    /*
                    printf("layer[%d]\n -------------------", _lay);
                    if ((_lay==11) || (_lay==12) || (_lay==13)) {
                        //print_mat(mnist_in);
                        //print_mat(layer_seq[_lay]->get_weight());
                        //printf("layer %d: %f\n", _lay, mnist_in[9]);
                    }
                    if ((_lay==2) || (_lay==5) || (_lay==8) || (_lay==11) || (_lay==14)) {
                    //    print_mat(mnist_label);
                    //    print_mat(layer_seq[_lay]->get_gradient());
                    //    printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //    print_mat(mnist_label);
                    } else {
                        //printf("layer %d: %f\n", _lay, mnist_in[9]);
                        //has_inf(layer_seq[_lay]->get_weight());
                        //has_nan(layer_seq[_lay]->get_weight());
                        //printf("layer %d: %f\n", _lay, mnist_label[9]);
                    }
                    has_inf(mnist_in);
                    has_nan(mnist_in);
                    //*/
                }
                //printf("Predicted: "); print_mat(mnist_in);
                //printf("Ground truth: "); print_mat(mnist_label);
                Matrix<float16_t> mnist_label = read_csv<float16_t>(label_filename, {BATCH_SIZE,10}, 0);
                mnist_in = cross_entropy_1.forward(mnist_in, mnist_label, true);
                //has_inf(mnist_in);
                //has_nan(mnist_in);
                auto stop = std::chrono::high_resolution_clock::now();
                printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
                float loss = average_loss(mnist_in);
                printf("Average loss: %f \n", loss);
            }

            {
                printf("start backprop --------------------------------------\n");
                
                auto start_2 = std::chrono::high_resolution_clock::now();
                //Matrix<float> mnist_label = read_csv<float>(label_filename, {BATCH_SIZE,10}, i*BATCH_SIZE);
                Matrix<float16_t> mnist_label = read_csv<float16_t>(label_filename, {BATCH_SIZE,10}, 0);
                mnist_label.m_data.reserve(BATCH_SIZE*784);

                mnist_label = cross_entropy_1.backprop(mnist_label);
                //printf("First backprop -------------------");
                //        print_mat(mnist_label);
                for (int _lay=layer_seq.size()-1; _lay>0; _lay--) {
                    mnist_label = layer_seq[_lay]->backprop(mnist_label);
                    /*
                    printf("layer[%d]\n -------------------", _lay);
                    //if (layer_seq[_lay]->get_gradient().data()==NULL) {
                    //    continue;
                    //}
                    //printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //print_mat(mnist_label);
                    if (_lay==14) {
                    //    print_mat(layer_seq[_lay]->get_gradient());
                    //    printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //    print_mat(mnist_label);
                    }
                    if (_lay==13) {
                    //    print_mat(layer_seq[_lay]->get_gradient());
                    //    printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //    print_mat(mnist_label);
                    }
                    if (_lay==12) {
                    //    print_mat(layer_seq[_lay]->get_gradient());
                    //    printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //    print_mat(mnist_label);
                    }
                    if ((_lay==2) || (_lay==5) || (_lay==8) || (_lay==11) || (_lay==14)) {
                    //    print_mat(mnist_label);
                    //    print_mat(layer_seq[_lay]->get_gradient());
                    //    printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //    print_mat(mnist_label);
                    } else {
                        //printf("layer %d: %f\n", _lay, mnist_label[9]);
                        //has_inf(layer_seq[_lay]->get_gradient());
                        //has_nan(layer_seq[_lay]->get_gradient());
                    }
                    //has_inf(mnist_label);
                    //has_nan(mnist_label);//*/
                }
    
                Matrix<PO2_5bits_t> mnist_label_PO2 = PO2_gradients(mnist_label);
                mnist_label.resize({BATCH_SIZE,784});
                //mnist_label.m_data.shrink_to_fit();
                //Matrix<float> mnist_in = read_csv<float>(input_filename, {BATCH_SIZE,784}, i*BATCH_SIZE);
                //Matrix<float16_t> mnist_in = read_csv<float16_t>(input_filename, {BATCH_SIZE,784}, 0);
                read_csv<float16_t>(input_filename, mnist_label, {BATCH_SIZE,784}, 0);
                //layer_seq[0]->backprop(mnist_label, mnist_in);
                layer_seq[0]->backprop(mnist_label_PO2, mnist_label);
                //has_inf(layer_seq[0]->get_gradient());
                //has_nan(layer_seq[0]->get_gradient());
                
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            {
                printf("start Adam --------------------------------------\n");
                auto start = std::chrono::high_resolution_clock::now();
                Adam_opt.update(layer_seq);
                auto stop = std::chrono::high_resolution_clock::now();
                
                /*// check update
                for (int _lay=0; _lay<layer_seq.size(); _lay++) {
                    printf("Adam layer[%d]\n -------------------", _lay);
                    if ((_lay==2) || (_lay==5) || (_lay==8) || (_lay==11) || (_lay==14)) {
                    //    print_mat(mnist_label);
                    //    print_mat(layer_seq[_lay]->get_gradient());
                    //    printf("layer %d: %f\n", _lay, mnist_label[9]);
                    //    print_mat(mnist_label);
                    } else {
                        printf("Adam layer %d: %f\n", _lay);
                        has_inf(layer_seq[_lay]->get_weight());
                        has_nan(layer_seq[_lay]->get_weight());
                        has_inf(layer_seq[_lay]->get_gradient());
                        has_nan(layer_seq[_lay]->get_gradient());
                        //printf("layer %d: %f\n", _lay, mnist_label[9]);
                    }
                }*/
                printf("Update (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            }
        }
    }
    
    return 0;
}