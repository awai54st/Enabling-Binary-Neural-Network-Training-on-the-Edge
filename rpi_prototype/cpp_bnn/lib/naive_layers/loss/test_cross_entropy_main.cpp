#include <chrono>
#include "../../utils/initialiser.h"
#include "loss.h"
#include "../common_layers/softmax_layer/softmax.h"



void test_ce_loss() {
    
    Matrix<float> test_input({2, 3});
    glorot_normal_initializer<float>(test_input);
    Matrix<float> test_label({2, 3});
    test_label.m_data = {1, 0, 0, 0, 0, 1};
    
    CrossEntropy<float> test_ce;
    Softmax<float> test_softmax;
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Cross Entropy\n");
    print_mat(test_input);
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float> test_output = test_softmax.forward(test_input);
    test_output = test_ce.forward(test_output, test_label);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    float loss = average_loss<float>(test_output);
    printf("Average loss: %f", loss);
    
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix<float> test_output_backprop = test_ce.backprop(test_label);
    test_output_backprop = test_softmax.backprop(test_output_backprop);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}

void test_ce_loss_2() {
    Matrix<float> test_input({3, 10});
    glorot_normal_initializer<float>(test_input);
    Matrix<float> test_label({3, 10});
    for (size_t i=0; i<test_label.shape()[0]; i++) {
        for (size_t j=0; j<test_label.shape()[1]; j++) {
            if (i==j) {
                test_label.set(1, i, j);
            }
        }
    }
    
    
    CrossEntropy<float> test_ce;
    Softmax<float> test_softmax;
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Cross Entropy\n");
    print_mat(test_input);
    print_mat(test_label);
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float> test_output = test_softmax.forward(test_input);
    test_output = test_ce.forward(test_output, test_label);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    float loss = average_loss<float>(test_output);
    printf("Average loss: %f", loss);
    
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix<float> test_output_backprop = test_ce.backprop(test_label);
    test_output_backprop = test_softmax.backprop(test_output_backprop);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}


void test_ce_loss_2d() {
    
    Matrix2D<float> test_input({2, 3});
    glorot_normal_initializer<float>(test_input);
    Matrix2D<float> test_label({2, 3});
    test_label.m_data[0] = {1, 0, 0};
    test_label.m_data[1] = {0, 0, 1};
    
    CrossEntropy<float, Matrix2D> test_ce;
    Softmax<float, Matrix2D> test_softmax;
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Cross Entropy 2d\n");
    print_mat(test_input);
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix2D<float> test_output = test_softmax.forward(test_input);
    test_output = test_ce.forward(test_output, test_label);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix2D<float> test_output_backprop = test_ce.backprop(test_label);
    test_output_backprop = test_softmax.backprop(test_output_backprop);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}


int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    
    //test_ce_loss();
    test_ce_loss_2();
    //test_ce_loss_2d();
    
    return 0;
}