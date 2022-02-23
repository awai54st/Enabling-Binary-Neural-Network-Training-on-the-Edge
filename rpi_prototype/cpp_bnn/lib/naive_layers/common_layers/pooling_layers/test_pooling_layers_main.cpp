#include <iostream>
#include <chrono>
#include "pooling_layers.h"
#include "../../../utils/initialiser.h"


void test_max_pooling() {
    std::vector<size_t> size = {2, 4, 4, 3};
    Matrix<float16_t> test_input(size);
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    test_input[0] = 10.0;
    test_input[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test max pooling layer \n");
    print_mat(test_input);
    MaxPooling<bool, Matrix, float16_t> test_MaxPooling(2, 2);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_MaxPooling.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    ones_like(test_output);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_MaxPooling.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}
/*
void test_max_pooling_2d() {
    std::vector<size_t> size = {2, 4, 4, 3};
    Matrix2D<float16_t> test_input(size);
    Matrix2D<float16_t> test_output;
    Matrix2D<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    test_input[0] = 10.0;
    test_input[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test max pooling 2d layer \n");
    print_mat(test_input);
    MaxPooling<bool, Matrix2D, float16_t> test_MaxPooling(2, 2);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_MaxPooling.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    ones_like(test_output);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_MaxPooling.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}
*/

int main(int argc, char * argv[]) {
    test_max_pooling();
    //test_max_pooling_2d();
    
    return 0;
}
/*
void test_speed_activation(const int size) {
    Matrix2D<float> test_input(size, size);
    glorot_normal_initializer_2d(test_input);

    test_input.data[0] = 02;
    test_input.data[3] = -5;
    std::cout << "test input: \n";
    //print_mat2d<Matrix2D<float>>(test_input);
    BinaryActivation test_BinaryActivation;

    auto start = std::chrono::high_resolution_clock::now();
    Matrix2D<float> test_output = test_BinaryActivation.forward(test_input, true);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    std::cout << test_input.data[0] << "\n";
    std::cout << test_output.data[0] << "\n";
    std::cout << test_input.data[3] << "\n";
    std::cout << test_output.data[3] << "\n";


    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix2D<float> test_output_back = test_BinaryActivation.backprop(test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    
}

void test_activation() {
    Matrix2D<float> test_input(4, 5, 1);
    Matrix2D<float> test_output;
    Matrix2D<float> test_output_backprop;
    glorot_normal_initializer_2d<float>(test_input);
    test_input.data[0] = 10.0;
    test_input.data[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test binary activation layer(is first)\n");
    print_mat2d(test_input);
    BinaryActivation test_BinaryActivation;
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_BinaryActivation.forward(test_input, true);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat2d(test_output);
    ones_initializer_2d(test_input);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_BinaryActivation.backprop(test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat2d(test_output_backprop);
}

int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    switch (CHOISE){
        case 0:
            {
                std::cout << "Test activation layer\n";
                test_activation();
            }
            break;
            
        case 1:
            {
                test_speed_activation(SIZE);
            }
            break;
    }
    
    return 0;
}
*/