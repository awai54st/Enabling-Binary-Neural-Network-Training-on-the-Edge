#include "flatten_layer.h"
#include <iostream>
#include <chrono>
#include "../../../utils/initialiser.h"


void test_flatten() {
    std::vector<size_t> size = {2, 4, 4, 3};
    Matrix<float> test_input(size);
    Matrix<float> test_output;
    Matrix<float> test_output_backprop;
    glorot_normal_initializer<float>(test_input);
    test_input[0] = 10.0;
    test_input[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test flatten layer \n");
    print_mat(test_input);
    Flatten test_Flatten;
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_Flatten.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_Flatten.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}
/*
void test_flatten_2d() {
    std::vector<size_t> size = {2, 4, 4, 3};
    Matrix2D<float> test_input(size);
    Matrix2D<float> test_output;
    Matrix2D<float> test_output_backprop;
    glorot_normal_initializer<float>(test_input);
    test_input[0] = 10.0;
    test_input[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test flatten layer \n");
    print_mat(test_input);
    Flatten<Matrix2D> test_Flatten;
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_Flatten.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_Flatten.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}
*/
int main(int argc, char * argv[]) {
    test_flatten();
    //test_flatten_2d();
    
    return 0;
}