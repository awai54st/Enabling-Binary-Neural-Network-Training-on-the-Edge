#include <iostream>
#include <chrono>
#include "vanilla_batchnorm.h"
#include "../../../utils/initialiser.h"

/*
void test_batch_norm_2d(const int SIZE) {
    Matrix<float> test_input({SIZE, 256});
    glorot_normal_initializer(test_input);
    
    std::cout << "test input: \n";
    //print_mat2d<Matrix2D<float>>(test_input);
    Vanilla_BatchNormDense test_BatchNormDense(0.9);

    std::cout << test_input.size() << ": before batch norm \n";
    auto start = std::chrono::high_resolution_clock::now();
    test_input = test_BatchNormDense.forward(test_input, true);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << test_input.size() << ": after batch norm \n";
    printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));


    auto start_2 = std::chrono::high_resolution_clock::now();
    test_input = test_BatchNormDense.backprop(test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
}

void test_batchnorm_dense_2d() {
    Matrix2D<float> test_input({2, 3});
    glorot_normal_initializer<float>(test_input);
    printf("test_input: \n");
    printf("size: %ld \n", test_input.size());
    print_mat(test_input);
    
    //print_mat2d<Matrix2D<float>>(test_input);
    Vanilla_BatchNormDense<Matrix2D> test_BatchNormDense(0.9);
    Matrix2D<float> test_output = test_BatchNormDense.forward(test_input, true);
    printf("test output: \n");
    print_mat(test_output);
    
    
    Matrix2D<float> test_output_backprop = test_BatchNormDense.backprop(test_output);
    printf("test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("test dbeta: \n");
    print_mat(test_BatchNormDense.dbeta);
    
    printf("test moving mean: \n");
    print_mat(test_BatchNormDense.moving_mean);
    
    printf("test moving var: \n");
    print_mat(test_BatchNormDense.moving_var);
}
*/
void test_batchnorm_dense() {
    Matrix<float> test_input({2, 3});
    glorot_normal_initializer<float>(test_input);
    printf("test_input: \n");
    printf("size: %ld \n", test_input.size());
    print_mat(test_input);
    
    //print_mat2d<Matrix2D<float>>(test_input);
    Vanilla_BatchNormDense test_BatchNormDense(0.9);
    Matrix<float> test_output = test_BatchNormDense.forward(test_input, true);
    printf("test output: \n");
    print_mat(test_output);
    
    
    Matrix<float> test_output_backprop = test_BatchNormDense.backprop(test_output);
    printf("test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("test dbeta: \n");
    print_mat(test_BatchNormDense.dbeta);
    
    printf("test moving mean: \n");
    print_mat(test_BatchNormDense.moving_mean);
    
    printf("test moving var: \n");
    print_mat(test_BatchNormDense.moving_var);
}

void test_batchnorm_conv() {
    Matrix<float> test_input({2, 3, 4, 5});
    glorot_normal_initializer<float>(test_input);
    printf("test_input: \n");
    printf("size: %ld \n", test_input.size());
    print_mat(test_input);
    
    //print_mat2d<Matrix2D<float>>(test_input);
    Vanilla_BatchNormConv test_BatchNormConv(0.9);
    Matrix<float> test_output = test_BatchNormConv.forward(test_input, true);
    printf("test output: \n");
    print_mat(test_output);
    
    
    Matrix<float> test_output_backprop = test_BatchNormConv.backprop(test_output);
    printf("test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("test dbeta: \n");
    print_mat(test_BatchNormConv.dbeta);
    
    printf("test moving mean: \n");
    print_mat(test_BatchNormConv.moving_mean);
    
    printf("test moving var: \n");
    print_mat(test_BatchNormConv.moving_var);
}

int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[1]);
    
    test_batchnorm_dense();
    //test_batchnorm_dense_2d();
    test_batchnorm_conv();
    
    return 0;
}