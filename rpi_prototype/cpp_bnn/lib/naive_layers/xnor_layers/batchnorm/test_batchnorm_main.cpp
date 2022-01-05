#include <iostream>
#include <chrono>
#include "xnor_batchnorm.h"
#include "../../../utils/initialiser.h"
/*
void test_batch_norm_2d(const int SIZE) {
    Matrix<float16_t> test_input({SIZE, 256});
    glorot_normal_initializer(test_input);
    
    std::cout << "test input: \n";
    //print_mat2d<Matrix2D<float>>(test_input);
    XNor_BatchNormDense test_BatchNormDense(0.9);

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
*/
void test_batchnorm_dense() {
    printf("test_batchnorm_dense ------------------------------------- \n");
    Matrix<float16_t> test_input({2, 3});
    glorot_normal_initializer<float16_t>(test_input);
    printf("test_input: \n");
    printf("size: %ld \n", test_input.size());
    print_mat(test_input);
    
    //print_mat2d<Matrix2D<float>>(test_input);
    XNor_BatchNormDense test_BatchNormDense(0.9);
    Matrix<float16_t> test_output = test_BatchNormDense.forward(test_input, true);
    printf("test output: \n");
    print_mat(test_output);
    
    
    Matrix<float16_t> test_output_backprop = test_BatchNormDense.backprop(test_output);
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
    Matrix<float16_t> test_input({2, 3, 3, 2});
    glorot_normal_initializer<float16_t>(test_input);
    printf("test_batchnorm_conv ------------------------------------- \n");
    printf("test_input: \n");
    printf("size: %ld \n", test_input.size());
    print_mat(test_input);
    
    //print_mat2d<Matrix2D<float>>(test_input);
    XNor_BatchNormConv test_XNor_BatchNormConv(0.9);
    Matrix<float16_t> test_output = test_XNor_BatchNormConv.forward(test_input, true);
    printf("test output: \n");
    print_mat(test_output);
    
    
    Matrix<float16_t> test_output_backprop = test_XNor_BatchNormConv.backprop(test_output);
    printf("test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("test dbeta: \n");
    print_mat(test_XNor_BatchNormConv.dbeta);
    
    printf("test moving mean: \n");
    print_mat(test_XNor_BatchNormConv.moving_mean);
    
    printf("test moving var: \n");
    print_mat(test_XNor_BatchNormConv.moving_var);
}

void test_batchnorm_dense_nan() {
    Matrix<float16_t> test_input({2, 10}, 0);
    std::vector<float16_t> a ={8 , -6 , 2 , -10 , -30 , -10 , 14 , 0 , -14 , 2 , -8 , 6 , -2 , 10 , 30 , 10 , -14 , 0 , 14 , -2};
    test_input.m_data = a;
    //glorot_normal_initializer<float>(test_input, 0);
    //for (int i = 0; i<test_input.size(); i++) {
    //    test_input[i] *= 700;
    //}
    printf("test_input: \n");
    print_mat(test_input);
    
    XNor_BatchNormDense test_BatchNormDense(0.9);
    Matrix<float16_t> test_output = test_BatchNormDense.forward(test_input, true);
    printf("test output: \n");
    print_mat(test_output);
    
    //glorot_normal_initializer<float>(test_output, 1);
    std::vector<float16_t> b = {0.198234 , 0.026828 , 0.198234 , 0.026828 , 0.026828 , -0.973172 , 0.198234 , 0.072926 , 0.026828 , 0.198234 , -0.977098 , 0.169227 , 0.0229024 , 0.169227 , 0.169227 , 0.169227 , 0.0229024 , 0.0622552, 0.169227 , 0.0229024};
    test_output.m_data = b;
    Matrix<float16_t> test_output_backprop = test_BatchNormDense.backprop(test_output);
    printf("test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("test dbeta: \n");
    print_mat(test_BatchNormDense.dbeta);
    
    printf("test moving mean: \n");
    print_mat(test_BatchNormDense.moving_mean);
    
    printf("test moving var: \n");
    print_mat(test_BatchNormDense.moving_var);
    
    printf("test var: \n");
    print_mat(test_BatchNormDense.var);
    
}

int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[1]);
    test_batchnorm_conv();
    test_batchnorm_dense();
    //test_batchnorm_dense_nan();
    
    return 0;
}