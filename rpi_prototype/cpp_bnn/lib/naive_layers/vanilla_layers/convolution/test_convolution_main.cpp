#include <iostream>
#include "vanilla_convolution.h"

void test_convolution_valid() {
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    printf("Test input: \n");
    print_mat(x);
    
    
    Vanilla_Convolution2D test_Convolution2D(1, 3);
    Matrix<float> test_output = test_Convolution2D.forward(x);
    printf("Test kernel: \n");
    print_mat(test_Convolution2D.kernel);
    
    printf("Test output: \n");
    print_mat(test_output);
    
    ones_like(test_output);
    Matrix<float> test_output_backprop = test_Convolution2D.backprop(test_output);
    printf("Test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("Test dkernel: \n");
    print_mat(test_Convolution2D.dkernel);
}

void test_convolution_same() {
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    printf("Test input: \n");
    print_mat(x);
    
    
    Vanilla_Convolution2D test_Convolution2D(1, 3, "same");
    Matrix<float> test_output = test_Convolution2D.forward(x);
    printf("Test kernel: \n");
    print_mat(test_Convolution2D.kernel);
    
    printf("Test output: \n");
    print_mat(test_output);
    
    ones_like(test_output);
    Matrix<float> test_output_backprop = test_Convolution2D.backprop(test_output);
    printf("Test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("Test dkernel: \n");
    print_mat(test_Convolution2D.dkernel);
}

void test_convolution_same_more_c() {
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    printf("Test input: \n");
    print_mat(x);
    
    
    Vanilla_Convolution2D test_Convolution2D(5, 3, "same");
    Matrix<float> test_output = test_Convolution2D.forward(x);
    printf("Test kernel: \n");
    print_mat(test_Convolution2D.kernel);
    
    printf("Test output: \n");
    print_mat(test_output);
    
    ones_like(test_output);
    Matrix<float> test_output_backprop = test_Convolution2D.backprop(test_output);
    printf("Test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("Test dkernel: \n");
    print_mat(test_Convolution2D.dkernel);
}


int main() {
    test_convolution_valid();
    test_convolution_same();
    test_convolution_same_more_c();
    //std::vector<size_t> kernel_shape = {3, 3, 3, 2};
    //Matrix<bool> kernel(kernel_shape);
    //ones_like<bool>(kernel);
    
    
    //Matrix<float> output = conv_op(x, kernel);
    //print_mat(test_output);
    
    //output = conv_op(x, kernel, "same");
    //print_mat(output);
    
    return 0;
}



/*
int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    return 0;
}
*/