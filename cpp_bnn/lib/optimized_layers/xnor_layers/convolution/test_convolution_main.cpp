#include <iostream>
#include "opt_xnor_convolution.h"


int main() {
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float16_t> x(x_shape);
    glorot_normal_initializer<float16_t>(x);
    
    printf("Test input valid: \n");
    print_mat(x);
    
    
    OPT_XNor_Convolution2D test_Convolution2D(1, 3);
    Matrix<float16_t> test_output = test_Convolution2D.forward(x);
    printf("Test kernel: \n");
    print_mat(test_Convolution2D.kernel);
    
    printf("Test output: \n");
    print_mat(test_output);
    
    ones_like(test_output);
    Matrix<float16_t> test_output_backprop = test_Convolution2D.backprop(test_output);
    printf("Test output backprop: \n");
    print_mat(test_output_backprop);
    
    printf("Test dkernel: \n");
    print_mat(test_Convolution2D.dkernel);
    
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