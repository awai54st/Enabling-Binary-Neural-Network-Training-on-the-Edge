#include <iostream>
#include "vanilla_convolution_ovw.h"

void test_convolution_valid() {
    printf("Test Vanilla_Convolution2D_OWH1D valid -------------------------------: \n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    printf("Test input: \n");
    print_mat(x);
    
    
    Vanilla_Convolution2D_OWH1D test_Convolution2D(1, 3);
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
    printf("Test Vanilla_Convolution2D_OWH1D same -------------------------------: \n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    printf("Test input: \n");
    print_mat(x);
    
    
    Vanilla_Convolution2D_OWH1D test_Convolution2D(1, 3, "same");
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

void test_convolution_same_more_filter() {
    printf("Test Vanilla_Convolution2D_OWH1D more filter---------------------------: \n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    printf("Test input: \n");
    print_mat(x);
    
    
    Vanilla_Convolution2D_OWH1D test_Convolution2D(5, 3, "same");
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
    test_convolution_same_more_filter();
    
    
    return 0;
}



/*
int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    return 0;
}
*/