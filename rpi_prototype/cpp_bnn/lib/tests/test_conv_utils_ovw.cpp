#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/conv_utils_ovw.h"


void test_convolution() {
    printf("Test convolution ---------------------------\n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    std::vector<size_t> kernel_shape = {3, 3, 3, 3};
    Matrix<bool> kernel(kernel_shape);
    ones_like<bool>(kernel);
    printf("Test input: ");
    print_mat(x);
    printf("kernel: ");
    print_mat_bool(kernel);
    
    /*
    printf("Test convolution valid ---------------------------\n");
    Matrix<float> output_2({2, 2, 2, 2}, 1);
    convolution(x, kernel, output_2, 1, 0);
    
    print_mat(output_2);
    */
    printf("Test convolution same ---------------------------\n");
    _convolution<float, bool>(x, kernel, "same");
    printf("shape of output 1: "); print_shape(x);
    print_mat(x);
}


void test_convolution_more() {
    printf("Test convolution more output channel ---------------------------\n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    std::vector<size_t> kernel_shape = {3, 3, 3, 5};
    Matrix<bool> kernel(kernel_shape);
    ones_like<bool>(kernel);
    printf("Test input: ");
    print_mat(x);
    printf("kernel: ");
    print_mat_bool(kernel);
    
    printf("Test convolution moc same ---------------------------\n");
    _convolution<float, bool>(x, kernel, "same");
    printf("shape of output 1: "); print_shape(x);
    print_mat(x);
}


void test_convolution_valid_more() {
    printf("Test convolution more output channel ---------------------------\n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    std::vector<size_t> kernel_shape = {3, 3, 3, 5};
    Matrix<bool> kernel(kernel_shape);
    ones_like<bool>(kernel);
    printf("Test input: ");
    print_mat(x);
    printf("kernel: ");
    print_mat_bool(kernel);
    
    printf("Test convolution moc valid ---------------------------\n");
    _convolution(x, kernel, "valid");
    printf("shape of output 2: "); print_shape(x);
    print_mat(x);
}


int main() {
    //test_convolution();
    //test_convolution_more();
    test_convolution_valid_more();
    return 0;
}