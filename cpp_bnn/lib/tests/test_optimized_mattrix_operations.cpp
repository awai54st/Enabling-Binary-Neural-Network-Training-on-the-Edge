#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/optimized_matrix_operations.h"


void test_matmul() {
    printf("Test matmul -----------------------------------\n");
    Matrix<float> A({2, 4}, 1);
    A.m_data = {1, 2, 3, 4, 5, 6, 7, 8};
    Matrix<float> B({4, 2}, 1);
    B.m_data = {1, 2, 3, 4, 5, 6, 7, 8};
    Matrix<float> C({2, 2}, 0);
    printf("Before matmul: \n");
    
    print_mat(A);
    print_mat(B);
    print_mat(C);
    
    matmul(A, B, C);
    printf("After matmul: \n");
    print_mat(A);
    print_mat(B);
    print_mat(C);
}

void test_matmul_transa() {
    printf("Test matmul_transa ---------------------------\n");
    Matrix<float> A({3, 2}, 1);
    A.m_data = {1, 2, 3, 4, 5, 6};
    Matrix<float> B({3, 2}, 1);
    B.m_data = {1, 2, 3, 4, 5, 6};
    Matrix<float> C({2, 2}, 0);
    printf("Before matmul: \n");
    
    print_mat(A);
    print_mat(B);
    print_mat(C);
    
    matmul_transa(A, B, C);
    printf("After matmul: \n");
    print_mat(A);
    print_mat(B);
    print_mat(C);
}

void test_matmul_transb() {
    printf("Test matmul_transb ---------------------------\n");
    Matrix<float> A({2, 3}, 1);
    A.m_data = {1, 2, 3, 4, 5, 6};
    Matrix<float> B({2, 3}, 1);
    B.m_data = {1, 2, 3, 4, 5, 6};
    Matrix<float> C({2, 2}, 0);
    printf("Before matmul: \n");
    
    print_mat(A);
    print_mat(B);
    print_mat(C);
    
    matmul_transb(A, B, C);
    printf("After matmul: \n");
    print_mat(A);
    print_mat(B);
    print_mat(C);
}

void test_convolution() {
    printf("Test convolution ---------------------------\n");
    std::vector<size_t> x_shape = {2, 4, 4, 3};
    Matrix<float> x(x_shape);
    glorot_normal_initializer<float>(x);
    
    std::vector<size_t> kernel_shape = {3, 3, 3, 2};
    Matrix<bool> kernel(kernel_shape);
    ones_like<bool>(kernel);
    print_mat(x);
    print_mat_bool(kernel);
    
    printf("Test convolution valid ---------------------------\n");
    Matrix<float> output_2({2, 2, 2, 2}, 1);
    convolution(x, kernel, output_2, 1, 0);
    
    print_mat(output_2);
    
    printf("Test convolution same ---------------------------\n");
    Matrix<float> output_1({2, 4, 4, 2}, 0);
    convolution(x, kernel, output_1, 1, 0);
    printf("shape of output 1: "); print_shape(output_1);
    print_mat(output_1);
}

int main() {
    //test_matmul();
    //test_matmul_transa();
    //test_matmul_transb();
    test_convolution();
    return 0;
}