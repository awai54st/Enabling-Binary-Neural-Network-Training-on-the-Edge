#include <iostream>
#include <chrono>
#include "test_code.h"
#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "gradient_quantisation_utils.h"
#include "dense.h"

void test_DynamicBitset2D() {
    Matrix2D<bool> test_db;
    test_db.init(4, 4);
    std::cout << "Size: " << test_db.data.size() << "\n";
    std::cout << "m x n: " << test_db.n_rows << " x " << test_db.n_cols << "\n";
    test_db.data[1] = 1;
    test_db.data[3] = 1;
    test_db.data[15] = 1;
    print_mat2d<Matrix2D<bool>>(test_db);
    
}

void test_Vector2D() {
    Matrix2D<float> test_db(4, 4, 1);
    std::cout << "Size: " << test_db.data.size() << "\n";
    std::cout << "m x n: " << test_db.n_rows << " x " << test_db.n_cols << "\n";
    test_db.data[1] = 10.92;
    test_db.data[3] = 20.43;
    test_db.data[15] = 15;
    std::cout << "data: " << test_db.data[1] << "\n";
    std::cout << "data: " << test_db.data[0] << "\n";
}

void test_glorot_normal_initializer_2d() {
    Matrix2D<float> test_db(4, 4, 1);
    glorot_normal_initializer_2d<float>(test_db);
    print_mat2d<Matrix2D<float>>(test_db);
}

void speed_test_glorot_normal_initializer_2d() {
    Matrix2D<float> test_db(1024, 1024, 1);
    auto start_2 = std::chrono::high_resolution_clock::now();
    glorot_normal_initializer_2d<float>(test_db);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("Init (use : %ld us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
}

/*
void test_bit2word() {
    Matrix2D<float> test_word(2, 3, 1);
    Matrix2D<boost::dynamic_bitset<>> test_bit(2, 3, 5);
    glorot_normal_initializer_2d(test_word);
    std::cout << "input: \n";
    print_mat2d<Matrix2D<float>>(test_word);
    
    word2bit(test_word.data, test_bit, 0, 5);
    print_mat2d<Matrix2D<float>>(test_word);
    //void bit2word(float & word, float & from_bit, int word_length = 5);
}
void test_po2_quantisation() {
    std::cout << "\n test_po2_quantisation: \n";
    Matrix2D<float> test_input(3, 4, 1);
    glorot_normal_initializer_2d(test_input);
    
    std::cout << "Input: \n";
    print_mat2d<Matrix2D<float>>(test_input);
    
    float test_bias = get_po2_bias(test_input);
    std::cout << "Bias: " << test_bias << "\n";
    
    scale_with_po2_bias(test_input, test_bias);
    std::cout << "Outptut: \n";
    print_mat2d<Matrix2D<float>>(test_input);
    
    log_quantize(test_input, test_bias);
    std::cout << "Outptut (gradient quantized): \n";
    print_mat2d<Matrix2D<float>>(test_input);
}
void test_DenseNotFirst() {
    Matrix2D<float> test_input(2, 3, 1);
    Matrix2D<float> test_output_forward;
    Matrix2D<float> test_output_backprop;
    glorot_normal_initializer_2d(test_input);
    DenseNotFirst test_dense(5);
    test_output_forward = test_dense.forward(test_input);
    test_output_backprop = test_dense.backprop(test_output_forward);
    
    std::cout << "input: \n";
    print_mat2d<Matrix2D<float>>(test_input);
    print_mat2d<Matrix2D<boost::dynamic_bitset<>>>(test_dense.packed_x);
    std::cout << "kernel: \n";
    print_mat2d<Matrix2D<float>>(test_dense.kernel);
    std::cout << "output (forward): \n";
    print_mat2d<Matrix2D<float>>(test_output_forward);
    std::cout << "output (backprop): \n";
    print_mat2d<Matrix2D<float>>(test_output_backprop);
}
*/