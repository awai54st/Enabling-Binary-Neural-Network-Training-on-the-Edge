#include <iostream>
#include <chrono>
#include "xnor_dense.h"


void test_speed_DenseNotFirst(const int size) {
    Matrix<float16_t> test_input({size, size});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer(test_input);
    XNor_Dense test_dense(size);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_dense.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_dense.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    
}

void test_DenseFirst() {
    Matrix<float16_t> test_input({2, 3});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Dense(is first)\n");
    print_mat(test_input);
    XNor_Dense test_dense(4, true);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_dense.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("Weights \n");
    print_mat(test_dense.kernel);
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_dense.backprop(test_output, test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("dweight \n");
    print_mat(test_dense.dkernel);
    printf("\nBackprop Result \n");
    //print_mat(test_output_backprop);
}

void test_DenseNotFirst() {
    Matrix<float16_t> test_input({2, 3});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Dense(is not first)\n");
    print_mat(test_input);
    XNor_Dense test_dense(4, false, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_dense.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("Weights \n");
    print_mat(test_dense.kernel);
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_dense.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("dweight \n");
    print_mat(test_dense.dkernel);
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}

void test_DenseNotFirst_2() {
    Matrix<float16_t> test_input({3, 10});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Dense(is not first)\n");
    print_mat(test_input);
    XNor_Dense test_dense(4, false, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_dense.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("Weights \n");
    print_mat(test_dense.kernel);
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_dense.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("dweight \n");
    print_mat(test_dense.dkernel);
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}


void test_DenseFirst_PO2() {
    Matrix<float16_t> test_input({2, 3});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Dense(is first)\n");
    print_mat(test_input);
    XNor_Dense test_dense(4, true);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_dense.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("Weights \n");
    print_mat(test_dense.kernel);
    printf("\nForward result \n");
    print_mat(test_output);
    
    Matrix<PO2_5bits_t> test_output_PO2 = PO2_gradients(test_output);
    // print_PO2(test_output_PO2);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_dense.backprop(test_output_PO2, test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("dweight \n");
    print_mat(test_dense.dkernel);
    printf("\nBackprop Result \n");
    //print_mat(test_output_backprop);
}

void test_DenseNotFirst_check_nan() {
    Matrix<float16_t> test_input({3, 10}, 0);
    //glorot_normal_initializer<float>(test_input);
    //for (int i = 0; i<test_input.size(); i++) {
    //    test_input[i] *= 700;
    //}
    
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Dense check nan\n");
    print_mat(test_input);
    XNor_Dense test_dense(4, false, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_dense.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("Weights \n");
    print_mat(test_dense.kernel);
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_dense.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("dweight \n");
    print_mat(test_dense.dkernel);
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}


int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    switch (CHOISE){
        case 0:
            {
                std::cout << "Test dense layer\n";
                test_DenseNotFirst();
                //test_DenseNotFirst_2();
                //test_DenseFirst();
                //test_DenseNotFirst_check_nan();
                test_DenseFirst();
                test_DenseFirst_PO2();
            }
            break;
            
        case 1:
            {
                test_speed_DenseNotFirst(SIZE);
            }
            break;
    }
    
    return 0;
}