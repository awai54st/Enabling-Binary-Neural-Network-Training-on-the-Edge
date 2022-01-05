#include <iostream>
#include <chrono>
#include "xnor_dense_po2.h"


void test_DenseNotFirst() {
    Matrix<float16_t> test_input({2, 3});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Dense(is not first)\n");
    print_mat(test_input);
    XNor_Dense_PO2 test_dense(4, false, 0);
    
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
    
    
    printf("PO2_5bits_t float--------------------------\n");
    
    {
        PO2_5bits_t test_po2(1, 1);
        float a = 1.4123;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f*2 = %f\n", a, a*2);
    }
    {
        PO2_5bits_t test_po2(1, -1);
        float a = 1.4123;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f/2 = %f\n", a, a/2);
    }
    
    {
        PO2_5bits_t test_po2(-1, 1);
        float a = 1.4123;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f/2 = %f\n", a, a*-2);
    }
    
    {
        PO2_5bits_t test_po2(-1, -1);
        float a = 1.4123;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f/-2 = %f\n", a, a/-2);
    }
    printf("PO2_5bits_t float16_t--------------------------\n");
    
    {
        PO2_5bits_t test_po2(1, 1);
        float16_t a = 1.2345;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f*2 = %f\n", a, a*2);
    }
    {
        PO2_5bits_t test_po2(1, -1);
        float16_t a = 1.2345;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f/2 = %f\n", a, a/2);
    }
    
    {
        PO2_5bits_t test_po2(-1, 1);
        float16_t a = 1.2345;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f/2 = %f\n", a, a*-2);
    }
    
    {
        PO2_5bits_t test_po2(-1, -1);
        float16_t a = 1.2345;

        printf("PO2_5bits_t-> sign: %d, value: %d * %f = %f\n", 
               int(test_po2.sign), int(test_po2.value), a, test_po2*a);
        printf("%f/-2 = %f\n", a, a/-2);
    }
    {
        test_DenseNotFirst();
    }
    return 0;
}