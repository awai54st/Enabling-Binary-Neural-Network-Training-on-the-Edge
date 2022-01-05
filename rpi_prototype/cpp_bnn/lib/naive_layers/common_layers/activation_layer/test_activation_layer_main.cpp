#include <iostream>
#include <chrono>
#include "activation_layer.h"
#include "../../../utils/initialiser.h"


void test_speed_activation(const int size) {
    Matrix<float> test_input({size, size});
    glorot_normal_initializer(test_input);

    test_input.m_data[0] = 02;
    test_input.m_data[3] = -5;
    std::cout << "test input: \n";
    //print_mat2d<Matrix2D<float>>(test_input);
    BinaryActivation<float> test_BinaryActivation;

    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float> test_output = test_BinaryActivation.forward(test_input, true);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    std::cout << test_input.m_data[0] << "\n";
    std::cout << test_output.m_data[0] << "\n";
    std::cout << test_input.m_data[3] << "\n";
    std::cout << test_output.m_data[3] << "\n";


    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix<float> test_output_back = test_BinaryActivation.backprop(test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    
}

void test_activation() {
    Matrix<float16_t> test_input({4, 5});
    Matrix<float16_t> test_output;
    Matrix<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    test_input.m_data[0] = 10.0;
    test_input.m_data[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test binary activation layer(is first)\n");
    print_mat(test_input);
    BinaryActivation<bool, Matrix, float16_t> test_BinaryActivation;
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_BinaryActivation.forward(test_input, true);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    ones_like(test_input);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_BinaryActivation.backprop(test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}

/*
void test_activation_2d() {
    Matrix2D<float16_t> test_input({4, 5});
    Matrix2D<float16_t> test_output;
    Matrix2D<float16_t> test_output_backprop;
    glorot_normal_initializer<float16_t>(test_input);
    test_input[0] = 10.0;
    test_input[4] = -10.0;
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test binary activation layer(is first)\n");
    print_mat(test_input);
    BinaryActivation<bool, Matrix2D, float16_t> test_BinaryActivation;
    
    auto start = std::chrono::high_resolution_clock::now();
    test_output = test_BinaryActivation.forward(test_input, true);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    ones_like(test_input);
    auto start_2 = std::chrono::high_resolution_clock::now();
    test_output_backprop = test_BinaryActivation.backprop(test_input);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}
*/
int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    switch (CHOISE){
        case 0:
            {
                std::cout << "Test activation layer\n";
                test_activation();
                //test_activation_2d();
            }
            break;
            
        case 1:
            {
                test_speed_activation(SIZE);
            }
            break;
    }
    
    return 0;
}