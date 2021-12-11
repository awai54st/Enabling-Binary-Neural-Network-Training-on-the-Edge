#include "softmax.h"
#include "../../../utils/initialiser.h"
#include <chrono>


void test_softmax() {
    
    Matrix<float16_t> test_input({2, 3});
    glorot_normal_initializer<float16_t>(test_input);
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Softmax\n");
    print_mat(test_input);
    Softmax<float16_t, Matrix> test_softmax;
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float16_t> test_output = test_softmax.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix<float16_t> test_output_backprop = test_softmax.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}

/*
void test_softmax_2d() {
    
    Matrix2D<float> test_input({2, 3});
    glorot_normal_initializer<float>(test_input);
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Softmax 2d\n");
    print_mat(test_input);
    Softmax<float, Matrix2D> test_softmax;
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix2D<float> test_output = test_softmax.forward(test_input);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    printf("\nForward result \n");
    print_mat(test_output);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    Matrix2D<float> test_output_backprop = test_softmax.backprop(test_output);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("\n Backward (use : %li us) \n\n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    printf("\nBackprop Result \n");
    print_mat(test_output_backprop);
}
*/
int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    
    test_softmax();
    //test_softmax_2d();
    
    
    return 0;
}