//#include <iostream>
#include "initializers.h"
#include "gradient_quantisation.h"
#include "bit_utils.h"
#include "dense.h"
#include <unistd.h>
#include <chrono>
#include <thread>


int main(void) {
    VECTOR_2D_T<float> tmp;
    VECTOR_2D_T<float> output;
    VECTOR_2D_T<float> output_back;
    tmp = glorot_normal_initializer_2d(2048, 2048, 0);
    
    DenseNotFirst_CPP dense_test(2048, true);
    auto start = std::chrono::high_resolution_clock::now();
    output = dense_test.forward(tmp);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    
    auto start_2 = std::chrono::high_resolution_clock::now();
    output_back = dense_test.backprop(tmp);
    auto stop_2 = std::chrono::high_resolution_clock::now();
    printf("Backward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
    return 0;
}