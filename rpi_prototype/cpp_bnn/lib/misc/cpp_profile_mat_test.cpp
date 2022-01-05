#include <iostream>
#include "boost_tutorial.h"
#include <unistd.h>
#include <chrono>
#include <thread>
#include <string>
#include <stdio.h>
#include <stdlib.h>



enum string_code {
    boost_bitset,
    boost_bool,
    boost_char,
    boost_float,
    vector_bool,
    eigen_bool,
    eigen_char,
    eigen_float,
    arma_float,
    arma_sint,
    arma_char
};

string_code hashit (std::string const& inString) {
    if (inString == "boost_bitset") return boost_bitset;
    if (inString == "boost_bool") return boost_bool;
    if (inString == "boost_char") return boost_char;
    if (inString == "boost_float") return boost_float;
    if (inString == "vector_bool") return vector_bool;
    if (inString == "eigen_bool") return eigen_bool;
    if (inString == "eigen_char") return eigen_char;
    if (inString == "eigen_float") return eigen_float;
    if (inString == "arma_sint") return arma_sint;
    if (inString == "arma_char") return arma_char;
    return arma_float;
};


int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[1]);
    //std::vector<std::string> allArgs(argv, argv + argc);
    std::cout << arma_float;
    switch (hashit(argv[2])) {
        case boost_bitset:
            //std::cout << "1\n";
            boost_dynamic_bitset_1d(SIZE*SIZE);
            boost_dynamic_bitset_2d(SIZE);
            {
                auto start_2 = std::chrono::high_resolution_clock::now();
                dynamic_bitset_matmul_bool_2d(SIZE);
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("eigen %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;

        case boost_bool:
            //std::cout << "5\n";
            boost_matrix_bool_1d(SIZE*SIZE);
            boost_matrix_bool_2d(SIZE);
            boost_matmul_bool_2d(SIZE);
            break;

        case boost_char:
            //std::cout << "5\n";
            boost_matrix_char_1d(SIZE*SIZE);
            boost_matrix_char_2d(SIZE);
            boost_matmul_char_2d(SIZE);
            break;

        case boost_float:
            //std::cout << "5\n";
            boost_matrix_float_1d(SIZE*SIZE);
            boost_matrix_float_2d(SIZE);
            boost_matmul_float_2d(SIZE);
            break;
        case vector_bool:
            //std::cout << "5\n";
            vector_bool_1d(SIZE*SIZE);
            vector_bool_2d(SIZE);
            {
                auto start_2 = std::chrono::high_resolution_clock::now();
                vector_matmul_bool_2d(SIZE);
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("eigen %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case eigen_bool:
            eigen_matrix_bool_1d(SIZE);
            eigen_matrix_bool_2d(SIZE);
            eigen_matmul_bool_2d(SIZE);
            break;
            
        case eigen_char:
            eigen_matrix_char_1d(SIZE);
            eigen_matrix_char_2d(SIZE);
            eigen_matmul_char_2d(SIZE);
            break;
            
            
        case eigen_float:
            eigen_matrix_float_1d(SIZE);
            eigen_matrix_float_2d(SIZE);
            eigen_matmul_float_2d(SIZE);
            break;
            
        case arma_float:
            arma_matrix_float_1d(SIZE);
            arma_matrix_float_2d(SIZE);
            arma_matmul_float_2d(SIZE);
            break;
            
        case arma_sint:
            arma_matrix_sint_1d(SIZE);
            arma_matrix_sint_2d(SIZE);
            arma_matmul_sint_2d(SIZE);
            break;
            
        case arma_char:
            arma_matrix_char_1d(SIZE);
            arma_matrix_char_2d(SIZE);
            arma_matmul_char_2d(SIZE);
            break;
            
    }
    return 0;
}