#include <iostream>
#include <Eigen/Dense>
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#define EIGEN_USE_BLAS

// eigen char
typedef Eigen::Matrix<signed char,Eigen::Dynamic,1> ArrayXc;
typedef Eigen::Matrix<signed char,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> ArrayXXc;
using  ArrayXXs = Eigen::Matrix<short,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>;
using  ArrayXXf = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>;

void eigen_matrix_char_1d(int SIZE) {
    ArrayXc Mat1f = ArrayXc::Constant(SIZE,1);
    // ArrayXb::Constant(5,true);
}
void eigen_matrix_char_2d(int SIZE) {
    ArrayXXc Mat2b = ArrayXXc::Constant(SIZE, SIZE,1);
}
void eigen_matmul_char_2d(int SIZE) {
    ArrayXXs a = ArrayXXs::Constant(SIZE, SIZE,-1);
    ArrayXXs b = ArrayXXs::Constant(SIZE, SIZE,1);
    
    ArrayXXs tmp = (a*b).eval();
    std::cout<<"output: ";
    std::cout << int(tmp(10,10)) << "\n";
    std::cout << int(a(10,10)) << "\n";
    signed char x =1;
    signed char y =-1;
    std::cout << x*y << "\n";
}

void eigen_matmul_short_2d(int SIZE) {
    ArrayXXs a = ArrayXXs::Constant(SIZE, SIZE,-1);
    ArrayXXs b = ArrayXXs::Constant(SIZE, SIZE,1);
    
    ArrayXXs tmp = (a*b).eval();
    //std::cout<<"output: ";
    //std::cout << int(tmp(10,10)) << "\n";
}

// eigen float
void eigen_matrix_float_1d(int SIZE) {
    Eigen::MatrixXf Mat1f = Eigen::MatrixXf::Random(SIZE, SIZE);
    // ArrayXb::Constant(5,true);
}
void eigen_matrix_float_2d(int SIZE) {
    Eigen::MatrixXf Mat2b = Eigen::MatrixXf::Random(SIZE, SIZE);
}
void eigen_matmul_float_2d(int SIZE) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Constant(SIZE,SIZE, -1);
    Eigen::MatrixXf b = Eigen::MatrixXf::Constant(SIZE,SIZE, 1);
    
    Eigen::MatrixXf tmp = (a*b).eval();
    std::cout<<"output: ";
    std::cout << tmp(10,10);
    std::cout << a(10,10);
}


enum string_code {
    int16,
    mix_8_32,
    float32
};

string_code hashit (std::string const& inString) {
    if (inString == "int16") return int16;
    if (inString == "mix_8_32") return mix_8_32;
    return float32;
};

int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[2]); 
    switch (hashit(argv[1])) {
        case int16:
            {
            ArrayXXs a = ArrayXXs::Constant(SIZE, SIZE,-1);
            ArrayXXs b = ArrayXXs::Constant(SIZE, SIZE,1);

            auto start_2 = std::chrono::high_resolution_clock::now();
            ArrayXXs tmp = (a*b).eval();
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Eigen3 %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32:
            {
            Eigen::MatrixXf a = Eigen::MatrixXf::Constant(SIZE,SIZE, -1);
            Eigen::MatrixXf b = Eigen::MatrixXf::Constant(SIZE,SIZE, 1);

            auto start_2 = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXf tmp = a*b;
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Eigen3 [float32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case mix_8_32:
            {
            ArrayXXc a = ArrayXXc::Constant(SIZE,SIZE, -1);
            ArrayXXf b = ArrayXXf::Constant(SIZE,SIZE, 1.0);
            ArrayXXf d = ArrayXXf::Constant(SIZE,SIZE, 0.0);

            auto start_2 = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i<SIZE; i++) {
                // cast char to float, then carry out dot product
                d.row(i) = a.row(i).cast<float>()*b;
            }
                
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Eigen3 [mix8_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
    }
    
    return 0;
}