#include <iostream>
#include <armadillo>
#include <cstdint>
#include <chrono>

// arma sint
typedef arma::Mat<char> c_mat;
typedef arma::Mat<short> s_mat;
typedef arma::Mat<float> f_mat;
void arma_matrix_sint_1d(int SIZE) {
    s_mat A(SIZE, 1);
    A.zeros();
}
void arma_matrix_sint_2d(int SIZE) {
    s_mat A(SIZE, SIZE);
    A.zeros();
}

void arma_matmul_sint_2d(int SIZE) {
    //arma::Mat<float> A(SIZE,SIZE,arma::fill::ones);
    s_mat A(SIZE,SIZE);
    s_mat B(SIZE,SIZE);
    A.ones();
    B.ones();
    
    s_mat C = (A*B).eval();
    std::cout << C(1,1);
}


enum string_code {
    int16,
    float32,
    mix_16_32,
    mix_8_32
};

string_code hashit (std::string const& inString) {
    if (inString == "int16") return int16;
    if (inString == "mix_16_32") return mix_16_32;
    if (inString == "mix_8_32") return mix_8_32;
    return float32;
};

int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[2]); 
    switch (hashit(argv[1])) {
        case int16:
            {
            s_mat A(SIZE,SIZE);
            s_mat B(SIZE,SIZE);
            A.ones();
            A *= -1;
            B.ones();

            auto start_2 = std::chrono::high_resolution_clock::now();
            s_mat C = (A*B).eval();
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Armadillo9 %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32:
            {
            f_mat A(SIZE,SIZE);
            f_mat B(SIZE,SIZE);
            A *= -1;
            A.ones();
            B.ones();
                
            auto start_2 = std::chrono::high_resolution_clock::now();
            // Use float32 variables in dot product
            f_mat C = (A*B).eval();
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Armadillo9 [float32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case mix_16_32:
            {
            s_mat A(SIZE,SIZE);
            f_mat B(SIZE,SIZE);
            f_mat C(SIZE,SIZE);
            A.ones();
            A *= -1;
            B.ones();
                
            auto start_2 = std::chrono::high_resolution_clock::now();
            
            for (arma::uword i = 0; i<SIZE; i++) {
                // cast short to float, then carry out dot product
                C.row(i) = arma::conv_to<arma::Row<float>>::from(A.row(i))*B;
            }
                
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Armadillo9 %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            
        case mix_8_32:
            {
            c_mat A(SIZE,SIZE);
            f_mat B(SIZE,SIZE);
            f_mat C(SIZE,SIZE);
            A.ones();
            A *= -1;
            B.ones();
                
            auto start_2 = std::chrono::high_resolution_clock::now();
            
            for (arma::uword i = 0; i<SIZE; i++) {
                // cast char to float, then carry out dot product
                C.row(i) = (arma::conv_to<arma::Row<float>>::from(A.row(i))*2-1)*B;
            }
                
            auto stop_2 = std::chrono::high_resolution_clock::now();
            printf("Armadillo9 [mix8_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
    }
    return 0;
}