#include <boost/dynamic_bitset.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <iostream>
#include <cstdint>
#include <chrono>

// boost float
void boost_matrix_float_1d(int SIZE) {
    boost::numeric::ublas::matrix<float> B1(SIZE, 1);
    //std::cout << "size of units: " << B1.size1() << "\n";
}

void boost_matrix_float_2d(int SIZE) {
    boost::numeric::ublas::matrix<float> B1(SIZE, SIZE);
    // std::cout << "size of units: " << B1.size1() << " , " << B1.size2() << "\n";
}
void boost_matmul_float_2d(int SIZE) {
    boost::numeric::ublas::matrix<float> a(SIZE, SIZE, 1);
    boost::numeric::ublas::matrix<float> b(SIZE, SIZE, 1);
        
    boost::numeric::ublas::matrix<float> tmp = boost::numeric::ublas::prod(a, b);
}

enum string_code {
    mix_1_32,
    int16,
    mix_8_32,
    float32
};

string_code hashit (std::string const& inString) {
    if (inString == "int16") return int16;
    if (inString == "mix_1_32") return mix_1_32;
    if (inString == "mix_8_32") return mix_8_32;
    return float32;
};

int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[2]); 
    switch (hashit(argv[1])) {
        case int16:
                {
                boost::numeric::ublas::matrix<short> a(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<short> b(SIZE, SIZE, 1);

                auto start_2 = std::chrono::high_resolution_clock::now();
                boost::numeric::ublas::matrix<short> tmp = boost::numeric::ublas::prod(a, b);
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Boost [int16] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case mix_1_32:
            {
                boost::numeric::ublas::matrix<bool> a(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<bool> b(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<char> c(SIZE, SIZE, 0);
                // initialize a to emulate random behaviour
                for (int i=0; i<a.size1(); i++) {
                    for (int j=0; j<a.size2(); j++) {
                        if (i%2 == 0) {
                            a(i, j) = 0;
                        }
                    }
                }

                auto start_2 = std::chrono::high_resolution_clock::now();
                
                
                for (int i=0; i<a.size1(); i++) {
                    for (int j=0; j<b.size2(); j++) {
                        float c_tmp = 0;
                        for (int common=0; common<a.size2(); common++) {
                            c_tmp += a(i, common) ? b(common,j):-b(common,j);
                        }
                        c(i, j) = c_tmp;
                    }
                }
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Boost [mix1_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
            
        case mix_8_32:
            {
                boost::numeric::ublas::matrix<short> a(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<float> b(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<float> c(SIZE, SIZE, 1);

                auto start_2 = std::chrono::high_resolution_clock::now();
                c = boost::numeric::ublas::prod(a, b);
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Boost [mix8_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32:
            {
                boost::numeric::ublas::matrix<float> a(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<float> b(SIZE, SIZE, 1);
                boost::numeric::ublas::matrix<float> c(SIZE, SIZE, 1);

                auto start_2 = std::chrono::high_resolution_clock::now();
                c = boost::numeric::ublas::prod(a, b);
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Boost [float32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
    }
    return 0;
}