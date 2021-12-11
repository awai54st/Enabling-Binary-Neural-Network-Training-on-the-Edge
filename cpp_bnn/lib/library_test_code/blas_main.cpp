#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <cstdint>
#include <iostream>


extern "C"
{
    #include <cblas.h>
}

void check_cblas() {
    float A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};         
    float B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};  
    float C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5}; 
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);
}

void init(int8_t* matrix, int row, int column) {
    for (int j = 0; j < column; j++){
        for (int i = 0; i < row; i++){
            matrix[j*row + i] = (int8_t)((float)rand())/RAND_MAX;
        }
    }
}
void init(float* matrix, int row, int column) {
    for (int j = 0; j < column; j++){
        for (int i = 0; i < row; i++){
            matrix[j*row + i] = ((float)rand())/RAND_MAX;
        }
    }
}
 
void print(const char * name, const float* matrix, int row, int column) {
    printf("Matrix %s has %d rows and %d columns:\n", name, row, column);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < column; j++){
            printf("%.3f ", matrix[j*row + i]);
        }
        printf("\n");
    }
    printf("\n");
}


enum string_code {
    float32_naive,
    int16,
    mix_8_32,
    float32
};

string_code hashit (std::string const& inString) {
    if (inString == "float32_naive") return float32_naive;
    if (inString == "int16") return int16;
    if (inString == "mix_8_32") return mix_8_32;
    return float32;
};

int main(int argc, char * argv[]) {
    int rowsA, colsB, common;
    int i,j,k;

    if (argc != 3){
        printf("Using defaults\n");
        rowsA = 400; colsB = 400; common = 400;
    } else {
        rowsA = atoi(argv[2]); colsB = atoi(argv[2]);common = atoi(argv[2]);
    }

    switch (hashit(argv[1])) {
        case float32_naive:
            {
                std::vector<float> A(rowsA * common); 
                std::vector<float> B(common * colsB);
                std::vector<float> D(rowsA * colsB);
                
                init(A.data(), rowsA, common); init(B.data(), common, colsB);
                auto start_2 = std::chrono::high_resolution_clock::now();
                for(i=0;i<colsB;i++){
                    for(j=0;j<rowsA;j++){
                        D[i*rowsA+j]=0;
                        for(k=0;k<common;k++){
                            D[i*rowsA+j]+=A[k*rowsA+j]*B[k+common*i];
                        }
                    }
                }
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("BLAS %d (use : %li us) \n", atoi(argv[2]), std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
                //print("A", A, rowsA, common); print("B", B, common, colsB);
                //print("C", C, rowsA, colsB); print("D", D, rowsA, colsB);
            }
            break;
        case int16:
            {
            }
            break;
        case mix_8_32:
            {
                std::vector<int8_t> A(rowsA * common, 1); 
                std::vector<float> B(common * colsB, 1);
                std::vector<float> C(rowsA * colsB);

                float one = 1.0, zero = 0.0;

                auto start_2 = std::chrono::high_resolution_clock::now();
                int compromisation = 2;
                for (int i = 0; i<rowsA; i+=compromisation ) {
                    // cast char to float, then carry out dot product
                    std::vector<float> A_tmp(A.begin(), A.begin()+rowsA*compromisation);
                    cblas_sgemm(
                        CblasColMajor,CblasNoTrans,CblasNoTrans, 
                        compromisation, colsB, common ,1.0,
                        A_tmp.data(), compromisation,
                        B.data(), common,
                        0.0, C.data()+rowsA, compromisation);
                }
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("BLAS [mix8_32] size: %d (use : %li us) \n", atoi(argv[2]), std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
                std::cout << C[1] << "output\n";
            }
            break;
        case float32:
            {
                std::vector<float> A(rowsA * common); 
                std::vector<float> B(common * colsB);
                std::vector<float> C(rowsA * colsB);
                enum CBLAS_ORDER order = CblasColMajor;
                enum CBLAS_TRANSPOSE transA = CblasNoTrans;
                enum CBLAS_TRANSPOSE transB = CblasNoTrans;

                float one = 1.0, zero = 0.0;
                init(A.data(), rowsA, common); init(B.data(), common, colsB);

                auto start_2 = std::chrono::high_resolution_clock::now();
                cblas_sgemm(order,transA,transB, rowsA, colsB, common ,1.0,A.data(), 
                           rowsA ,B.data(), common ,0.0,C.data(), rowsA);
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("BLAS [float32] size: %d (use : %li us) \n", atoi(argv[2]), std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
    }

    return 0;
}