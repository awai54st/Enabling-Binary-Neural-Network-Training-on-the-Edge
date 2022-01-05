#include "data_types.h"
#include "initializers.h"
#include <random>
//#include <omp.h>

VECTOR_2D_T<float> glorot_normal_initializer_2d(int c_in, int c_out, int seed) {
    // Function overloading
    srand(seed);
    //float FLOAT_MAX = 2/sqrt(c_in);
    //float SCALE = RAND_MAX/(FLOAT_MAX);
    float SCALE = RAND_MAX*sqrt(c_in)/2;
    float MID_POINT = 1/sqrt(c_in);
    
    VECTOR_2D_T<float> random_arr(c_in, VECTOR_1D_T<float>(c_out));
    
    int i, j;
    for (i = 0; i < c_in; i++) {
        for (j = 0; j < c_out; j++) {
            random_arr[i][j] = (float) rand()/SCALE-MID_POINT;
        }
    }
    
    
    /*
    int i, j;
    #pragma omp parallel for private(i, j) num_threads(4)
    for (i = 0; i < c_in; i++) {
        for (j = 0; j < c_out; j++) {
            random_arr[i][j] = (float) rand()/SCALE-MID_POINT;
        }
    }
    
    for (VECTOR_1D_T<float> &i : random_arr) {
        for (float &j : i) {
            j = ((float) rand())/SCALE-MID_POINT;
        }
    }
    */
    return random_arr;
};

/*
VECTOR_4d glorot_normal_initializer_4d(int w, int h, int c_in, int c_out, int seed=0) {
    // Function overloading
    srand(seed);
    float FLOAT_MAX = 2/sqrt(w * h * c_in);
    float MID_POINT = FLOAT_MAX/2;
    
    VECTOR_4d random_arr(
        w, VECTOR_3d(
            h, VECTOR_2d(
                c_in, VECTOR_1d(c_out)
            )
        )
    );
    
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < c_in; ++k) {
                for (int l = 0; l < c_out; ++l) {
                    random_arr[i][j][k][l] = (float) rand()/RAND_MAX-0.5;
                }
            }
        }
    }
    return random_arr;
};
*/




/*
float abs_max_2d(const VECTOR_2d x) {
    float max_val = 0;
    for(auto i : x) {
        for (auto j:i) {
            if (j >= 0) {
                max_val = std::max(max_val, j);
            } else {
                max_val = std::max(max_val, -j);
            }
        }
    } 
    
    return max_val;
};

abs_max_2d(x)

VECTOR_2d gradient_quatisation_preprocessing(VECTOR_2d x, int bias) {
    float max_val = 0;
    for(int i=0; i<x.size(); i++) {
        for (int j=0; j<x.size(); j++) {
            x[i][j] = x[i][j] * (float)(1<<bias);
        }
    }
    
    return x;
};

gradient_quatisation_preprocessing(x, 4)

VECTOR_2d gradient_preprocessing(VECTOR_2d x, int width) {
    float max_value = (1<<(width-1));
    int shift = 1;
    
    
    float max_val = 0;
    for(int i=0; i<x.size(); i++) {
        for (int j=0; j<x.size(); j++) {
            if (x[i][j] >= 0) {
                x[i][j] = std::min(
                    max_value, std::max(
                        -max_value, round(log2(x[i][j]))
                    )
                );
            } else {
                x[i][j] = std::min(
                    max_value, std::max(
                        -max_value, round(-log2(x[i][j]))
                    )
                );
            }
        }
    }
    
    return x;
};

gradient_preprocessing(x, 4)

VECTOR_2d gradient_quatisation_postprocessing(VECTOR_2d x, int bias) {
    float max_val = 0;
    for(int i=0; i<x.size(); i++) {
        for (int j=0; j<x.size(); j++) {
            x[i][j] = x[i][j] * (float)(1<<bias);
        }
    }
    
    return x;
};

gradient_quatisation_postprocessing(x, 4)

union unholy_t {
    float f;
    int i;
};
unholy_t unholy;
unholy.f=4.0; // put in a float
unholy.i

unholy_t unholy;
unholy.f=-2.0;
std::cout << unholy.f ;
unholy.i = unholy.i & 0x7fFFffFF;
unholy.f
*/