#ifndef initialiser_h
#define initialiser_h

#include "data_type.h"
#include <cmath>
#include <random>

template <class T>
void _glorot_normal_initializer(T & mat, int seed=0) {
    std::default_random_engine generator;
    generator.seed(seed);
    
    std::vector<size_t> shape = mat.shape();
    
    float fan_out = 1.0;
    for (int i=0; i<shape.size()-1; i++) {
        fan_out *= float(shape[i]);
    }
    float stdv = 1.0/sqrtf(fan_out);
    //printf("stdv: %f ------------------------------------------\n", stdv);
    std::normal_distribution<double> distribution(0, stdv);
    
    int i;
    const int mat_size = mat.size();
    for (i = 0; i < mat_size; i++) {
        mat[i] = distribution(generator);
        
    }
};

template <typename T>
void glorot_normal_initializer(Matrix<T> & mat, int seed=0) {
    _glorot_normal_initializer<Matrix<T>>(mat, seed); 
};
template <typename T>
void glorot_normal_initializer(Matrix2D<T> & mat, int seed=0) { _glorot_normal_initializer<Matrix2D<T>>(mat, seed); };

template <class T>
void _glorot_uniform_initializer(T & mat, int seed=0) {
    // Function overloading
    srand(seed);
    std::vector<size_t> shape = mat.shape();
    int ndim = mat.ndim();
    
    int fan_out = 1;
    for (int i=1; i<ndim; i++) {
        fan_out *= shape[i];
    }
    
    
    float SCALE = RAND_MAX*sqrt(fan_out)/2;
    float MID_POINT = 1/sqrt(fan_out);
    
    int i;
    const int mat_size = mat.size();
    for (i = 0; i < mat_size; i++) {
        mat[i] = (float)rand()/SCALE-MID_POINT;
        
    }
};

template <typename T>
void glorot_uniform_initializer(Matrix<T> & mat, int seed=0) { _glorot_normal_initializer<Matrix<T>>(mat, seed); };
template <typename T>
void glorot_uniform_initializer(Matrix2D<T> & mat, int seed=0) { _glorot_normal_initializer<Matrix2D<T>>(mat, seed); };


template <class T>
void _ones_initializer(T & mat_2d, int seed=0) {
    // Function overloading
    int i;
    const int mat_2d_size = mat_2d.size();
    for (i = 0; i < mat_2d_size; i++) {
        mat_2d[i] = 1;
    }
};

template <typename T>
void ones_initializer(Matrix<T> & mat_2d, int seed=0) { _ones_initializer<Matrix<T>>(mat_2d, seed); };
template <typename T>
void ones_initializer(Matrix2D<T> & mat_2d, int seed=0) { _ones_initializer<Matrix2D<T>>(mat_2d, seed); };


template <class T>
void _ones_like(T & mat_2d) {
    // Function overloading
    int i;
    const int mat_2d_size = mat_2d.size();
    for (i = 0; i < mat_2d_size; i++) {
        mat_2d[i] = 1;
    }
};

template <typename T>
void ones_like(Matrix2D<T> & mat_2d) { _ones_like<Matrix2D<T>>(mat_2d); };

template <typename T>
void ones_like(Matrix<T> & mat_2d) { _ones_like<Matrix<T>>(mat_2d); };

/*
template <typename T>
void ones_initializer_2d(Matrix2D<T> & mat_2d, int seed=0) {
    // Function overloading
    int i;
    const int mat_2d_size = mat_2d.size;
    for (i = 0; i < mat_2d_size; i++) {
        mat_2d.data[i] = T(1);
    }
};

template <typename T>
void glorot_normal_initializer_2d(Matrix2D<T> & mat_2d, int seed=0) {
    // Function overloading
    srand(seed);
    float SCALE = RAND_MAX*sqrt(mat_2d.n_cols)/2;
    float MID_POINT = 1/sqrt(mat_2d.n_cols);
    
    int i;
    const int mat_2d_size = mat_2d.n_rows*mat_2d.n_cols;
    for (i = 0; i < mat_2d_size; i++) {
        mat_2d.data[i] = T((float)rand()/SCALE-MID_POINT);
    }
};
*/
#endif