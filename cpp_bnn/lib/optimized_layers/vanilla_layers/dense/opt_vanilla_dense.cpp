#include <iostream>
#include <chrono>
#include "opt_vanilla_dense.h"

        
OPT_Vanilla_Dense::OPT_Vanilla_Dense(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

OPT_Vanilla_Dense::~OPT_Vanilla_Dense() {};



Matrix<float> OPT_Vanilla_Dense::forward(Matrix<float> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        float_x.resize(x_shape);
        kernel.resize({x_shape[1], units}, 0);
        dkernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float>(kernel, random_seed);
        is_built = true;
    }
    // store x for backpropagation
    if (not is_first_layer) {
        float_x = x;
    }
    // pack bits
    Matrix<float> bit_packed_kernel({x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel);
    
    // forward operation
    //Matrix<float> y({x_shape[0], units});
    //matmul(x, bit_packed_kernel, y);
    dot_opt(x, bit_packed_kernel);
    return x;
};


// If it is first layer, don't calculate
Matrix<float> OPT_Vanilla_Dense::backprop(Matrix<float> &dy) {
    std::vector<size_t> float_x_shape = float_x.shape();
    // Matrix<float> dx(float_x_shape); 
    
    // dw
    //float_x.transpose(1,0);
    float_x.transpose(1, 0);
    dot_opt<float, float, float>(float_x, dy, dkernel);
    size_t kernel_size = kernel.size();
    float_x.transpose(0, 1);
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel.m_data[i] >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
        
    // dx
    Matrix<float> bit_packed_kernel({float_x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel, false);
    bit_packed_kernel.transpose(1,0);
    // bit_packed_kernel.transpose(1,0);
    dot_opt<float, float>(dy, bit_packed_kernel);
    
    
    return dy;
}


//Matrix<float> backprop(Matrix<float> dy) {};
void OPT_Vanilla_Dense::backprop(Matrix<float> &dy, Matrix<float> & original_input) {
    // dw
    //float_x.transpose(1,0);
    float_x.transpose(1, 0);
    dot_opt<float, float, float>(float_x, dy, dkernel);
    size_t kernel_size = kernel.size();
    float_x.transpose(0, 1);
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel.m_data[i] >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
    float_x.transpose(0, 1);
        
    return;
}

/*
Matrix<float> OPT_Vanilla_Dense::forward(Matrix<float> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        float_x.resize(x_shape);
        kernel.resize({x_shape[1], units}, 0);
        dkernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float>(kernel, random_seed);
        is_built = true;
    }
    // store x for backpropagation
    if (not is_first_layer) {
        float_x = x;
    }
    // pack bits
    Matrix<float> bit_packed_kernel({x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel);
    
    // forward operation
    Matrix<float> y({x_shape[0], units});
    matmul(x, bit_packed_kernel, y);
    return y;
};


// If it is first layer, don't calculate
Matrix<float> OPT_Vanilla_Dense::backprop(Matrix<float> &dy) {
    std::vector<size_t> float_x_shape = float_x.shape();
    Matrix<float> dx(float_x_shape); 
        
    // dx
    Matrix<float> bit_packed_kernel({float_x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel, false);
    // bit_packed_kernel.transpose(1,0);
    matmul_transb(dy, bit_packed_kernel, dx);
    
    // dw
    //float_x.transpose(1,0);
    matmul_transa<float>(float_x, dy, dkernel);
    size_t kernel_size = kernel.size();
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel.m_data[i] >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
    
    return dx;
}


//Matrix<float> backprop(Matrix<float> dy) {};
void OPT_Vanilla_Dense::backprop(Matrix<float> &dy, Matrix<float> & original_input) {
    // dw
    //float_x.transpose(1,0);
    matmul_transa<float>(original_input, dy, dkernel);
    size_t kernel_size = kernel.size();
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel.m_data[i] >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
    
    return;
}
*/