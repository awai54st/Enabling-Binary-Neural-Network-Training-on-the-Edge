#include <iostream>
#include <chrono>
#include "vanilla_dense.h"
#include <cmath>

        
Vanilla_BNNDense::Vanilla_BNNDense(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

Vanilla_BNNDense::~Vanilla_BNNDense() {};

Matrix<float> Vanilla_BNNDense::forward(Matrix<float> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        //float_x.resize(x_shape);
        kernel.resize({x_shape[1], units}, 0);
        dkernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float>(kernel, random_seed);
        is_built = true;
    }
    if (!is_first_layer) {
        float_x = x;
    }
    Matrix<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel);
    
    Matrix<float> y({x_shape[0], units});
    dot<float, bool, float>(x, bit_packed_kernel, y);
    return y;
};


//Matrix<float> backprop(Matrix<float> dy) {};
Matrix<float> Vanilla_BNNDense::backprop(Matrix<float> &dy) {
    std::vector<size_t> float_x_shape = float_x.shape();

    // dx
    Matrix<float> dx(float_x_shape);
    Matrix<bool> bit_packed_kernel({float_x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel, false);
    bit_packed_kernel.transpose(1,0);
    dot<float, bool, float>(dy, bit_packed_kernel, dx);
    // dw
    float_x.transpose(1,0);
    dot<float, float, float>(float_x, dy, dkernel);
    float_x.transpose(0,1);
    //size_t kernel_size = kernel.size();
    //for (size_t i=0; i<kernel_size; i++) {
    //    if (fabs(kernel[i]) >= 1) {
    //        dkernel[i] = 0;
    //    }
    //}
    
    return dx;
}

// first
void Vanilla_BNNDense::backprop(Matrix<float> & dy, Matrix<float> & original_input) {
    // dw
    original_input.transpose(1,0);
    dot<float, float, float>(original_input, dy, dkernel);
    original_input.transpose(0,1);
    size_t kernel_size = kernel.size();
    for (size_t i=0; i<kernel_size; i++) {
        if (fabs(kernel[i]) >= 1) {
            dkernel[i] = 0;
        }
    }
    
    return;
};