#include <iostream>
#include <chrono>
#include "vanilla_dense_ovw.h"

        
Vanilla_Dense_OVW::Vanilla_Dense_OVW(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

Vanilla_Dense_OVW::~Vanilla_Dense_OVW() {};

Matrix2D<float> Vanilla_Dense_OVW::forward(Matrix2D<float> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        float_x.resize(x_shape);
        kernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float>(kernel, random_seed);
        is_built = true;
    }
    if (not is_first_layer) {
        float_x = x;
    }
    Matrix2D<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel);
    
    float_dot_bit_ovw(x, bit_packed_kernel);
    return x;
};


//Matrix<float> backprop(Matrix<float> dy) {};
Matrix2D<float> Vanilla_Dense_OVW::backprop(Matrix2D<float> &dy) {
    // dw
    dkernel.resize(kernel.shape(), 0);
    float_x.transpose(1,0);
    float_dot_float<float, Matrix2D>(float_x, dy, dkernel);
    float_x.transpose(0,1);
    size_t kernel_size = kernel.size();
    
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel[i] >= 1) {
            dkernel[i] = 0;
        }
    }
    
    std::vector<size_t> float_x_shape = float_x.shape();

    // dx
    Matrix2D<bool> bit_packed_kernel({float_x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel, false);
    bit_packed_kernel.transpose(1,0);
    float_dot_bit_ovw(dy, bit_packed_kernel);
    
    return dy;
}


// first
void Vanilla_Dense_OVW::backprop(Matrix2D<float> &dy, Matrix2D<float> & original_input) {
    // dw
    dkernel.resize(kernel.shape(), 0);
    original_input.transpose(1,0);
    float_dot_float<float, Matrix2D>(original_input, dy, dkernel);
    original_input.transpose(0,1);
    size_t kernel_size = kernel.size();
    
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel[i] >= 1) {
            dkernel[i] = 0;
        }
    }
    
    return;
}