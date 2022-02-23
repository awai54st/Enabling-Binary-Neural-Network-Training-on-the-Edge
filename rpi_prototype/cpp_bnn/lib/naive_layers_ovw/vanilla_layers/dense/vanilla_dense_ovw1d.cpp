#include <iostream>
#include <chrono>
#include "vanilla_dense_ovw.h"
#include <cmath>
        
Vanilla_Dense_OVW1D::Vanilla_Dense_OVW1D(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

Vanilla_Dense_OVW1D::~Vanilla_Dense_OVW1D() {};

Matrix<float>& Vanilla_Dense_OVW1D::forward(Matrix<float> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        //float_x.resize(x_shape);
        kernel.resize({x_shape[1], units}, 0);
        dkernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float>(kernel, random_seed);
        // dkernel.resize({x_shape[1], units}, 0);
        is_built = true;
    }
    if (not is_first_layer) {
        float_x = x;
    }
    Matrix<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel);
    
    float_dot_bit(x, bit_packed_kernel);
    return x;
};


// not first layer
Matrix<float>& Vanilla_Dense_OVW1D::backprop(Matrix<float> &dy) {
    std::vector<size_t> float_x_shape = float_x.shape();
    //if (!is_built[1]) {
    //    dkernel.resize({float_x_shape[1], units}, 0);
    //    is_built[1] = true;
    //}
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
    
        
    // dx
    //printf("Input: "); print_mat(dy);
    Matrix<bool> bit_packed_kernel({float_x_shape[1], units});
    pack_bits<float>(kernel, bit_packed_kernel, false);
    bit_packed_kernel.transpose(1,0);
    float_dot_bit(dy, bit_packed_kernel);
    //printf("Kernel: "); print_mat(kernel);
    //printf("Kernel bool: "); print_mat_bool(bit_packed_kernel);
    
    return dy;
}


// first layer
void Vanilla_Dense_OVW1D::backprop(Matrix<float> &dy, Matrix<float> & original_input) {
    dkernel.resize({original_input.shape()[1], units}, 0);
    original_input.transpose(1,0);
    dot<float, float, float>(original_input, dy, dkernel);
    original_input.transpose(0,1);
    
    //size_t kernel_size = kernel.size();
    //for (size_t i=0; i<kernel_size; i++) {
    //    if (fabs(kernel[i]) >= 1) {
    //        dkernel[i] = 0;
    //    }
    //}
    
    return;
}