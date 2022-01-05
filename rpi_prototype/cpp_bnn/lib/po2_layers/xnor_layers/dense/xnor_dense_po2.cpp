#include <iostream>
#include <chrono>
#include "xnor_dense_po2.h"


XNor_Dense_PO2::XNor_Dense_PO2(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

XNor_Dense_PO2::~XNor_Dense_PO2() {};

Matrix<float16_t> &XNor_Dense_PO2::forward(Matrix<float16_t> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        if (not is_first_layer) {
            packed_x.resize(x_shape);
        }
        kernel.resize({x_shape[1], units}, 0);
        gradient_scale = 1/float(kernel.size());
        glorot_normal_initializer<float16_t>(kernel, random_seed);
        is_built = true;
    }
    // Matrix<bool> bit_packed_kernel(x.n_cols, units, 1);
    Matrix<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float16_t>(kernel, bit_packed_kernel);
    if (not is_first_layer) {
        pack_bits<float16_t>(x, packed_x);
    }
    
    float_dot_bit(x, bit_packed_kernel);
    return x;
};


// not first layer
Matrix<float16_t> &XNor_Dense_PO2::backprop(Matrix<float16_t> &dy) {
    // PO2 dy
    dkernel.resize(kernel.shape(), 0);
    // const int8_t dy_bias = get_po2_bias<float16_t, Matrix>(dy);
    Matrix<PO2_5bits_t>dy_5_bits(dy.shape()); 
    PO2_gradients(dy, dy_5_bits);
    
    // dw
    //Matrix<bool> packed_x_t({packed_x_shape[1], packed_x_shape[0]});
    //transpose<bool>(packed_x, packed_x_t);
    packed_x.transpose(1, 0);
    //float N = float(kernel.size());
    //std::cout << "Size of N: " << N << "\n";

    dot<bool, PO2_5bits_t, bool>(packed_x, dy_5_bits, dkernel);
    //dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    packed_x.transpose(0, 1);

    // dx
    Matrix<bool> bit_packed_kernel(kernel.shape());
    pack_bits<float16_t>(kernel, bit_packed_kernel);
    bit_packed_kernel.transpose(1,0);
    dy.resize({packed_x.shape()[0], kernel.shape()[0]});
    //printf("After transpose: "); print_shape(bit_packed_kernel);
    dot<PO2_5bits_t, bool, float16_t>(dy_5_bits, bit_packed_kernel, dy);
    return dy;
}

// first layer
void XNor_Dense_PO2::backprop(Matrix<float16_t> &dy, Matrix<float16_t> & original_input) {
    // PO2 dy
    dkernel.resize(kernel.shape(), 0);
    
    //const int8_t dy_bias = get_po2_bias<float16_t, Matrix>(dy);
    Matrix<PO2_5bits_t>dy_5_bits(dy.shape()); 
    PO2_gradients(dy, dy_5_bits);
    
    //float N = float(kernel.size());
    
    original_input.transpose(1,0);
    dot<float16_t, PO2_5bits_t, bool>(original_input, dy_5_bits, dkernel);
    //dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    original_input.transpose(0,1);
    
    return;
}