#include <iostream>
#include <chrono>
#include <cmath>
#include "xnor_dense.h"


XNor_Dense::XNor_Dense(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

XNor_Dense::~XNor_Dense() {};

Matrix<float16_t> XNor_Dense::forward(Matrix<float16_t> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        if (not is_first_layer) {
            packed_x.resize(x_shape);
        }
        kernel.resize({x_shape[1], units}, 0);
        dkernel.resize({x_shape[1], units}, 0);
        gradient_scale = 1/float(kernel.size());
        glorot_normal_initializer<float16_t>(kernel, random_seed);
        is_built = true;
    }
    // Matrix<bool> bit_packed_kernel(x.n_cols, units, 1);
    Matrix<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float16_t>(kernel, bit_packed_kernel);
    if (not is_first_layer) {
        pack_bits<float16_t>(x, packed_x, false);
    }
    
    Matrix<float16_t> y({x_shape[0], units});
    dot<float16_t, bool, float16_t>(x, bit_packed_kernel, y);
    return y;
};


/*
Matrix<float> bit_dot_bit(const Matrix<bool> & x, const Matrix<bool> & kernel) {
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> kernel_shape = kernel.shape();
    Matrix<float> y({x_shape[0], kernel_shape[1]});
    
    for (int i=0; i<x_shape[0]; i++) {
        for (int j=0; j<kernel[1]; j++) {
            short tmp_y = 0;
            for (int common=0; common<x_shape[1]; common++) {
                if (x(i, common) == kernel(common, j)) {
                    tmp_y += 1.0;
                } else {
                    tmp_y -= 1.0;
                }
            }
            y.m_data[(i*kernel_shape[0])+j] = (float) tmp_y;
        }
    }
    return y;
}

Matrix<float> DenseNotFirst::forward(const Matrix<bool> &x, bool is_training) {
    if (!is_built) {
        kernel.resize(x.n_cols, units, 0);
        dkernel.resize(x.n_cols, units, 0);
        glorot_normal_initializer<float16_t>(kernel, random_seed);
        is_built = true;
    }
    // Matrix<bool> bit_packed_kernel(x.n_cols, units, 1);
    Matrix<bool> bit_packed_kernel(units, x.n_cols, 1);
    pack_bits<float16_t>(kernel, bit_packed_kernel, true);
    packed_x = x;
    return bit_dot_bit(packed_x, bit_packed_kernel);
};
*/


//Matrix<float> backprop(Matrix<float> dy) {};
Matrix<float16_t> XNor_Dense::backprop(Matrix<float16_t> &dy) {
    // PO2 dy
    const int8_t dy_bias = get_po2_bias<float16_t>(dy);
    //printf("dy_bias: %d", dy_bias);
    scale_with_po2_bias<float16_t>(dy, dy_bias);
    log_quantize<float16_t>(dy, dy_bias, 4);
    //print_mat(dy);
    
    // dx
    std::vector<size_t> packed_x_shape = packed_x.shape();
    Matrix<bool> bit_packed_kernel({packed_x_shape[1], units});
    Matrix<float16_t> dx(packed_x_shape);
    pack_bits<float16_t>(kernel, bit_packed_kernel, false);
    bit_packed_kernel.transpose(1,0);
    dot<float16_t, bool, float16_t>(dy, bit_packed_kernel, dx);

    // dw
    //Matrix<bool> packed_x_t({packed_x_shape[1], packed_x_shape[0]});
    //transpose<bool>(packed_x, packed_x_t);
    packed_x.transpose(1, 0);
    //std::cout << "Size of N: " << N << "\n";

    dot<bool, float16_t, bool>(packed_x, dy, dkernel);
    //float N = float(kernel.size());
    //dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    packed_x.transpose(0, 1);
    return dx;
}

void XNor_Dense::backprop(Matrix<float16_t> & dy, Matrix<float16_t> & original_input) {
    const int8_t dy_bias = get_po2_bias<float16_t, Matrix>(original_input);
    scale_with_po2_bias<float16_t, Matrix>(original_input, dy_bias);
    log_quantize<float16_t, Matrix>(original_input, dy_bias, 4);
    
    // dw //Since it is first layer, don't calculate
    //float N = float(kernel.size());
    //std::cout << "Size of N: " << N << "\n";
    
    //Matrix<float> float_x_t(float_x.n_cols, float_x.n_rows, 1);
    //transpose<float>(float_x, float_x_t);
    //float_dot_float(float_x_t, dy, dkernel);
    original_input.transpose(1,0);
    dot<float16_t, float16_t, bool>(original_input, dy, dkernel);
    //dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    original_input.transpose(0,1);
    
    return;
};

void XNor_Dense::backprop(Matrix<PO2_5bits_t> & dy, Matrix<float16_t> & original_input) {
    original_input.transpose(1,0);
    dot<float16_t, PO2_5bits_t, bool>(original_input, dy, dkernel);
    original_input.transpose(0,1);
    
    return;
};