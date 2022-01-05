#include <iostream>
#include <chrono>
#include "opt_xnor_dense.h"


OPT_XNor_Dense::OPT_XNor_Dense(int units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

OPT_XNor_Dense::~OPT_XNor_Dense() {};

Matrix<float16_t> OPT_XNor_Dense::forward(Matrix<float16_t> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        if (not is_first_layer) {
            packed_x.resize(x_shape);
        }
        kernel.resize({x_shape[1], units}, 0);
        gradient_scale = 1/float(kernel.size());
        dkernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float16_t>(kernel, random_seed);
        is_built = true;
    }
    // Matrix<bool> bit_packed_kernel(x.n_cols, units, 1);
    Matrix<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float16_t>(kernel, bit_packed_kernel);
    
    if (not is_first_layer) {
        pack_bits<float16_t>(x, packed_x, false);
    }
    
    dot_opt(x, bit_packed_kernel);
    return x;
};


template <typename T>
void set_data(Matrix<T> from_x, Matrix<float>& to_x) {
    for (size_t i=0; i<to_x.size(); i++) {
        to_x.m_data[i] = float(from_x.m_data[i]);
    }
};


//Matrix<float> backprop(Matrix<float> dy) {};
Matrix<float16_t> OPT_XNor_Dense::backprop(Matrix<float16_t> &dy) {
    // PO2 dy
    //const int8_t dy_bias = get_po2_bias<float16_t>(dy);
    //scale_with_po2_bias<float16_t>(dy, dy_bias);
    //log_quantize<float16_t>(dy, dy_bias, 4);
    
    // dw
    packed_x.transpose(1, 0);
    dot_opt<bool, float16_t, bool>(packed_x, dy, dkernel);
    //dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    packed_x.transpose(0, 1);
    size_t kernel_size = kernel.size();
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel.m_data[i] >= 1) {
            dkernel.m_data[i] = 0;
        }
    }

    // dx
    Matrix<bool> bit_packed_kernel(kernel.shape());
    pack_bits<float16_t>(kernel, bit_packed_kernel);
    bit_packed_kernel.transpose(1,0);
    //printf("After transpose: "); print_shape(bit_packed_kernel);
    dot_opt<float16_t, bool>(dy, bit_packed_kernel);
    
    return dy;
}


//Matrix<float> backprop(Matrix<float> dy) {};
void OPT_XNor_Dense::backprop(Matrix<float16_t> &dy, Matrix<float16_t> & original_input) {
    // PO2 dy
    //const int8_t dy_bias = get_po2_bias<float16_t>(dy);
    //scale_with_po2_bias<float16_t>(dy, dy_bias);
    //log_quantize<float16_t>(dy, dy_bias, 4);
    
    
    original_input.transpose(1,0);

    dot_opt<float16_t, float16_t, bool>(original_input, dy, dkernel);
    //dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    original_input.transpose(0,1);
    size_t kernel_size = kernel.size();
    for (size_t i=0; i<kernel_size; i++) {
        if (kernel.m_data[i] >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
    
    return;
}