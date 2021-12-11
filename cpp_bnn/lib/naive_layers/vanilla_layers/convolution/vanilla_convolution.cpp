#include "vanilla_convolution.h"


Vanilla_Convolution2D::Vanilla_Convolution2D(size_t filters, size_t kernel_size, std::string padding, bool is_first_layer, int random_seed): filters(filters), kernel_size(kernel_size), padding(padding), is_first_layer(is_first_layer), random_seed(random_seed) {};

Vanilla_Convolution2D::~Vanilla_Convolution2D() {};


Matrix<float> Vanilla_Convolution2D::forward(Matrix<float> & x, bool is_training) {
    x_shape = x.shape();
    std::vector<size_t> kernel_shape = {kernel_size, kernel_size, x_shape[3], filters};
    std::vector<size_t> output_shape = get_output_shape<size_t>(x_shape, kernel_shape, padding, 1);
    if (!is_built) {
        kernel.resize(kernel_shape);
        dkernel.resize(kernel_shape);
        // float_x.resize(x_shape);
        glorot_normal_initializer<float>(kernel, random_seed);
        is_built = true;
    }
    if (not is_first_layer) {
        float_x = x;
    }
    Matrix<bool> packed_kernel(kernel_shape);
    pack_bits(kernel, packed_kernel);
    
    Matrix<float> y(output_shape, 0);
    size_t pad_height = get_number_of_pad(x_shape[1], output_shape[1], kernel_shape[0], strides);
    size_t pad_width = get_number_of_pad(x_shape[2], output_shape[2], kernel_shape[1], strides);
    _convolution<float, bool>(x, packed_kernel, y, {1,1}, {pad_height,pad_width}, 0);
    
    return y;
}


Matrix<float> Vanilla_Convolution2D::backprop(Matrix<float> & dy) {
    // dw
    float_x.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    
    std::vector<size_t> float_x_shape = float_x.shape();
    std::vector<size_t> dkernel_shape = dkernel.shape();
    std::vector<size_t> dy_shape = dy.shape();
    size_t pad_height = get_number_of_pad(float_x_shape[1], dkernel_shape[1], dy_shape[0], strides);
    size_t pad_width = get_number_of_pad(float_x_shape[2], dkernel_shape[2], dy_shape[1], strides);
    
    _convolution<float, float>(float_x, dy, dkernel, {1, 1}, {pad_height,pad_width}, 0);
    dkernel.transpose(0, 1, 2, 3);
    float_x.transpose(0, 1, 2, 3);
    dy.transpose(0, 1, 2, 3);
    
    // if abs(kernel) value >= 1, turn off its gradient
    size_t _size = kernel.size();
    for (size_t i=0; i<_size; i++) {
        if (fabs(kernel[i]) >= 1) {
            dkernel[i] = 0;
        }
    }
    
    // dx
    Matrix<float> dx(x_shape);
    Matrix<bool> packed_kernel(kernel.shape());
    pack_bits(kernel, packed_kernel);
    packed_kernel.transpose(0, 1, 3, 2);
    packed_kernel.reverse(true, true);
    
    std::vector<size_t> packed_kernel_shape = packed_kernel.shape();
    dy_shape = dy.shape();
    pad_height = get_number_of_pad(dy_shape[1], x_shape[1], packed_kernel_shape[0], strides);
    pad_width = get_number_of_pad(dy_shape[2], x_shape[2], packed_kernel_shape[1], strides);
    _convolution<float, bool>(dy, packed_kernel, dx, {1, 1}, {pad_height,pad_width}, 0);
    
    return dx;
}


void Vanilla_Convolution2D::backprop(Matrix<float> & dy, Matrix<float> & original_input) {
    // dw
    original_input.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    
    std::vector<size_t> float_x_shape = original_input.shape();
    std::vector<size_t> dkernel_shape = dkernel.shape();
    std::vector<size_t> dy_shape = dy.shape();
    size_t pad_height = get_number_of_pad(float_x_shape[1], dkernel_shape[1], dy_shape[0], strides);
    size_t pad_width = get_number_of_pad(float_x_shape[2], dkernel_shape[2], dy_shape[1], strides);
    
    _convolution<float, float>(
        original_input, dy, dkernel, {1, 1}, {pad_height,pad_width}, 0);
    dkernel.transpose(0, 1, 2, 3);
    //original_input.transpose(0, 1, 2, 3);
    dy.transpose(0, 1, 2, 3);
    
    // if abs(kernel) value >= 1, turn off its gradient
    //size_t _size = kernel.size();
    //for (size_t i=0; i<_size; i++) {
    //    if (fabs(kernel[i]) >= 1) {
    //        dkernel[i] = 0;
    //    }
    //}
    return;
}