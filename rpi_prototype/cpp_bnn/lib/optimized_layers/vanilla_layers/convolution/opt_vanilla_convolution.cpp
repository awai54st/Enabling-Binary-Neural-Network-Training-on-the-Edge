#include "opt_vanilla_convolution.h"


OPT_Vanilla_Convolution2D::OPT_Vanilla_Convolution2D(size_t filters, size_t kernel_size, std::string padding, bool is_first_layer, int random_seed): filters(filters), kernel_size(kernel_size), padding(padding), is_first_layer(is_first_layer), random_seed(random_seed) {};

OPT_Vanilla_Convolution2D::~OPT_Vanilla_Convolution2D() {};


Matrix<float> OPT_Vanilla_Convolution2D::forward(Matrix<float> & x, bool is_training) {
    x_shape = x.shape();
    std::vector<size_t> kernel_shape = {kernel_size, kernel_size, x_shape[3], filters};
    if (!is_built) {
        kernel.resize(kernel_shape);
        dkernel.resize(kernel_shape);
        float_x.resize(x_shape);
        glorot_normal_initializer<float>(kernel, random_seed);
        is_built = true;
    }
    if (not is_first_layer) {
        float_x = x;
    }
    Matrix<bool> packed_kernel(kernel_shape);
    pack_bits(kernel, packed_kernel, false);
    
    // Calculate the output shape based on input shape, kernel shape, stride and padding
    std::vector<size_t> output_shape = get_output_shape<size_t>(x_shape, kernel_shape, padding, stride);
    Matrix<float> y(output_shape, 0);
    
    //convolution<float, bool, float>(x, packed_kernel, y);
    convolution<float, bool>(x, packed_kernel, output_shape);
    return x;
}


Matrix<float> OPT_Vanilla_Convolution2D::backprop(Matrix<float> & dy) {
    
    // dw
    float_x.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    //float_conv_float<float>(float_x, dy, dkernel);
    //dkernel = float_conv_op<float>(float_x, dy, dkernel.shape());
    //dkernel = float_conv_op<float>(float_x, dy, "valid");
    convolution<float, float, float>(float_x, dy, dkernel);
    dkernel.transpose(0, 1, 2, 3);
    
    // if abs(kernel) value >= 1, turn off its gradient
    size_t _size = kernel.size();
    for (size_t i=0; i<_size; i++) {
        if (fabs(kernel.m_data[i]) >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
    float_x.transpose(0, 1, 2, 3);
    
    // dx
    //Matrix<float> dx(x_shape);
    dy.transpose(0, 1, 2, 3);
    Matrix<bool> packed_kernel(kernel.shape());
    pack_bits(kernel, packed_kernel, false);
    packed_kernel.transpose(0, 1, 3, 2);
    packed_kernel.reverse(true, true);
    //convolution<float, bool, float>(dy, packed_kernel, dx);
    convolution<float, bool>(dy, packed_kernel, x_shape);
    
    return dy;
}


void OPT_Vanilla_Convolution2D::backprop(Matrix<float> & dy, Matrix<float> & original_input) {
    
    // dw
    original_input.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    convolution<float, float, float>(original_input, dy, dkernel);
    dkernel.transpose(0, 1, 2, 3);
    
    // if abs(kernel) value >= 1, turn off its gradient
    size_t _size = kernel.size();
    for (size_t i=0; i<_size; i++) {
        if (fabs(kernel.m_data[i]) >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
    original_input.transpose(0, 1, 2, 3);
    
    
    return ;
}