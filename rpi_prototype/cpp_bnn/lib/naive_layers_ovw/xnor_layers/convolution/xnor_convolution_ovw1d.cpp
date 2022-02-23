#include "xnor_convolution_ovw.h"



XNor_Convolution2D_OWH1D::XNor_Convolution2D_OWH1D(size_t filters, size_t kernel_size, std::string padding, bool is_first_layer, int random_seed): filters(filters), kernel_size(kernel_size), padding(padding), is_first_layer(is_first_layer), random_seed(random_seed) {};

XNor_Convolution2D_OWH1D::~XNor_Convolution2D_OWH1D() {};


Matrix<float16_t>& XNor_Convolution2D_OWH1D::forward(Matrix<float16_t> & x, bool is_training) {
    x_shape = x.shape();
    std::vector<size_t> kernel_shape = {kernel_size, kernel_size, x_shape[3], filters};
    std::vector<size_t> output_shape = get_output_shape<size_t>(x_shape, kernel_shape, padding, 1);
    if (!is_built) {
        kernel.resize(kernel_shape);
        gradient_scale = 1/float(kernel.size());
        dkernel.resize(kernel_shape);
        //packed_x.resize({x_shape[3], x_shape[1], x_shape[2], x_shape[0]});
        if (not is_first_layer) {
            packed_x.resize(x_shape);
        }
        glorot_normal_initializer<float16_t>(kernel, random_seed);
        is_built = true;
    }
    if (not is_first_layer) {
        pack_bits(x, packed_x);
    }
    
    //pack_bits(x, packed_x, false);
    Matrix<bool> packed_kernel(kernel_shape);
    pack_bits(kernel, packed_kernel);
    
    size_t pad_height = get_number_of_pad(x_shape[1], output_shape[1], kernel_shape[0], strides);
    size_t pad_width = get_number_of_pad(x_shape[2], output_shape[2], kernel_shape[1], strides);
    
    _convolution_ovw<float16_t, bool>(x, packed_kernel, output_shape, {1,1}, {pad_height,pad_width}, 0);
    
    //return binary_conv_op(x, packed_kernel, padding);
    return x;
}



Matrix<float16_t>& XNor_Convolution2D_OWH1D::backprop(Matrix<float16_t> & dy) {
    const int8_t dy_bias = get_po2_bias<float16_t>(dy);
    scale_with_po2_bias<float16_t>(dy, dy_bias);
    log_quantize<float16_t>(dy, dy_bias, 4);
    
    unsigned long long int one = 1;
    clip<int8_t>(dy_bias, -63, 63);
    float pad_val = 1.0/float(one<<2*abs(dy_bias));
    
    // dw
    packed_x.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    
    std::vector<size_t> packed_x_shape = packed_x.shape();
    std::vector<size_t> dkernel_shape = dkernel.shape();
    std::vector<size_t> dy_shape = dy.shape();
    size_t pad_height = get_number_of_pad(packed_x_shape[1], dkernel_shape[1], dy_shape[0], strides);
    size_t pad_width = get_number_of_pad(packed_x_shape[2], dkernel_shape[2], dy_shape[1], strides);
    _convolution<bool, float16_t>(packed_x, dy, dkernel, {1, 1}, {pad_height,pad_width}, 0);
    
    dkernel.transpose(0, 1, 2, 3);
    packed_x.transpose(0, 1, 2, 3);
    dy.transpose(0, 1, 2, 3);

    //float N = float(kernel.size());
    //dkernel_calculation<float16_t>(dkernel, kernel, 1.0/sqrt(N));

    // dx
    Matrix<bool> packed_kernel(kernel.shape());
    pack_bits(kernel, packed_kernel);
    packed_kernel.transpose(0, 1, 3, 2);
    packed_kernel.reverse(true, true);
    
    std::vector<size_t> packed_kernel_shape = packed_kernel.shape();
    dy_shape = dy.shape();
    
    pad_height = get_number_of_pad(dy_shape[1], x_shape[1], packed_kernel_shape[0], strides);
    pad_width = get_number_of_pad(dy_shape[2], x_shape[2], packed_kernel_shape[1], strides);
    
    // clip so that bit shift will not cause unknown behaviour
    _convolution_ovw<float16_t, bool>(dy, packed_kernel, x_shape, {1,1}, {pad_height,pad_width}, pad_val);
    return dy;
}

// first layer
void XNor_Convolution2D_OWH1D::backprop(Matrix<float16_t> &dy, Matrix<float16_t> & original_input) {
    const int8_t dy_bias = get_po2_bias<float16_t>(dy);
    scale_with_po2_bias<float16_t>(dy, dy_bias);
    log_quantize<float16_t>(dy, dy_bias, 4);
    
    // dw
    original_input.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    //printf("original_input_shape: "); print_shape(original_input);
    //printf("dy: "); print_shape(dy);
    //printf("dkernel: "); print_shape(dkernel);
    
    std::vector<size_t> float_x_shape = original_input.shape();
    std::vector<size_t> dkernel_shape = dkernel.shape();
    std::vector<size_t> dy_shape = dy.shape();
    size_t pad_height = get_number_of_pad(float_x_shape[1], dkernel_shape[1], dy_shape[0], strides);
    size_t pad_width = get_number_of_pad(float_x_shape[2], dkernel_shape[2], dy_shape[1], strides);
    
    _convolution<float16_t, float16_t>(original_input, dy, dkernel, {1, 1}, {pad_height,pad_width}, 0);
    
    dkernel.transpose(0, 1, 2, 3);
    //dy.transpose(0, 1, 2, 3);
    //original_input.transpose(0, 1, 2, 3);

    //float N = float(kernel.size());
    //dkernel_calculation<float16_t>(dkernel, kernel, 1.0/sqrt(N));
    
    return;
}

// first layer
void XNor_Convolution2D_OWH1D::backprop(Matrix<PO2_5bits_t> &dy, Matrix<float16_t> & original_input) {
    const int8_t dy_bias = get_po2_bias<PO2_5bits_t>(dy);
    
    // dw
    original_input.transpose(3, 1, 2, 0);
    dy.transpose(1, 2, 0, 3);
    dkernel.transpose(2, 0, 1, 3);
    
    std::vector<size_t> float_x_shape = original_input.shape();
    std::vector<size_t> dkernel_shape = dkernel.shape();
    std::vector<size_t> dy_shape = dy.shape();
    size_t pad_height = get_number_of_pad(float_x_shape[1], dkernel_shape[1], dy_shape[0], strides);
    size_t pad_width = get_number_of_pad(float_x_shape[2], dkernel_shape[2], dy_shape[1], strides);
    
    _convolution<float16_t, PO2_5bits_t>(original_input, dy, dkernel, {1, 1}, {pad_height,pad_width}, 0);
    
    dkernel.transpose(0, 1, 2, 3);
    //dy.transpose(0, 1, 2, 3);
    //original_input.transpose(0, 1, 2, 3);

    //float N = float(kernel.size());
    //dkernel_calculation<float16_t>(dkernel, kernel, 1.0/sqrt(N));
    
    return;
}