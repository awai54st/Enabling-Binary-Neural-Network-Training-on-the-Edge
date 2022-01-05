#include <iostream>
#include <chrono>
#include "xnor_dense_ovw.h"


XNor_Dense_OVW::XNor_Dense_OVW(size_t units, bool is_first_layer, int random_seed): 
    units(units), is_first_layer(is_first_layer), random_seed(random_seed){};

XNor_Dense_OVW::~XNor_Dense_OVW() {};

Matrix2D<float16_t> XNor_Dense_OVW::forward(Matrix2D<float16_t> &x, bool is_training) {
    std::vector<size_t> x_shape = x.shape();
    if (!is_built) {
        if (not is_first_layer) {
            packed_x.resize(x_shape);
        }
        kernel.resize({x_shape[1], units}, 0);
        dkernel.resize({x_shape[1], units}, 0);
        glorot_normal_initializer<float16_t>(kernel, random_seed);
        is_built = true;
    }
    // Matrix<bool> bit_packed_kernel(x.n_cols, units, 1);
    Matrix2D<bool> bit_packed_kernel({x_shape[1], units});
    pack_bits<float16_t>(kernel, bit_packed_kernel);
    if (is_first_layer) {
        //float_x.resize(x_shape);
        float_x = x;
    } else {
        pack_bits<float>(x, packed_x, false);
    }
    
    float_dot_bit_ovw(x, bit_packed_kernel);
    return x;
};



template <typename T>
void dkernel_calculation(Matrix2D<T>& dkernel, Matrix2D<T>& kernel, const float scale) {
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    const std::size_t size = dkernel.size();
    
    std::size_t i;
    for (i=0; i<size; i++) {
        if (dkernel[i] >= 0) {
            dkernel[i] = 1*scale;
        } else {
            dkernel[i] = -1*scale;
        }
    }
    
    for (i=0; i<size; i++) {
        if (fabs(kernel[i]) >= 1) {
            dkernel[i] = 0;
        }
    }
};

//Matrix<float> backprop(Matrix<float> dy) {};
Matrix2D<float16_t> XNor_Dense_OVW::backprop(Matrix2D<float16_t> &dy) {
    // PO2 dy
    const int8_t dy_bias = get_po2_bias<float, Matrix2D>(dy);
    scale_with_po2_bias<float, Matrix2D>(dy, dy_bias);
    log_quantize<float, Matrix2D>(dy, dy_bias, 4);
    
    if (not is_first_layer) {
        // dw
        //Matrix<bool> packed_x_t({packed_x_shape[1], packed_x_shape[0]});
        //transpose<bool>(packed_x, packed_x_t);
        packed_x.transpose(1, 0);
        float N = (float) (kernel.size());
        //std::cout << "Size of N: " << N << "\n";

        bit_dot_float<float16_t, Matrix2D>(packed_x, dy, dkernel);
        dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
        packed_x.transpose(0, 1);
        
        // dx
        std::vector<size_t> packed_x_shape = packed_x.shape();
        Matrix2D<bool> bit_packed_kernel({packed_x_shape[1], units});
        pack_bits<float16_t>(kernel, bit_packed_kernel, false);
        bit_packed_kernel.transpose(1,0);
        float_dot_bit_ovw(dy, bit_packed_kernel);
        return dy;
    }
    // dw //Since it is first layer, don't calculate
    float N = (float) (kernel.size());
    //std::cout << "Size of N: " << N << "\n";
    
    //Matrix<float> float_x_t(float_x.n_cols, float_x.n_rows, 1);
    //transpose<float>(float_x, float_x_t);
    //float_dot_float(float_x_t, dy, dkernel);
    float_x.transpose(1,0);
    float_dot_float<float16_t, Matrix2D>(float_x, dy, dkernel);
    dkernel_calculation<float16_t>(dkernel, kernel, 1/sqrt(N));
    float_x.transpose(0,1);
    
    return dy;
}