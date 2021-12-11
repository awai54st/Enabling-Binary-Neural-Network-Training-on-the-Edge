#include <math.h>
#include <numeric>
#include "data_types.h"
#include "dense.h"
#include <chrono>


VECTOR_2D_FLOAT float_dot_bit(const VECTOR_2D_FLOAT &x, const VECTOR_2D_T<bool> &y){
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    
    const std::size_t m = x.size(), n = x[0].size(), c = y.size();
    VECTOR_2D_FLOAT out(m, VECTOR_1D_FLOAT(n));
    
    std::size_t i, j, i_bit, bit_quotient;
    for (i=0; i<m; i++) {
        for (j=0; j<c; j++) {
            for (i_bit=0; i_bit<n; i_bit++) {
                out[i][j] += (y[j][i_bit]) ? x[i][i_bit]: -x[i][i_bit];
            }
        }
    }
    return out;
};


VECTOR_2D_FLOAT bit_dot_float(const VECTOR_2D_T<bool>& x, const VECTOR_2D_FLOAT& y, const float scale=1.0) {
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    
    const std::size_t m = x.size(), n = x[0].size(), c = y.size();
    VECTOR_2D_FLOAT out(m, VECTOR_1D_FLOAT(y[0].size()));
    
    std::size_t i, j, i_bit, bit_quotient;
    for (i=0; i<m; i++) {
        for (i_bit=0; i_bit<c; i_bit++) {
            for (j=0; j<y[0].size(); j++) {
                out[i][j] += (x[i][i_bit]) ? y[i_bit][j]: -y[i_bit][j];
            }
            out[i][j] *= scale;
        }
    }
    return out;
};


VECTOR_2D_FLOAT bit_dot_bit(const VECTOR_2D_T<bool>& x, const VECTOR_2D_T<bool>& y) {
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    const std::size_t m = x.size(), n = x[0].size(), c = y.size();
    VECTOR_2D_FLOAT out(m, VECTOR_1D_FLOAT(y[0].size()));
    
    const auto XNor = [](bool x, bool y) -> float { return x==y? 1 : -1; };
    std::size_t i, j;
    for (i=0; i<m; i++) {
        for (j=0; j<c; j++) {
            out[i][j] = std::inner_product(
                x[i].begin(), x[i].end(),
                y[j].begin(), 0.0,
                std::plus<float>(), XNor
            );
        }
    }
    return out;
};


// http://www.math.uaa.alaska.edu/~afkjm/csce211/handouts/SeparateCompilation.pdf
// Constructor
DenseNotFirst_CPP::DenseNotFirst_CPP(int units, int random_seed, bool is_training): 
units(units), random_seed(random_seed), is_built(false), is_training(is_training) {};

//deconstructor
DenseNotFirst_CPP::~DenseNotFirst_CPP() {};

VECTOR_2D_T<float> DenseNotFirst_CPP::forward(const VECTOR_2D_T<float> x) {
    // For efficient dot product:
    //     kernel is transposed
    //     x is not transposed
    VECTOR_2D_T<bool> bit_packed_kernel;
    bit_packed_x = pack_bits(x, false);
    
    if (!is_built) {
        kernel = glorot_normal_initializer_2d(x[0].size(), units, random_seed);
        bit_packed_kernel = pack_bits(kernel, true);
        is_built = true;
    }

    return bit_dot_bit(bit_packed_x, bit_packed_kernel);
}
    
VECTOR_2D_T<float> DenseNotFirst_CPP::backprop(VECTOR_2D_T<float> dy) {
    VECTOR_2D_T<float> dx;


    const float dy_bias = -round(log2(abs_max_2d(dy) ) ) + 8;
    dy = pre_gradient_quatisation(dy, dy_bias);
    dy = gradient_quantisation(dy, 4, dy_bias);


    // dx
    VECTOR_2D_T<bool> bit_packed_kernel = pack_bits(kernel, false);
    //std::cout << "max bit: " << bit_packed_x.MAX_BIT << "\n";
    dx = float_dot_bit(dy, bit_packed_kernel);

    // dw
    auto start = std::chrono::high_resolution_clock::now();
    VECTOR_2D_T<bool> x_t = transpose(bit_packed_x);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("transpose (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
    float N = (float) (x_t.size()*units);
    //std::cout << "Size of N: " << N << "\n";
    start = std::chrono::high_resolution_clock::now();
    dkernel = bit_dot_float(x_t, dy, 1/sqrt(N));
    stop = std::chrono::high_resolution_clock::now();
    printf("dw (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));

    return dx;
}