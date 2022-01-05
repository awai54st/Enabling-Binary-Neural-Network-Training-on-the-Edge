#include "gradient_quantisation_utils.h"
#include <algorithm> // max
#include <cmath> // log2, round

template <class T>
T sign(T value) {
    return value>=0? 1:-1;
};

template <class T>
constexpr const T& clip(const T &value, const T &min_val, const T &max_val) {
    return std::min(
        max_val, std::max(min_val, value) );
};

int8_t get_po2_bias(const Matrix2D<float> & x) {
    float max_val = x.data[0];
    
    // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
    const int mat_2d_size = x.n_rows*x.n_cols;
    for (int i =0; i<mat_2d_size; i++) {
        max_val = std::max(max_val, std::abs(x.data[i]));
    }
    return int8_t(-round(log2(max_val)) + 8);
};

void scale_with_po2_bias(Matrix2D<float> & x, const int8_t bias) {
    const auto bias_shift = [bias](float x) {
        if (bias>=0) {
            return x * (float) (1<<bias);
        } else {
            return x / (float) (1<<bias);
        }
    };
    
    const int mat_2d_size = x.n_rows*x.n_cols;
    for (int i =0; i<mat_2d_size; i++) {
        x.data[i] = bias_shift(x.data[i]);
    }
    return;
};


void log_quantize(Matrix2D<float> & x, const int8_t bias, const int8_t width) {
    const float max_value = (1<<(width-1));
    
    // currying
    auto _clip_partial = [max_value](float f) {
        return clip<float>(
            round( log2(std::abs(f)+1e-45f ) ), 
            -max_value, (max_value-1));
    };
    
    int i;
    int8_t tmp_po2;
    const int mat_2d_size = x.n_rows*x.n_cols;
    for (int i =0; i<mat_2d_size; i++) {
        tmp_po2 = _clip_partial(x.data[i]) - bias;
        float tmp_val = x.data[i];
        if (tmp_po2 < 0) {
            x.data[i] = sign<float>(tmp_val) / (1 << -tmp_po2 );
        } else {
            x.data[i] = sign<float>(tmp_val) * (1 << tmp_po2 );
        }
    }
    
    return;
};
