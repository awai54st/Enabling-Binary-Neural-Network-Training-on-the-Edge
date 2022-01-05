#ifndef gradient_quantisation_h
#define gradient_quantisation_h

#include "data_type.h"
#include <algorithm> // max
#include <cmath> // log2, round

template <class T>
T sign(T value) {
    return value>=0? 1:-1;
};
template <class T>
T sign_po2(T value) {
    return value<0? 1:0;
};

//constexpr const T& clip(const T &value, const T &min_val, const T &max_val) {
template <class T>
const T& clip(const T &value, const T &min_val, const T &max_val) {
    return std::min(
        max_val, std::max(min_val, value) );
};

template <class T, template<typename> class MAT_CONTAINER = Matrix>
int8_t get_po2_bias(const MAT_CONTAINER<T> & x, const float eps=1e-34) {
    float max_val = x[0];
    //printf("max_value: %f", max_val);
    
    // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
    const int mat_2d_size = x.size();
    for (int i =0; i<mat_2d_size; i++) {
        max_val = std::max(max_val, std::fabs(x[i]));
    }
    
    int8_t tmp_bias = -round(log2(max_val+eps)) + 8;
    return clip<int8_t>(tmp_bias, -119, 119);
    
    //return int8_t(-round(log2(max_val+eps)) + 8);
};

template <class T, template<typename> class MAT_CONTAINER = Matrix>
int8_t get_po2_bias(const MAT_CONTAINER<PO2_5bits_t> & x, const float eps=1e-34) {
    int8_t max_val = x[0].value;
    //printf("max_value: %f", max_val);
    
    // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
    const int mat_2d_size = x.size();
    for (int i =0; i<mat_2d_size; i++) {
        max_val = std::max(max_val, x[i].value);
    }
    
    int8_t tmp_bias = max_val + 8;
    return clip<int8_t>(tmp_bias, -119, 119);
    
    //return int8_t(-round(log2(max_val+eps)) + 8);
};

template <class T, template<typename> class MAT_CONTAINER = Matrix>
void scale_with_po2_bias(MAT_CONTAINER<T> & x, const int8_t bias) {
    const auto bias_shift = [bias](float x) {
        if (bias>=0) {
            return x * float(1<<bias);
        } else {
            return x / float(1<<bias);
        }
    };
    
    const int mat_2d_size = x.size();
    for (int i =0; i<mat_2d_size; i++) {
        x[i] = bias_shift(x[i]);
    }
    return;
};

template <class T, template<typename> class MAT_CONTAINER = Matrix>
void log_quantize(MAT_CONTAINER<T> & x, const int8_t bias, const int8_t width=4, const float eps=1e-34) {
    const float max_value = float(1<<(width-1));
    
    // currying
    auto _clip_partial = [max_value, eps](float f) {
        return clip<float>(
            round( log2(fabs(f+eps)) ), 
            -max_value, (max_value-1));
    };
    
    long long int tmp_po2;
    const int mat_2d_size = x.size();
    for (int i =0; i<mat_2d_size; i++) {
        float tmp_x = x[i];
        tmp_po2 = _clip_partial(tmp_x) - bias;
        if (tmp_po2 < 0) {
            long long unsigned int a=1;
            x[i] = sign<T>(tmp_x) / float(a << -tmp_po2 );
        } else {
            long long unsigned int a=1;
            x[i] = sign<T>(tmp_x) * float(a << tmp_po2 );
        }
    }
    
    return;
};


template <class T, template<typename> class MAT_CONTAINER = Matrix>
void log_quantize(MAT_CONTAINER<T> & x, MAT_CONTAINER<PO2_5bits_t> & x_po2, const int8_t bias, const int8_t width=4, const float eps=1e-34) {
    const int8_t max_value = (1<<(width-1));
    
    // currying
    auto _clip_partial = [max_value, eps](float f) {
        return clip<float>(
            round( log2(fabs(f+eps)) ), 
            -max_value, (max_value-1));
    };
    
    const int mat_2d_size = x.size();
    for (int i =0; i<mat_2d_size; i++) {
        float tmp_x = x[i];
        //x_po2[i].sign = int8_t(sign_po2<T>(tmp_x));
        //x_po2[i].value = _clip_partial(tmp_x) - bias;
        x_po2[i].set(tmp_x, (_clip_partial(tmp_x) - bias));
        //printf("x_po2 sign: %d <->", x_po2[i].sign);
        //printf("x_po2 value: %d \n", x_po2[i].value);
    }
    
    return;
};

template <typename IN_T=float16_t, typename OUT_T=PO2_5bits_t>
Matrix<OUT_T> PO2_gradients(Matrix<IN_T>& dy, const float eps=1e-34) {
    Matrix<OUT_T> dy_PO2(dy.shape());
    
    const int8_t dy_bias = get_po2_bias<IN_T>(dy);
    scale_with_po2_bias<IN_T>(dy, dy_bias);
    log_quantize<IN_T>(dy, dy_PO2, dy_bias, 4, eps);
    
    //for (int i=0; i<dy_PO2.size(); i++) {
    //    printf("dy_PO2 sign: %d <->", dy_PO2[i].sign);
    //    printf("dy_PO2 value: %d \n", dy_PO2[i].value);
    //}
    
    
    return dy_PO2;
}


template <typename IN_T=float16_t, typename OUT_T=PO2_5bits_t>
void PO2_gradients(Matrix<IN_T>& dy, Matrix<OUT_T> &dy_PO2, const float eps=1e-34) {
    
    const int8_t dy_bias = get_po2_bias<IN_T>(dy);
    scale_with_po2_bias<IN_T>(dy, dy_bias);
    log_quantize<IN_T>(dy, dy_PO2, dy_bias, 4, eps);
    
    //for (int i=0; i<dy_PO2.size(); i++) {
    //    printf("dy_PO2 sign: %d <->", dy_PO2[i].sign);
    //    printf("dy_PO2 value: %d \n", dy_PO2[i].value);
    //}
    return;
}
template <typename T>
void dkernel_calculation(Matrix<T>& dkernel, const Matrix<T>& kernel, const float scale) {
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    const std::size_t size = dkernel.size();
    
    std::size_t i;
    for (i=0; i<size; i++) {
        if (dkernel.m_data[i] >= 0) {
            dkernel.m_data[i] = scale; // 1*scale;
        } else {
            dkernel.m_data[i] = -scale; // -1*scale;
        }
    }
    
    for (i=0; i<size; i++) {
        if (fabs(kernel.m_data[i]) >= 1) {
            dkernel.m_data[i] = 0;
        }
    }
};

#endif
