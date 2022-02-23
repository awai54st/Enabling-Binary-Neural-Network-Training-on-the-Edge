#include "softmax.h"
#include <cmath>
#include <algorithm>

template class Softmax<float, Matrix, Matrix<float>>;
template class Softmax<float, Matrix, Matrix<float>&>;
template class Softmax<float16_t, Matrix, Matrix<float16_t>>;
template class Softmax<float16_t, Matrix, Matrix<float16_t>&>;
//template class Softmax<float, Matrix2D>;
//template class Softmax<float16_t, Matrix2D>;

template <typename WEIGHT_DTYPE, template<typename> class MAT_CONTAINER, typename RET_DTYPE>
Softmax<WEIGHT_DTYPE, MAT_CONTAINER, RET_DTYPE>::Softmax() {};

template <typename WEIGHT_DTYPE, template<typename> class MAT_CONTAINER, typename RET_DTYPE>
Softmax<WEIGHT_DTYPE, MAT_CONTAINER, RET_DTYPE>::~Softmax() {};

template <typename WEIGHT_DTYPE, template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE Softmax<WEIGHT_DTYPE, MAT_CONTAINER, RET_DTYPE>::forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training) {
    const std::vector<size_t> x_shape = x.shape();
    MAT_CONTAINER<WEIGHT_DTYPE> row_max_val({x_shape[0]});
    
    // find max of each row
    for (size_t i=0; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            row_max_val.set(std::max(row_max_val(i), x(i,j)), i);
        }
    }
    
    // shift x for stable softmax
    for (size_t i=0; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            x.set(float(x(i,j))-row_max_val(i), i, j);
        }
    }
    
    // exponential of x
    for (size_t i=0; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            x.set(exp(float(x(i,j))), i, j);
        }
    }
    
    // row sum of exp x
    Matrix<float> row_sum_val({x_shape[0]});
    for (size_t i=0; i<x_shape[0]; i++) {
        float tmp_sum = 0;
        for (size_t j=0; j<x_shape[1]; j++) {
            tmp_sum += x(i,j);
        }
        row_sum_val.set(tmp_sum, i);
    }
    
    // output
    for (size_t i=0; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            x.set(float(x(i,j))/(row_sum_val(i)+eps), i, j);
        }
    }
    
    return x;
    
    
};

template <typename WEIGHT_DTYPE, template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE Softmax<WEIGHT_DTYPE, MAT_CONTAINER, RET_DTYPE>::backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy) {
    return dy;
};