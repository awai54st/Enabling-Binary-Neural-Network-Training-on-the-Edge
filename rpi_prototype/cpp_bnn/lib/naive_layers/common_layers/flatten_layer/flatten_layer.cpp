#include "flatten_layer.h"

template class Flatten<Matrix, float, Matrix<float>>;
template class Flatten<Matrix, float, Matrix<float>&>;
template class Flatten<Matrix, float16_t, Matrix<float16_t>>;
template class Flatten<Matrix, float16_t, Matrix<float16_t>&>;
//template class Flatten<Matrix2D, float>;
//template class Flatten<Matrix2D, float16_t>;

template <template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
Flatten<MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::~Flatten() {};

template <template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
RET_DTYPE Flatten<MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training) {
    x_shape = x.shape();
    x.reshape(x_shape[0], x_shape[1]*x_shape[2]*x_shape[3], 0, 0);
    return x;
}


template <template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
RET_DTYPE Flatten<MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy) {
    dy.reshape(x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    return dy;
}