#include "activation_layer.h"
#include <cmath>

// explicit instantiations
template class BinaryActivation<bool, Matrix, float16_t, Matrix<float16_t>>;
template class BinaryActivation<bool, Matrix, float16_t, Matrix<float16_t>&>;
template class BinaryActivation<float, Matrix, float, Matrix<float>>;
template class BinaryActivation<float, Matrix, float, Matrix<float>&>;
//template class BinaryActivation<bool, Matrix2D, float16_t, Matrix2D<float16_t>>;
//template class BinaryActivation<float, Matrix2D, float, Matrix2D<float>>;

template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
BinaryActivation<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::BinaryActivation(void) {};

//Deconstructors
template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
BinaryActivation<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::~BinaryActivation() {};

template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
RET_DTYPE BinaryActivation<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training) {
    //Matrix2D<bool> y(x.n_rows, x.n_cols);
    std::vector<size_t> x_shape = x.shape();
    
    if (is_built==false) {
        packed_y.resize(x_shape);
        is_built = true;
    }
    const unsigned int size = x.size();
    
    for (int i=0; i<size; i++) {
        if (fabs(x[i]) >1) {
            packed_y[i] = 0;
        } else {
            packed_y[i] = 1;
        }
    }
    
    for (int i=0; i<size; i++) {
        if (x[i] < 0) {
            x[i] = -1;
            //x[i] = 0;
        } else {
            x[i] = 1;
        }
    }
    //return y;
    return x;
};

template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
RET_DTYPE BinaryActivation<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy) {
    const unsigned int size = packed_y.size();
    for (int i=0; i<size; i++) {
        if (packed_y[i] == 0) {
            dy[i] = 0;
        }
    }
    return dy;
};