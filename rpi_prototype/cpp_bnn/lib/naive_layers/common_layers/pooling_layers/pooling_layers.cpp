#include <iostream>
#include <algorithm>  //std::max()
#include "pooling_layers.h"

template class MaxPooling<bool, Matrix, float16_t, Matrix<float16_t>>;
template class MaxPooling<bool, Matrix, float16_t, Matrix<float16_t>&>;
template class MaxPooling<float, Matrix, float, Matrix<float>>;
template class MaxPooling<float, Matrix, float, Matrix<float>&>;
//template class MaxPooling<bool, Matrix2D, float16_t>;
//template class MaxPooling<float, Matrix2D, float>;

template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
MaxPooling<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::MaxPooling(int kernel_size, int stride): 
kernel_size(kernel_size), stride(stride) {};

//Deconstructors
template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
MaxPooling<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::~MaxPooling() {};

template <template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE>
void _set_to_input(MAT_CONTAINER<WEIGHT_DTYPE> & x, float max_val, std::vector<size_t> index, std::vector<size_t> output_shape) {
    size_t stride_k = output_shape[3];
    size_t stride_j = output_shape[2]*stride_k;
    size_t stride_i = output_shape[1]*stride_j;
    
    x[index[0]*stride_i+index[1]*stride_j+index[2]*stride_k+index[3]] = max_val;
}

template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
RET_DTYPE MaxPooling<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training) {
    //Matrix2D<bool> y(x.n_rows, x.n_cols);
    //if (is_built==false) {
    //    packed_x.resize(x.shape());
    //    is_built = true;
    //}
    x_shape = x.shape();
    std::vector<size_t> output_shape = {x_shape[0], x_shape[1]/stride, x_shape[2]/stride, x_shape[3]};
    
    packed_x.resize(x_shape);
    std::vector<size_t> strides = x.strides();
    size_t i, j, k, l, j_s, k_s;
    
    for (i=0; i<output_shape[0]; i++) {
        for (l=0; l<output_shape[3]; l++) {
            for (j=0; j<output_shape[1]; j++) {
                for (k=0; k<output_shape[2]; k++) {
                    int j_ori = j*stride;
                    int k_ori = k*stride;
                    int j_tmp = j_ori;
                    int k_tmp = k_ori;
                    float tmp_max = x(i, j_tmp, k_tmp, l);
                    for (j_s=0; j_s<stride; j_s++) {
                        for (k_s=0; k_s<stride; k_s++) {
                            if (x(i, j_ori+j_s, k_ori+k_s, l)>tmp_max) {
                                j_tmp = j_ori+j_s;
                                k_tmp = k_ori+k_s;
                                tmp_max = x(i, j_tmp, k_tmp, l);
                            }
                        }
                    }
                    packed_x.set(1, i, j_tmp, k_tmp, l);
                    _set_to_input<MAT_CONTAINER, WEIGHT_DTYPE>(x, tmp_max, {i, j, k, l}, output_shape);
                }
            }
        }
    }
    x.resize(output_shape);
    return x;
};


template <template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE>
void _align_memory_before_pool(MAT_CONTAINER<WEIGHT_DTYPE> &x, std::vector<size_t> output_shape, size_t stride=2) {

    std::vector<size_t> o_shape = x.shape();
    std::vector<size_t> o_strides = x.strides();
    //printf("x before resize");
    //print_mat(x);
    x.resize(output_shape);
    
    for (int i=(o_shape[0]-1); i>-1; i--) {
        for (int l=(o_shape[3]-1); l>-1; l--) {
            for (int j=(o_shape[1]-1); j>-1; j--) {
                int j_out = j*2;
                for (int k=(o_shape[2]-1); k>-1; k--) {
                    int k_out = k*2;
                    float grad_val = x[i*o_strides[0]+j*o_strides[1]+k*o_strides[2]+l];
                    //printf("index 1: %d, %d, %d, %d \n", i, j, k, l);
                    //printf("stride: %d\n", stride);
                    //printf("stride: %f\n", grad_val);
                    for (int j_s=0; j_s<stride; j_s++) {
                        for (int k_s=0; k_s<stride; k_s++) {
                            //printf("index: %d, %d, %d, %d\n", i, j_out+j_s, k_out+k_s, l);
                            x.set(grad_val, i, j_out+j_s, k_out+k_s, l);
                        }
                    }
                }
            }
        }
    }
    //printf("after resize: ");
    //print_mat(x);
}


template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE, typename RET_DTYPE>
RET_DTYPE MaxPooling<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE>::backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy) {
    std::vector<size_t> shape = dy.shape();
    
    _align_memory_before_pool<MAT_CONTAINER, WEIGHT_DTYPE>(dy, x_shape, stride);
    int size = packed_x.size();
    
    for (int i=0; i<size; i++) {
        if (packed_x[i] == 0) {
            dy[i] = 0;
        }
    }
    
    /*
    for (int i=0; i<shape[0]; i++) {
        for (int j=0; j<shape[1]; j++) {
            for (int k=0; k<shape[2]; k++) {
                int j_ori = j*2;
                int k_ori = k*2;
                for (int l=0; l<shape[3]; l++) {
                    for (int j_s=0; j_s<stride; j_s++) {
                        for (int k_s=0; k_s<stride; k_s++) {
                            if (packed_x(i, j_ori+j_s, k_ori+k_s, l) == 0) {
                                // dx.set(dy(i, j, k, l), i, j_ori+j_s, k_ori+k_s, l);
                                dy.set(0, i, j_ori+j_s, k_ori+k_s, l);
                            }
                        }
                    }
                }
            }
        }
    }
    */
    return dy;
};
/*
template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE>
MAT_CONTAINER<float> MaxPooling<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE>::forward(MAT_CONTAINER<float> & x, bool is_training) {
    //Matrix2D<bool> y(x.n_rows, x.n_cols);
    //if (is_built==false) {
    //    packed_x.resize(x.shape());
    //    is_built = true;
    //}
    x_shape = x.shape();
    std::vector<size_t> output_shape = {x_shape[0], x_shape[1]/stride, x_shape[2]/stride, x_shape[3]};
    
    MAT_CONTAINER<float> y(output_shape);
    packed_x.resize(x_shape);
    std::vector<size_t> strides = x.strides();
    size_t i, j, k, l, j_s, k_s;
    
    for (i=0; i<output_shape[0]; i++) {
        for (l=0; l<output_shape[3]; l++) {
            for (j=0; j<output_shape[1]; j++) {
                for (k=0; k<output_shape[2]; k++) {
                    int j_ori = j*stride;
                    int k_ori = k*stride;
                    int j_tmp = j_ori;
                    int k_tmp = k_ori;
                    float tmp_max = x(i, j_tmp, k_tmp, l);
                    for (j_s=0; j_s<stride; j_s++) {
                        for (k_s=0; k_s<stride; k_s++) {
                            if (x(i, j_ori+j_s, k_ori+k_s, l)>tmp_max) {
                                j_tmp = j_ori+j_s;
                                k_tmp = k_ori+k_s;
                                tmp_max = x(i, j_tmp, k_tmp, l);
                            }
                        }
                    }
                    packed_x.set(1, i, j_tmp, k_tmp, l);
                    y.set(tmp_max, i, j, k, l);
                }
            }
        }
    }
    return y;
};

template <class PACK_T, template<typename> class MAT_CONTAINER, typename WEIGHT_DTYPE>
MAT_CONTAINER<float> MaxPooling<PACK_T, MAT_CONTAINER, WEIGHT_DTYPE>::backprop(MAT_CONTAINER<float> & dy) {
    std::vector<size_t> shape = dy.shape();
    MAT_CONTAINER<float> dx(x_shape);
    int size = packed_x.size();
    for (int i=0; i<shape[0]; i++) {
        for (int j=0; j<shape[1]; j++) {
            for (int k=0; k<shape[2]; k++) {
                int j_ori = j*2;
                int k_ori = k*2;
                for (int l=0; l<shape[3]; l++) {
                    for (int j_s=0; j_s<stride; j_s++) {
                        for (int k_s=0; k_s<stride; k_s++) {
                            if (packed_x(i, j_ori+j_s, k_ori+k_s, l) == 1) {
                                dx.set(dy(i, j, k, l), i, j_ori+j_s, k_ori+k_s, l);
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return dx;
};
*/
