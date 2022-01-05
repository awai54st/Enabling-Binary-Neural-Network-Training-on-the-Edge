#include "xnor_batchnorm.h"
#include <iostream>
#include <math.h> 


template class XNor_BatchNormConv<Matrix, Matrix<float16_t>&>;
template class XNor_BatchNormConv<Matrix, Matrix<float16_t>>;
//template class XNor_BatchNormConv<Matrix2D>;

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
XNor_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::XNor_BatchNormConv(float momentum): momentum(momentum) {};

//Deconstructors
template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
XNor_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::~XNor_BatchNormConv() {};

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
void XNor_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::build(size_t n_cols) {
    //std::cout << "n_cols: " << n_cols;
    moving_mean.resize({n_cols, 1});
    moving_var.resize({n_cols, 1}, 1);
    //mu.resize({n_cols, 1});
    var.resize({n_cols, 1});
    beta.resize({n_cols, 1});
    dbeta.resize({n_cols, 1});
    y_mean.resize({n_cols, 1});
    is_built = true;
};


template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE XNor_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::forward(MAT_CONTAINER<float16_t> & x, bool is_training) {
    //Matrix2D<float> y(x.n_rows, x.n_cols);
    
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0]*input_shape[1]*input_shape[2];
    if (is_built==false) {
        build(input_shape[3]);
        packed_y.resize(input_shape);
    }
    MAT_CONTAINER<float16_t> mu({input_shape[3], 1}, 0);
    if (is_training) {
        l1_batch_norm_mean_4d<float16_t>(x, mu);
        l1_batch_norm_var_4d<float16_t>(x, mu, var);
        update_moving_x(moving_mean, mu, momentum);
        update_moving_x(moving_var, var, momentum);
    } else {
        mu = moving_mean;
        var = moving_var;
    }
    compute_output_4d<float16_t>(x, mu, var, beta, eps);
    if (is_training == true) {
        pack_bits(x, packed_y, false);
        abs_mean_4d<float16_t>(x, y_mean);
    }
    
    return x;
};


template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE XNor_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::backprop(MAT_CONTAINER<float16_t> &dy) {
    const std::vector<size_t> input_shape = dy.shape();
    const std::vector<size_t> packed_y_shape = packed_y.shape();
    const float N = input_shape[0]*input_shape[1]*input_shape[2];
    beta_gradient_4d<float16_t>(dbeta, dy);
    
    get_dy_norm_x_4d<float16_t>(dy, var);
    
    
    // term 1 mean
    MAT_CONTAINER<float> dy_norm_x_mean({packed_y_shape[3], 1});
    for (size_t l=0; l < input_shape[3]; l++) {
        float tmp_mean = 0;
        for (size_t i=0; i < input_shape[0]; i++) {
            for (size_t j=0; j < input_shape[1]; j++) {
                for (size_t k=0; k < input_shape[2]; k++) {
                    tmp_mean += dy(i, j, k, l);
                }
            }
        }
        dy_norm_x_mean[l] = tmp_mean;
    }
    for (size_t i=0; i<dy_norm_x_mean.size(); i++) {
        dy_norm_x_mean[i] /= N;
    }
    
    // term 2
    // packed_y;
    
    // term 3
    MAT_CONTAINER<float> term_3({packed_y_shape[3], 1});
    for (size_t l=0; l < input_shape[3]; l++) {
        float term_3_tmp = 0;
        for (size_t i=0; i < input_shape[0]; i++) {
            for (size_t j=0; j < input_shape[1]; j++) {
                for (size_t k=0; k < input_shape[2]; k++) {
                    if (packed_y(i, j, k, l)) {
                        term_3_tmp += dy(i, j, k, l)*float(y_mean[l]);
                    } else {
                        term_3_tmp -= dy(i, j, k, l)*float(y_mean[l]);
                    }
                }
            }
        }
        term_3[l] = term_3_tmp;
    }
    for (size_t i=0; i<term_3.size(); i++) {
        term_3[i] /= N;
    }
    
    // term 1 full
    //for (size_t i=0; i < input_shape[0]; i++) {
    //    for (size_t j=0; j < input_shape[1]; j++) {
    //        for (size_t k=0; k < input_shape[2]; k++) {
    //            for (size_t l=0; l < input_shape[3]; l++) {
    //                dy.set(dy(i, j, k, l)-dy_norm_x_mean[l], i, j, k, l);
    //            }
    //        }
    //    }
    //}
    //printf("term_1: \n");
    //print_mat(dy);
    
    // output
    // term_1 - term_2*term_3
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            for (size_t k=0; k < input_shape[2]; k++) {
                for (size_t l=0; l < input_shape[3]; l++) {
                    if (packed_y(i, j, k, l)) {
                        dy.set(dy(i, j, k, l) - term_3[l]-dy_norm_x_mean[l], i, j, k, l);
                    } else {
                        dy.set(dy(i, j, k, l) + term_3[l]-dy_norm_x_mean[l], i, j, k, l);
                    }
                }
            }
        }
    }
    return dy;
};
