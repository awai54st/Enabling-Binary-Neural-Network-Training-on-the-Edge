#include "xnor_batchnorm.h"
#include <iostream>
#include <math.h> 

template class XNor_BatchNormDense<Matrix, Matrix<float16_t>&>;
template class XNor_BatchNormDense<Matrix, Matrix<float16_t>>;
//template class XNor_BatchNormDense<Matrix2D>;

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
XNor_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::XNor_BatchNormDense(float momentum): momentum(momentum) {};

//Deconstructors
template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
XNor_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::~XNor_BatchNormDense() {};

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
void XNor_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::build(size_t n_cols) {
    //std::cout << "n_cols: " << n_cols;
    moving_mean.resize({n_cols, 1}, 0);
    moving_var.resize({n_cols, 1}, 1);
    var.resize({n_cols, 1}, 0);
    beta.resize({n_cols, 1}, 0);
    y_mean.resize({n_cols, 1}, 0);
    //mu.resize({n_cols, 1});
    is_built = true;
};

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE XNor_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::forward(MAT_CONTAINER<float16_t> & x, bool is_training) {
    //Matrix2D<float> y(x.n_rows, x.n_cols);
    
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0];
    if (is_built==false) {
        build(input_shape[1]);
        packed_y.resize(input_shape);
    }
    MAT_CONTAINER<float16_t> mu({input_shape[1], 1}, 0);
    if (is_training) {
        l1_batch_norm_mean<float16_t>(x, mu);
        l1_batch_norm_var<float16_t>(x, mu, var);
        update_moving_x<float16_t>(moving_mean, mu, momentum);
        update_moving_x<float16_t>(moving_var, var, momentum);
    } else {
        mu = moving_mean;
        var = moving_var;
    }
    compute_output(x, mu, var, beta);
    if (is_training == true) {
        pack_bits(x, packed_y);
        abs_mean(x, y_mean);
    }
    
    return x;
};


template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE XNor_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::backprop(MAT_CONTAINER<float16_t> &dy) {
    const std::vector<size_t> input_shape = dy.shape();
    const std::vector<size_t> packed_y_shape = packed_y.shape();
    const float N = input_shape[0];
    
    
    dbeta.resize({packed_y_shape[1], 1}, 0);
    beta_gradient<float16_t>(dbeta, dy);
    
    get_dy_norm_x<float16_t>(dy, var, eps);
    
    // term 1
    MAT_CONTAINER<float> dy_norm_x_mean({packed_y_shape[1], 1});
    for (size_t j=0; j < input_shape[1]; j++) {
        float tmp_dy_norm_x_mean = 0;
        for (size_t i=0; i < input_shape[0]; i++) {
            tmp_dy_norm_x_mean += dy(i, j);
        }
        tmp_dy_norm_x_mean /= N;
        dy_norm_x_mean[j] = tmp_dy_norm_x_mean;
    }
    //for (size_t i=0; i<dy_norm_x_mean.size(); i++) {
    //    dy_norm_x_mean[i] /= N;
    //}
    //printf("dy_norm_x_mean: ");print_mat(dy_norm_x_mean);
    // term 2
    // packed_y;
    
    //get_term_3(term_3, dy, y_mean, packed_y, 1.0/N);
    // term 3
    MAT_CONTAINER<float> term_3({packed_y_shape[1], 1});
    for (size_t j=0; j < input_shape[1]; j++) {
        float tmp_term_3 = 0;
        for (size_t i=0; i < input_shape[0]; i++) {
            if (packed_y(i, j)) {
                tmp_term_3 += float(dy(i, j)*y_mean[j]);
            } else {
                tmp_term_3 -= float(dy(i, j)*y_mean[j]);
            }
        }
        tmp_term_3 /= N;
        term_3[j] = tmp_term_3;
    }
    //printf("y_mean: "); print_mat(y_mean);
    //for (size_t i=0; i<term_3.size(); i++) {
    //    term_3[i] /= N;
    //}
    
    // output
    // term_1 - term_2*term_3
    //printf("packed_y: ");print_mat_bool(packed_y);
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            if (packed_y(i, j)==1) {
                dy.set(dy(i, j) - term_3[j] - dy_norm_x_mean[j], i, j);
            } else {
                dy.set(dy(i, j) + term_3[j] - dy_norm_x_mean[j], i, j);
            }
        }
    }
    return dy;
};