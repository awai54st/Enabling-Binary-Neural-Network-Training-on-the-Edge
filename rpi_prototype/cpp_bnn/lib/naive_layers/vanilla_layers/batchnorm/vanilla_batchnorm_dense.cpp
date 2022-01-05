#include "vanilla_batchnorm.h"
#include <iostream>


template class Vanilla_BatchNormDense<Matrix, Matrix<float>>;
template class Vanilla_BatchNormDense<Matrix, Matrix<float>&>;
//template class Vanilla_BatchNormDense<Matrix2D>;

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
Vanilla_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::Vanilla_BatchNormDense(float momentum): momentum(momentum) {};

//Deconstructors
template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
Vanilla_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::~Vanilla_BatchNormDense() {};

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
void Vanilla_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::build(size_t n_cols) {
    //std::cout << "n_cols: " << n_cols;
    moving_mean.resize({n_cols, 1});
    moving_var.resize({n_cols, 1}, 1);
    //mu.resize({n_cols, 1});
    var.resize({n_cols, 1});
    beta.resize({n_cols, 1});
    dbeta.resize({n_cols, 1});
    is_built = true;
};


template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE Vanilla_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::forward(MAT_CONTAINER<float> & x, bool is_training) {
    //Matrix2D<float> y(x.n_rows, x.n_cols);
    
    const std::vector<size_t> input_shape = x.shape();
    if (is_built==false) {
        build(input_shape[1]);
        is_built = true;
    }
    MAT_CONTAINER<float> mu({input_shape[1], 1}, 0);
    if (is_training) {
        l1_batch_norm_mean<float>(x, mu);
        l1_batch_norm_var<float>(x, mu, var);
        update_moving_x(moving_mean, mu, momentum);
        update_moving_x(moving_var, var, momentum);
    } else {
        mu = moving_mean;
        var = moving_var;
    }
    compute_output(x, mu, var, beta, eps);
    y = x;
    
    return x;
};

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE Vanilla_BatchNormDense<MAT_CONTAINER, RET_DTYPE>::backprop(MAT_CONTAINER<float> &dy) {
    beta_gradient<float>(dbeta, dy);
    
    const std::vector<size_t> input_shape = dy.shape();
    const std::vector<size_t> y_shape = y.shape();
    const float N = input_shape[0];
    
    get_dy_norm_x(dy, var);
    
    // term 1 mean
    MAT_CONTAINER<float> dy_norm_x_mean({y_shape[1]});
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            dy_norm_x_mean[j] += dy(i, j);
        }
    }
    for (size_t i=0; i<dy_norm_x_mean.size(); i++) {
        dy_norm_x_mean[i] /= N;
    }
    
    // term 2
    // y;
    
    // term 3
    MAT_CONTAINER<float> term_3({y_shape[1], 1});
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            term_3[j] += (dy(i, j)*y(i, j)/N);
        }
    }
    
    // output
    // term_1 - term_2*term_3
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            dy.set(dy(i, j)-dy_norm_x_mean[j] - term_3[j]*y(i, j), i, j);
        }
    }
    
    return dy;
};
