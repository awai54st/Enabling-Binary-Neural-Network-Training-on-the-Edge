#include "vanilla_batchnorm.h"
#include <iostream>
//#include <math.h> 


template class Vanilla_BatchNormConv<Matrix, Matrix<float>>;
template class Vanilla_BatchNormConv<Matrix, Matrix<float>&>;
//template class Vanilla_BatchNormConv<Matrix2D>;


template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
Vanilla_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::Vanilla_BatchNormConv(float momentum): momentum(momentum) {};

//Deconstructors
template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
Vanilla_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::~Vanilla_BatchNormConv() {};

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
void Vanilla_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::build(size_t n_cols) {
    //std::cout << "n_cols: " << n_cols;
    moving_mean.resize({n_cols, 1});
    moving_var.resize({n_cols, 1}, 1);
    //mu.resize({n_cols, 1});
    var.resize({n_cols, 1});
    beta.resize({n_cols, 1});
    dbeta.resize({n_cols, 1});
    is_built = true;
};

/*
template <template<typename> class MAT_CONTAINER>
void Vanilla_BatchNormConv<MAT_CONTAINER>::update_moving_x(MAT_CONTAINER<float> & moving_x, MAT_CONTAINER<float> & x) {
    const unsigned int moving_x_size = moving_x.size();
    for (int i=0; i<moving_x_size; i++) {
        moving_x[i] = moving_x[i]*momentum + (1.0-momentum)*x[i];
    }
};

template <template<typename> class MAT_CONTAINER>
void Vanilla_BatchNormConv<MAT_CONTAINER>::compute_output(MAT_CONTAINER<float> & x) {
    const std::vector<size_t> input_shape = x.shape();
    for (int i = 0; i<input_shape[0]; i++) {
        for (int j = 0; j<input_shape[1]; j++) {
            for (int k = 0; k<input_shape[2]; k++) {
                for (int l = 0; l<input_shape[3]; l++) {
                    x.set(x(i, j, k, l) - float(mu[l]), i, j, k, l);
                    x.set(x(i, j, k, l) / float(var[l]), i, j, k, l);
                    x.set(x(i, j, k, l) + float(beta[l]), i, j, k, l);
                }
            }
        }
    }
};
*/

template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE Vanilla_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::forward(MAT_CONTAINER<float> & x, bool is_training) {
    //Matrix2D<float> y(x.n_rows, x.n_cols);
    
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0]*input_shape[1]*input_shape[2];
    if (is_built==false) {
        build(input_shape[3]);
        is_built=true;
    }
    MAT_CONTAINER<float> mu({input_shape[3], 1});
    if (is_training) {
        l1_batch_norm_mean_4d<float>(x, mu);
        //printf("mu: "); print_mat(mu);
        l1_batch_norm_var_4d<float>(x, mu, var);
        //printf("var: "); print_mat(var);
        update_moving_x(moving_mean, mu, momentum);
        update_moving_x(moving_var, var, momentum);
    } else {
        mu = moving_mean;
        var = moving_var;
    }
    //printf("x b4: "); print_mat(x);
    compute_output_4d<float, MAT_CONTAINER>(x, mu, var, beta, eps);
    //printf("x after: "); print_mat(x);
    if (is_training == true) {
        y = x;
    }
    
    return x;
};



template <template<typename> class MAT_CONTAINER, typename RET_DTYPE>
RET_DTYPE Vanilla_BatchNormConv<MAT_CONTAINER, RET_DTYPE>::backprop(MAT_CONTAINER<float> &dy) {
    beta_gradient_4d<float>(dbeta, dy);
    
    const std::vector<size_t> input_shape = dy.shape();
    const std::vector<size_t> y_shape = y.shape();
    const float N = input_shape[0]*input_shape[1]*input_shape[2];
    
    get_dy_norm_x_4d(dy, var);
    
    // term 1 mean
    MAT_CONTAINER<float> dy_norm_x_mean({y_shape[3]});
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            for (size_t k=0; k < input_shape[2]; k++) {
                for (size_t l=0; l < input_shape[3]; l++) {
                    dy_norm_x_mean[l] += dy(i, j, k, l);
                }
            }
        }
    }
    for (size_t i=0; i<dy_norm_x_mean.size(); i++) {
        dy_norm_x_mean[i] /= N;
    }
    
    
    // term 2
    // y;
    
    // term 3
    MAT_CONTAINER<float> term_3({y_shape[3], 1});
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            for (size_t k=0; k < input_shape[2]; k++) {
                for (size_t l=0; l < input_shape[3]; l++) {
                    term_3[l] += (dy(i, j, k, l)*y(i, j, k, l)/N);
                }
            }
        }
    }
    
    // output
    // term_1 - term_2*term_3
    for (size_t i=0; i < input_shape[0]; i++) {
        for (size_t j=0; j < input_shape[1]; j++) {
            for (size_t k=0; k < input_shape[2]; k++) {
                for (size_t l=0; l < input_shape[3]; l++) {
                    dy.set(dy(i, j, k, l)-dy_norm_x_mean[l] - term_3[l]*y(i, j, k, l), i, j, k, l);
                }
            }
        }
    }
    
    return dy;
};
