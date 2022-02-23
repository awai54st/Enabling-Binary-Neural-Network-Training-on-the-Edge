#ifndef batch_norm_utils_h
#define batch_norm_utils_h

#include "data_type.h"
#include <math.h> 



template <typename T=float, template<typename> class MAT_CONTAINER>
void update_moving_x(MAT_CONTAINER<T> & moving_x, const MAT_CONTAINER<T> & _x, const float momentum) {
    const unsigned int moving_x_size = moving_x.size();
    for (int i=0; i<moving_x_size; i++) {
        moving_x[i] = moving_x[i]*momentum + (1.0-momentum)*_x[i];
    }
}

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void l1_batch_norm_mean(const MAT_CONTAINER<T> & x, MAT_CONTAINER<T> & output) {
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0];
    
    //const size_t _size = output.size();
    //for (size_t i=0; i<_size; i++){
    //    output[i] = 0;
    //}
    
    for (int j = 0; j<input_shape[1]; j++) {
        float tmp_output = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            tmp_output += x(i,j);
        }
        tmp_output /= N;
        output[j] = tmp_output;
    }
    
    //for (size_t i=0; i<_size; i++){
    //    output[i] /= N;
    //}
}

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void l1_batch_norm_mean_4d(const MAT_CONTAINER<T> & x, MAT_CONTAINER<T> & output) {
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0]*input_shape[1]*input_shape[2];

    //const size_t _size = output.size();
    //for (size_t i=0; i<_size; i++){
    //    output[i] = 0;
    //}
    
    for (int l = 0; l<input_shape[3]; l++) {
        float tmp_output = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            for (int j = 0; j<input_shape[1]; j++) {
                for (int k = 0; k<input_shape[2]; k++) {
                    tmp_output += (x(i,j, k, l));
                }
            }
        }
        tmp_output /= N;
        output[l] = tmp_output;
    }
};


template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void l1_batch_norm_var(const MAT_CONTAINER<T> & x, const MAT_CONTAINER<T> & mu, MAT_CONTAINER<T> & var) {
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0];
    
    //const size_t _size = var.size();
    //for (size_t i=0; i<_size; i++){
    //    var[i] = 0;
    //}
    
    for (int j = 0; j<input_shape[1]; j++) {
        float tmp_var = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            tmp_var += fabs(x(i,j)-mu[j]);
        }
        tmp_var /= N;
        var[j] += tmp_var;
    }
    /*
    const size_t _size = var.size();
    for (size_t i=0; i<_size; i++){
        var.m_data[i] /= N;
    }
    */
}

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void l1_batch_norm_var_4d(const MAT_CONTAINER<T> & x, const MAT_CONTAINER<T> & mu, MAT_CONTAINER<T> & var) {
    const std::vector<size_t> input_shape = x.shape();
    const float N = input_shape[0]*input_shape[1]*input_shape[2];
    
    //const size_t _size = var.size();
    //for (size_t i=0; i<_size; i++){
    //    var[i] = 0;
    //}
    
    for (int l = 0; l<input_shape[3]; l++) {
        float tmp_var = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            for (int j = 0; j<input_shape[1]; j++) {
                for (int k = 0; k<input_shape[2]; k++) {
                    tmp_var += fabs(x(i,j,k,l)-mu[l]);
                }
            }
        }
        tmp_var /= N;
        var[l] = tmp_var;
    }
};


template <typename T=float, template<typename> class MAT_CONTAINER>
void compute_output(MAT_CONTAINER<T> & x, MAT_CONTAINER<T> & mu, MAT_CONTAINER<T> & var, MAT_CONTAINER<T> & beta, float eps=1e-34) {
    const std::vector<size_t> input_shape = x.shape();
    for (int i = 0; i<input_shape[0]; i++) {
        for (int j = 0; j<input_shape[1]; j++) {
            float tmp_x = x(i, j);
            tmp_x -= mu[j];
            tmp_x /= (float(var[j])+eps);
            tmp_x += beta[j];
            
            x.set(tmp_x, i, j);
        }
    }
}

template <typename T=float, template<typename> class MAT_CONTAINER>
void compute_output_4d(MAT_CONTAINER<T> & x, MAT_CONTAINER<T> & mu, MAT_CONTAINER<T> & var, MAT_CONTAINER<T> & beta, float eps=1e-34) {
    const std::vector<size_t> input_shape = x.shape();
    for (int i = 0; i<input_shape[0]; i++) {
        for (int j = 0; j<input_shape[1]; j++) {
            for (int k = 0; k<input_shape[2]; k++) {
                for (int l = 0; l<input_shape[3]; l++) {
                    float x_tmp = x(i, j, k, l);
                    x_tmp -= mu[l];
                    x_tmp /= (float(var[l])+eps);
                    x_tmp += beta[l];
                    x.set(x_tmp , i, j, k, l);
                }
            }
        }
    }
};

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void beta_gradient(MAT_CONTAINER<T> & dbeta, const MAT_CONTAINER<T> & dy) {
    const std::vector<size_t> input_shape = dy.shape();
    
    //const size_t _size = dbeta.size();
    //for (size_t i=0; i<_size; i++){
    //    dbeta[i] = 0;
    //}
    
    for (int j = 0; j<input_shape[1]; j++) {
        float tmp_beta = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            tmp_beta += T(dy(i, j));
        }
        dbeta[j] = tmp_beta;
    }
}

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void beta_gradient_4d(MAT_CONTAINER<T> & dbeta, const MAT_CONTAINER<T> & dy) {
    const std::vector<size_t> input_shape = dy.shape();
    
    //const size_t _size = dbeta.size();
    //for (size_t i=0; i<_size; i++){
    //    dbeta[i] = 0;
    //}
    
    for (int l = 0; l<input_shape[3]; l++) {
        float tmp_beta = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            for (int j = 0; j<input_shape[1]; j++) {
                for (int k = 0; k<input_shape[2]; k++) {
                    tmp_beta += T(dy(i, j, k, l));
                }
            }
        }
        dbeta[l] = tmp_beta;
    }
};

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void get_dy_norm_x(MAT_CONTAINER<T> & dy, MAT_CONTAINER<T> & var, float eps=1e-34) {
    const std::vector<size_t> input_shape = dy.shape();
    for (int i = 0; i<input_shape[0]; i++) {
        for (int j = 0; j<input_shape[1]; j++) {
            //printf("dy: %f", dy(i, j));
            //printf("var: %f", var[j]);
            //printf("norm: %f", dy(i, j)/ (float(var[j])+eps));
            dy.set( float(dy(i, j)) / (float(var[j])+eps), i, j);
        }
    }
}

template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void get_dy_norm_x_4d(MAT_CONTAINER<T> & dy, MAT_CONTAINER<T> & var, float eps=1e-34) {
    const std::vector<size_t> input_shape = dy.shape();
    for (int i = 0; i<input_shape[0]; i++) {
        for (int j = 0; j<input_shape[1]; j++) {
            for (int k = 0; k<input_shape[2]; k++) {
                for (int l = 0; l<input_shape[3]; l++) {
                    dy.set(float(dy(i, j, k, l))/(float(var[l])+eps), i, j, k, l);
                }
            }
        }
    }
};


// proposed batch norm calculation
template <template<typename> class MAT_CONTAINER>
void abs_mean(const MAT_CONTAINER<float16_t> & x, MAT_CONTAINER<float16_t> & output) {
    const std::vector<size_t> input_shape = x.shape();
    float scale = input_shape[0];
    
    //const size_t _size = output.size();
    //for (size_t i=0; i<_size; i++){
    //    output[i] = 0;
    //}
    
    for (int j = 0; j<input_shape[1]; j++) {
        float tmp_output = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            tmp_output += fabs(x(i, j));
        }
        tmp_output /= scale;
        output[j] = tmp_output;
    }
    
    //for (int i = 0; i<_size; i++) {
    //    output[i] /= scale;
    //}
}


template <typename T=float, template<typename> class MAT_CONTAINER = Matrix>
void abs_mean_4d(const MAT_CONTAINER<T> & x, MAT_CONTAINER<T> & output) {
    const std::vector<size_t> input_shape = x.shape();
    
    //const size_t _size = output.size();
    //for (size_t i=0; i<_size; i++){
    //    output[i] = 0;
    //}
    
    float scale = input_shape[0]*input_shape[1]*input_shape[2];
    for (int l = 0; l<input_shape[3]; l++) {
        float tmp_output = 0;
        for (int i = 0; i<input_shape[0]; i++) {
            for (int j = 0; j<input_shape[1]; j++) {
                for (int k = 0; k<input_shape[2]; k++) {
                    tmp_output += fabs(x(i, j, k, l));
                }
            }
        }
        tmp_output /= scale;
        output[l] = tmp_output;
    }
    
    //for (int i = 0; i<_size; i++) {
    //    output[i] /= scale;
    //}
};

#endif