#include "optimizers.h"
#include <cmath>
#include "../utils/check_utils.h"


template class Adam<float, Matrix<float>>;
template class Adam<float, Matrix<float>&>;
template class Adam<float16_t, Matrix<float16_t>>;
template class Adam<float16_t, Matrix<float16_t>&>;

/*
template <typename WEIGHT_DTYPE=float>
float calculate_corrected_gradient(Matrix<WEIGHT_DTYPE> &grads, Matrix<WEIGHT_DTYPE> &m_adam, Matrix<WEIGHT_DTYPE> &v_adam, const float lr, const float beta1, const float beta2, const int curr_step) {
    float adam_lr = lr * sqrtf(1.0-pow(beta2, float(curr_step)))/(1.0-pow(beta1, float(curr_step)));
    //printf("lr: %f\n", lr);
    //printf("denominator: %f\n", (1.0-pow(beta1, float(curr_step))));
    //printf("sqrt: %f\n", sqrtf(1.0-pow(beta2, float(curr_step))));
    //printf("Adam new lr: %f\n", adam_lr);
    
    size_t grads_size = grads.size();
    for (size_t i=0; i<grads_size; i++) {
        m_adam[i] = m_adam[i]*beta1 + (1.0-beta1)*grads[i];
        v_adam[i] = v_adam[i]*beta2 + (1.0-beta2)*(grads[i]*grads[i]);
        //printf("beta2: %f\n", (1.0-beta2));
        //printf("grad^2: %f\n", (grads[i]*grads[i]));
        //printf("v_adam: %f\n", v_adam[i]);
    }
    //printf("Calculate adam gradient");
    //has_inf(m_adam);
    //has_nan(m_adam);
    //has_inf(v_adam);
    //has_nan(v_adam);
    
    return adam_lr;
}
*/
template <typename WEIGHT_DTYPE=float>
void adam_update_weights(Matrix<WEIGHT_DTYPE> &weights, Matrix<WEIGHT_DTYPE> &m_adam, Matrix<WEIGHT_DTYPE> &v_adam, const float lr, const float eps) {
    size_t grads_size = weights.size();
    //printf("Before Update weight\n");
    //has_inf(weights);
    //has_nan(v_adam);
    for (size_t i=0; i<grads_size; i++) {
        float weight_adam_grad = (lr*m_adam[i]/(sqrtf(float(v_adam[i])+eps)) );
        weights[i] -= weight_adam_grad;
        //printf("weight grad: %f\n", weight_adam_grad);
        //printf("weights[%d] update %f * %f/%.8f \n", i, lr, m_adam[i], (sqrtf(v_adam[i])+eps));
    }
    //printf("Update weight\n");
    //has_nan(weights);
    //has_inf(weights);
    //print_mat(weights);
    return;
}

template <typename WEIGHT_DTYPE, typename RET_DTYPE>
Adam<WEIGHT_DTYPE, RET_DTYPE>::Adam(float lr, float beta1, float beta2, size_t reserve_size): 
    lr(lr), beta1(beta1), beta2(beta2), reserve_size(reserve_size){};

template <typename WEIGHT_DTYPE, typename RET_DTYPE>
Adam<WEIGHT_DTYPE, RET_DTYPE>::~Adam() {};

// https://yasenh.github.io/post/cpp-diary-1-emplace_back/
template <typename WEIGHT_DTYPE, typename RET_DTYPE>
void Adam<WEIGHT_DTYPE, RET_DTYPE>::update(std::vector<std::unique_ptr<BaseLayer<Matrix, WEIGHT_DTYPE, RET_DTYPE>>> &layers) {
    size_t n_layers = layers.size();
    // printf("Size\n");
    if (not is_built) {
        m_adam_arrs.reserve(reserve_size);
        v_adam_arrs.reserve(reserve_size);
        for (size_t layer_idx=0; layer_idx<n_layers; layer_idx++) {
            // printf("layer_idx: %d\n", layer_idx);
            // std::cout << "bool: %d\n" << layers[layer_idx]->get_gradient().data();
            if (layers[layer_idx]->get_weight().data() == NULL) {
                continue;
            }
            std::vector<size_t> grad_shape = layers[layer_idx]->get_weight().shape();
            Matrix<WEIGHT_DTYPE> m_adam(grad_shape, 0);
            Matrix<WEIGHT_DTYPE> v_adam(grad_shape, 0);
            m_adam_arrs.push_back(m_adam);
            v_adam_arrs.push_back(v_adam);
            idxes.emplace_back(layer_idx);
        }
        is_built = true;
    }
    curr_step++;
    float new_lr = lr * sqrtf(1.0-pow(beta2, float(curr_step)))/(1.0-pow(beta1, float(curr_step)));
    
    for (size_t i=0; i<idxes.size(); i++) {

        size_t grads_size = m_adam_arrs[i].size();
        for (size_t i_grad=0; i_grad<grads_size; i_grad++) {
            float grad = layers[idxes[i]]->get_gradient(i_grad);
            m_adam_arrs[i][i_grad] = m_adam_arrs[i][i_grad]*beta1 + (1.0-beta1)*grad;
            v_adam_arrs[i][i_grad] = v_adam_arrs[i][i_grad]*beta2 + (1.0-beta2)*(grad*grad);
        }
        // float new_lr = calculate_corrected_gradient(layers[idxes[i]]->get_gradient(), m_adam_arrs[i], v_adam_arrs[i], lr, beta1, beta2, curr_step);
        // printf("calculate m v\n");
        adam_update_weights(layers[idxes[i]]->get_weight(), m_adam_arrs[i], v_adam_arrs[i], new_lr, eps);
        // printf("update m v\n");
    }
}