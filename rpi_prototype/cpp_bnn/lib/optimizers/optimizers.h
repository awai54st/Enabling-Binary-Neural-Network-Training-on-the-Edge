#ifndef optimizers_h
#define optimizers_h

#include "../utils/data_type.h"
#include "../utils/base_layer.h"

template <typename WEIGHT_DTYPE=float>
float calculate_corrected_gradient(Matrix<WEIGHT_DTYPE> &grads, Matrix<WEIGHT_DTYPE> &m_adam, Matrix<WEIGHT_DTYPE> &v_adam, const float lr=0.001, const float beta1=0.9, const float beta2=0.999, const size_t curr_step=1);

template <typename WEIGHT_DTYPE=float, typename RET_DTYPE=Matrix<WEIGHT_DTYPE>>
class Adam {
    public:
        bool is_built = false;
        float lr;
        float beta1;
        float beta2;
        float eps = 1e-8;
        int curr_step=0;
        std::vector<Matrix<WEIGHT_DTYPE>> m_adam_arrs;
        std::vector<Matrix<WEIGHT_DTYPE>> v_adam_arrs;
        std::vector<char> idxes;
        size_t reserve_size;
    
        //Constructors
        Adam(float lr=0.001, float beta1= 0.9, float beta2=0.999, size_t reserve_size=10);
        //Deconstructors
        ~Adam();
    
        void update(std::vector<std::unique_ptr<BaseLayer<Matrix, WEIGHT_DTYPE, RET_DTYPE>>> & layers);
};

#endif