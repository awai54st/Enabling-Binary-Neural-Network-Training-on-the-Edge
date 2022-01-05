#ifndef vanilla_batchnorm_h
#define vanilla_batchnorm_h

#include "../../../utils/data_type.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/batch_norm_utils.h"
#include "../../../utils/base_layer.h"

template <template<typename> class MAT_CONTAINER = Matrix, typename RET_DTYPE=Matrix<float>>
class Vanilla_BatchNormDense : public BaseLayer<MAT_CONTAINER, float, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        float momentum;
        float eps=1e-34;
        MAT_CONTAINER<float> y;
        MAT_CONTAINER<float> beta;
        MAT_CONTAINER<float> dbeta;
        MAT_CONTAINER<float> moving_mean;
        MAT_CONTAINER<float> moving_var;
        //MAT_CONTAINER<float> mu;
        MAT_CONTAINER<float> var;
        
        //Constructors
        Vanilla_BatchNormDense(float momentum=0.9);
        //Deconstructors
        ~Vanilla_BatchNormDense();
        void build(size_t n_cols);
        RET_DTYPE forward(MAT_CONTAINER<float> & x, bool is_training=false);
        RET_DTYPE backprop(MAT_CONTAINER<float> & dy);
        float get_gradient(size_t index) {return dbeta[index];};
        MAT_CONTAINER<float> & get_weight() {return beta;};
};

template <template<typename> class MAT_CONTAINER = Matrix, typename RET_DTYPE=Matrix<float>>
class Vanilla_BatchNormConv : public BaseLayer<MAT_CONTAINER, float, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        float momentum;
        float eps=1e-34;
        MAT_CONTAINER<float> y;
        MAT_CONTAINER<float> beta;
        MAT_CONTAINER<float> dbeta;
        MAT_CONTAINER<float> moving_mean;
        MAT_CONTAINER<float> moving_var;
        //MAT_CONTAINER<float> mu;
        MAT_CONTAINER<float> var;
        
        //Constructors
        Vanilla_BatchNormConv(float momentum=0.9);
        //Deconstructors
        ~Vanilla_BatchNormConv();
        void build(size_t n_cols);
        //void update_moving_x(MAT_CONTAINER<float> & moving_x, MAT_CONTAINER<float> & x);
        //void compute_output(MAT_CONTAINER<float> & x);
        RET_DTYPE forward(MAT_CONTAINER<float> & x, bool is_training=false);
        RET_DTYPE backprop(MAT_CONTAINER<float> & dy);
        float get_gradient(size_t index) {return dbeta[index];};
        MAT_CONTAINER<float> & get_weight() {return beta;};
};

#endif