#ifndef xnor_batchnorm_h
#define xnor_batchnorm_h

#include "../../../utils/data_type.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/base_layer.h"
#include "../../../utils/batch_norm_utils.h"

template <template<typename> class MAT_CONTAINER = Matrix, typename RET_DTYPE=Matrix<float16_t>>
class XNor_BatchNormDense : public BaseLayer<MAT_CONTAINER, float16_t, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        float momentum;
        float eps=1e-34;
        MAT_CONTAINER<bool> packed_y;
        MAT_CONTAINER<float16_t> y_mean;
        MAT_CONTAINER<float16_t> beta;
        MAT_CONTAINER<float16_t> dbeta;
        MAT_CONTAINER<float16_t> moving_mean;
        MAT_CONTAINER<float16_t> moving_var;
        MAT_CONTAINER<float16_t> var;
        
        //Constructors
        XNor_BatchNormDense(float momentum=0.9);
        //Deconstructors
        ~XNor_BatchNormDense();
        void build(size_t n_cols);
        RET_DTYPE forward(MAT_CONTAINER<float16_t> & x, bool is_training=false);
        RET_DTYPE backprop(MAT_CONTAINER<float16_t> & dy);
        float get_gradient(size_t index) {return dbeta[index];};
        MAT_CONTAINER<float16_t> & get_weight() {return beta;};
};


template <template<typename> class MAT_CONTAINER = Matrix, typename RET_DTYPE=Matrix<float16_t>>
class XNor_BatchNormConv : public BaseLayer<MAT_CONTAINER, float16_t, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        float momentum;
        float eps=1e-34;
        MAT_CONTAINER<bool> packed_y;
        MAT_CONTAINER<float16_t> y_mean;
        MAT_CONTAINER<float16_t> beta;
        MAT_CONTAINER<float16_t> dbeta;
        MAT_CONTAINER<float16_t> moving_mean;
        MAT_CONTAINER<float16_t> moving_var;
        //MAT_CONTAINER<float16_t> mu;
        MAT_CONTAINER<float16_t> var;
        
        //Constructors
        XNor_BatchNormConv(float momentum=0.9);
        //Deconstructors
        ~XNor_BatchNormConv();
        void build(size_t n_cols);
        void compute_output(MAT_CONTAINER<float> & x);
        RET_DTYPE forward(MAT_CONTAINER<float16_t> & x, bool is_training=false);
        RET_DTYPE backprop(MAT_CONTAINER<float16_t> & dy);
        float get_gradient(size_t index) {return dbeta[index];};
        MAT_CONTAINER<float16_t> & get_weight() {return beta;};
};

#endif