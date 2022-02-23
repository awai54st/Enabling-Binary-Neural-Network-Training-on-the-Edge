#ifndef softmax_h
#define softmax_h


#include "../../../utils/data_type.h"
#include "../../../utils/base_layer.h"

template <typename WEIGHT_DTYPE=float, template<typename> class MAT_CONTAINER = Matrix, typename RET_DTYPE=MAT_CONTAINER<WEIGHT_DTYPE>>
class Softmax : public BaseLayer<MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        MAT_CONTAINER<WEIGHT_DTYPE> w;
        float eps=1e-34;
        
        //Constructors
        Softmax(void);
        //Deconstructors
        ~Softmax();
        RET_DTYPE forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training=true);
        RET_DTYPE backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy);
        float get_gradient(size_t index) {return w[index];};
        MAT_CONTAINER<WEIGHT_DTYPE> & get_weight() {return w;};
};

#endif