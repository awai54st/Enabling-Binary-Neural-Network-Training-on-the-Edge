#ifndef pooling_layers_h
#define pooling_layers_h

#include "../../../utils/data_type.h"
#include "../../../utils/base_layer.h"

template <class PACK_T, template<typename> class MAT_CONTAINER = Matrix, typename WEIGHT_DTYPE=float, typename RET_DTYPE=MAT_CONTAINER<WEIGHT_DTYPE>>
class MaxPooling : public BaseLayer<MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        int stride;
        int kernel_size;
        MAT_CONTAINER<PACK_T> packed_x;
        std::vector<size_t> x_shape;
        MAT_CONTAINER<WEIGHT_DTYPE> w;
        
        //Constructors
        MaxPooling(int kernel_size = 2, int stride=2);
        //Deconstructors
        ~MaxPooling();
        RET_DTYPE forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training=false);
        RET_DTYPE backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy);
        float get_gradient(size_t index) {return w[index];};
        MAT_CONTAINER<WEIGHT_DTYPE> & get_weight() {return w;};
};

#endif