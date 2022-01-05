#ifndef activation_layer_h
#define activation_layer_h

#include "../../../utils/data_type.h"
#include "../../../utils/base_layer.h"


template <class PACK_T, template<typename> class MAT_CONTAINER = Matrix, typename WEIGHT_DTYPE=float, typename RET_DTYPE=MAT_CONTAINER<WEIGHT_DTYPE>>
class BinaryActivation : public BaseLayer<MAT_CONTAINER, WEIGHT_DTYPE, RET_DTYPE> {
    public:
        bool is_built = false;
        bool is_training;
        MAT_CONTAINER<PACK_T> packed_y;
        MAT_CONTAINER<WEIGHT_DTYPE> w;
        
        //Constructors
        BinaryActivation(void);
        //Deconstructors
        ~BinaryActivation();
        RET_DTYPE forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training);
        RET_DTYPE backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy);
        float get_gradient(size_t index) {return w[index];};
        MAT_CONTAINER<WEIGHT_DTYPE> & get_weight() {return w;};
};


#endif