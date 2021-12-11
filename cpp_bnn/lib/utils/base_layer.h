#ifndef base_layer_h
#define base_layer_h

#include <memory>

template <template<typename> class MAT_CONTAINER = Matrix, typename WEIGHT_DTYPE=float, typename RET_DTYPE=MAT_CONTAINER<WEIGHT_DTYPE>>
class BaseLayer {
    public:
        virtual RET_DTYPE forward(MAT_CONTAINER<WEIGHT_DTYPE> & x, bool is_training=true) {
            return x;
        };
        virtual RET_DTYPE backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy) {
            return dy;
        };
        virtual void backprop(MAT_CONTAINER<WEIGHT_DTYPE> & dy, MAT_CONTAINER<WEIGHT_DTYPE> & original_input) {
            return;
        };
        virtual void backprop(MAT_CONTAINER<PO2_5bits_t> & dy, MAT_CONTAINER<WEIGHT_DTYPE> & original_input) {
            return;
        };
        virtual void update_trainable_variables(MAT_CONTAINER<float> & dkernel) {
            return;
        };
        virtual float get_gradient(size_t index);
        virtual MAT_CONTAINER<WEIGHT_DTYPE> & get_weight();
};

#endif