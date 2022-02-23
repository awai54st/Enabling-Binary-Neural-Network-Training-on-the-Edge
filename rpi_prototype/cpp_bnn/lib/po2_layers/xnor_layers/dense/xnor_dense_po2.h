#ifndef xnor_dense_po2_h
#define xnor_dense_po2_h


#include "../../../utils/data_type.h"
#include "../../../utils/initialiser.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/base_layer.h"
#include "../../../utils/gradient_quantisation.h"
#include "../../../utils/dot_utils.h"


class XNor_Dense_PO2 : public BaseLayer<Matrix, float16_t, Matrix<float16_t>&> {
    public:
        bool is_built = false;
        bool is_training = false;
        bool is_first_layer = false;
        size_t units;
        int random_seed;
        Matrix<bool> packed_x;
        Matrix<float16_t> kernel;
        float gradient_scale;
        Matrix<bool> dkernel;
        //Constructors
        XNor_Dense_PO2(size_t units, bool is_first_layer = false, int random_seed=0);
        //Deconstructors
        ~XNor_Dense_PO2();
        Matrix<float16_t>& forward(Matrix<float16_t> & x, bool is_training=false);
        Matrix<float16_t>& backprop(Matrix<float16_t> & dy);
        void backprop(Matrix<float16_t> & dy, Matrix<float16_t> & original_input);
        float get_gradient(size_t index) {
            if (fabs(kernel[index]) >= 1) {
                return 0;
            } else {
                if (dkernel[index]) {
                    return 1.0/gradient_scale;
                } else {
                    return -1.0/gradient_scale;
                }
            }
        };
        Matrix<float16_t> & get_weight() {return kernel;};
};

#endif