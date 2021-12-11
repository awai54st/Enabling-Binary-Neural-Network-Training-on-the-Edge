#ifndef opt_xnor_convolution_h
#define opt_xnor_convolution_h


#include "../../../utils/data_type.h"
#include "../../../utils/initialiser.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/gradient_quantisation.h"
#include "../../../utils/optimized_matrix_operations.h"
#include "../../../utils/base_layer.h"


class OPT_XNor_Convolution2D : public BaseLayer<Matrix, float16_t> {
    public:
        bool is_built = false;
        bool is_training = false;
        bool is_first_layer = false;
        size_t filters;
        size_t kernel_size;
        size_t strides = 1;
        int random_seed;
        std::string padding;
    
        Matrix<bool> packed_x;
        std::vector<size_t> x_shape;
        Matrix<float> float_x;
        Matrix<float16_t> kernel;
        float gradient_scale;
        Matrix<bool> dkernel;
        //Constructors
        OPT_XNor_Convolution2D(size_t filters, size_t kernel_size, std::string padding="valid", bool is_first_layer=false, int random_seed=0);
        //Deconstructors
        ~OPT_XNor_Convolution2D();
    
        Matrix<float16_t> forward(Matrix<float16_t> & x, bool is_training=false);
        Matrix<float16_t> backprop(Matrix<float16_t> & dy);
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