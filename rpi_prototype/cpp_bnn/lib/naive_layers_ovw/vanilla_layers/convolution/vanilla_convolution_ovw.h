#ifndef vanilla_convolution_ovw_h
#define vanilla_convolution_ovw_h


#include "../../../utils/data_type.h"
#include "../../../utils/initialiser.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/conv_utils_ovw.h"
#include "../../../utils/base_layer.h"

class Vanilla_Convolution2D_OWH1D : public BaseLayer<Matrix, float, Matrix<float>&> {
    public:
        bool is_built = false;
        bool is_training = false;
        bool is_first_layer = false;
        size_t filters;
        size_t kernel_size;
        int random_seed;
        std::string padding;
        size_t strides = 1;
    
        std::vector<size_t> x_shape;
        Matrix<float> float_x;
        Matrix<float> kernel;
        Matrix<float> dkernel;
        //Constructors
        Vanilla_Convolution2D_OWH1D(size_t filters, size_t kernel_size, std::string padding="valid", bool is_first_layer=false, int random_seed=0);
        //Deconstructors
        ~Vanilla_Convolution2D_OWH1D();
    
        Matrix<float>& forward(Matrix<float> & x, bool is_training=false);
        Matrix<float>& backprop(Matrix<float> & dy);
        void backprop(Matrix<float> &dy, Matrix<float> & original_input);
        float get_gradient(size_t index) {
            if (fabs(kernel[index]) >= 1) {
                return 0;
            } else {
                return dkernel[index];
            }
        };
        Matrix<float> & get_weight() {return kernel;};
};

#endif