#ifndef vanilla_dense_ovw_h
#define vanilla_dense_ovw_h

#include "../../../utils/data_type.h"
#include "../../../utils/initialiser.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/dot_utils.h"
#include "../../../utils/base_layer.h"

class Vanilla_Dense_OVW : public BaseLayer<Matrix2D> {
    public:
        bool is_built = false;
        bool is_training = false;
        bool is_first_layer = false;
        size_t units;
        int random_seed;
        Matrix2D<float> float_x;
        Matrix2D<float> kernel;
        Matrix2D<float> dkernel;
        //Constructorss
        Vanilla_Dense_OVW(size_t units, bool is_first_layer = false, int random_seed=0);
        //Deconstructors
        ~Vanilla_Dense_OVW();
        Matrix2D<float> forward(Matrix2D<float> & x, bool is_training=false);
        Matrix2D<float> backprop(Matrix2D<float> & dy);
        void backprop(Matrix2D<float> & dy, Matrix2D<float> &original_input);
        float get_gradient(size_t index) {
            if (fabs(kernel[index]) >= 1) {
                return 0;
            } else {
                return dkernel[index];
            }
        };
        Matrix2D<float> & get_weight() {return kernel;};
};

class Vanilla_Dense_OVW1D : public BaseLayer<Matrix, float, Matrix<float>&> {
    public:
        bool is_built = false;
        bool is_training = false;
        bool is_first_layer = false;
        size_t units;
        int random_seed;
        Matrix<float> float_x;
        Matrix<float> kernel;
        Matrix<float> dkernel;
        //Constructorss
        Vanilla_Dense_OVW1D(size_t units, bool is_first_layer = false, int random_seed=0);
        //Deconstructors
        ~Vanilla_Dense_OVW1D();
        Matrix<float>& forward(Matrix<float> & x, bool is_training=false);
        void backprop(Matrix<float> & dy, Matrix<float> & original_input);
        Matrix<float>& backprop(Matrix<float> & dy);
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
        
//Matrix2D<float> forward(const Matrix2D<bool> & x, bool is_training=false);