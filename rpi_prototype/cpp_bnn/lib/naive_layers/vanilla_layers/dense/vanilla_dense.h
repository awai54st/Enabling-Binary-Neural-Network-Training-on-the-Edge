#ifndef vanilla_dense_h
#define vanilla_dense_h

#include "../../../utils/data_type.h"
#include "../../../utils/initialiser.h"
#include "../../../utils/bit_packing.h"
#include "../../../utils/dot_utils.h"
#include "../../../utils/base_layer.h"

class Vanilla_BNNDense : public BaseLayer<Matrix> {
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
        Vanilla_BNNDense(size_t units, bool is_first_layer = false, int random_seed=0);
        //Deconstructors
        ~Vanilla_BNNDense();
        Matrix<float> forward(Matrix<float> & x, bool is_training=false);
        Matrix<float> backprop(Matrix<float> & dy);
        void backprop(Matrix<float> & dy, Matrix<float> & original_input);
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