#ifndef DENSE_H
#define DENSE_H

#include "data_types.h"
#include "initializers.h"
#include "bit_utils.h"
#include "gradient_quantisation.h"
    
//VECTOR_2D_FLOAT float_dot_bit(const VECTOR_2D_FLOAT &x, const VECTOR_2D_T<bool> &y);
//VECTOR_2D_FLOAT bit_dot_float(const VECTOR_2D_T<bool>& x, const VECTOR_2D_FLOAT& y, const float scale=1.0);

class DenseNotFirst_CPP {
    private:
        bool is_built;
    public:
        int units;
        bool is_training;

        // Constructor
        DenseNotFirst_CPP(int units, int random_seed=0, bool is_training=false);
        //Deconstructors
        ~DenseNotFirst_CPP();

        int random_seed;
        VECTOR_2d kernel;
        VECTOR_2D_T<bool> bit_packed_x;
        VECTOR_2D_FLOAT dkernel;

        VECTOR_2d forward(const VECTOR_2d x);
        VECTOR_2D_T<float> backprop(VECTOR_2D_T<float> dy);
};
#endif