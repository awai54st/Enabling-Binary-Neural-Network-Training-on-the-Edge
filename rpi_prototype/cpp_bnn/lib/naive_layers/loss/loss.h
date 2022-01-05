#ifndef loss_h
#define loss_h


#include "../../utils/data_type.h"

template <class IN_T=float>
float average_loss(Matrix<IN_T> loss_mat);

template <class IN_T, template<typename> class MAT_CONTAINER = Matrix >
class CrossEntropy {
    public:
        bool is_built = false;
        bool is_training;
        float eps = 1e-34;
        MAT_CONTAINER<IN_T> softmax_output;
        
        //Constructors
        CrossEntropy(void);
        //Deconstructors
        ~CrossEntropy();
        MAT_CONTAINER<IN_T> forward(MAT_CONTAINER<IN_T> & softmax_output, MAT_CONTAINER<IN_T> & labels, bool is_training=true);
        MAT_CONTAINER<IN_T> backprop(MAT_CONTAINER<IN_T> & labels);
};

#endif