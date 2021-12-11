#include "loss.h"
#include <cmath>

template class CrossEntropy<float, Matrix>;
template class CrossEntropy<float16_t, Matrix>;
//template class CrossEntropy<float, Matrix2D>;

template float average_loss<float>(Matrix<float> loss_mat);
template float average_loss<float16_t>(Matrix<float16_t> loss_mat);


template <class IN_T, template<typename> class MAT_CONTAINER>
CrossEntropy<IN_T, MAT_CONTAINER>::CrossEntropy() {};

template <class IN_T, template<typename> class MAT_CONTAINER>
CrossEntropy<IN_T, MAT_CONTAINER>::~CrossEntropy() {};

template <class IN_T, template<typename> class MAT_CONTAINER>
MAT_CONTAINER<IN_T> CrossEntropy<IN_T, MAT_CONTAINER>::forward(MAT_CONTAINER<IN_T> & predictions, MAT_CONTAINER<IN_T> & labels, bool is_training) {
    const std::vector<size_t> prediction_shape = predictions.shape();
    softmax_output = predictions;
    // label - log(sofmax_output)
    for (size_t i=0; i<prediction_shape[0]; i++) {
        for (size_t j=0; j<prediction_shape[1]; j++) {
            predictions.set(labels(i,j)*-log(predictions(i,j)+eps), i,j);
        }
    }
    
    // CE output (Assume only one ground truth class for each row)
    MAT_CONTAINER<IN_T> output({prediction_shape[0], 1});
    for (size_t i=0; i<prediction_shape[0]; i++) {
        float tmp_output = 0;
        for (size_t j=0; j<prediction_shape[1]; j++) {
            tmp_output += predictions(i,j);
        }
        output.set(tmp_output, i, 0);
    }
    
    return output;
};

template <class IN_T, template<typename> class MAT_CONTAINER>
MAT_CONTAINER<IN_T> CrossEntropy<IN_T, MAT_CONTAINER>::backprop(MAT_CONTAINER<IN_T> & labels) {
    const std::vector<size_t> labels_shape = labels.shape();
    MAT_CONTAINER<IN_T> dy(labels_shape);
    
    // CE Grad
    for (size_t i=0; i<labels_shape[0]; i++) {
        for (size_t j=0; j<labels_shape[1]; j++) {
            if (labels(i,j) == 1.0) {
                dy.set(-1/(softmax_output(i,j)+eps), i, j);
            } else {
                dy.set(0, i, j);
            }
        }
    }
    
    // Row sum of CE*sofmax_output
    MAT_CONTAINER<float> row_sum({labels_shape[0]});
    
    for (size_t i=0; i<labels_shape[0]; i++) {
        for (size_t j=0; j<labels_shape[1]; j++) {
            row_sum[i] += (dy(i, j) * softmax_output(i,j));
        }
    }
    
    // Softmax Grad
    for (size_t i=0; i<labels_shape[0]; i++) {
        for (size_t j=0; j<labels_shape[1]; j++) {
            dy.set(softmax_output(i,j)*(dy(i, j)-row_sum[i]), i, j);
        }
    }
    
    return dy;
};


template <class IN_T=float>
float average_loss(Matrix<IN_T> loss_mat) {
    size_t loss_mat_size = loss_mat.size();
    float loss = 0;
    for (size_t i=0; i<loss_mat_size; i++) {
        loss += loss_mat[i];
    }
    loss /= loss_mat_size;
    return loss;
};