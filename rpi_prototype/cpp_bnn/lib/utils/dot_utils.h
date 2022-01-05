#ifndef dot_utils_h
#define dot_utils_h

#include "data_type.h"
#include "arithmetic_ops.h"

template <typename IN_T=float, typename KER_T=float, typename OUT_T=float>
void dot(Matrix<IN_T> &x, Matrix<KER_T> &kernel, Matrix<OUT_T> &output){
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> kernel_shape = kernel.shape();
    for (int i=0; i<x_shape[0]; i++) {
        //std::vector<float> tmp_y(kernel_shape[1], 0);
        for (int j=0; j<kernel_shape[1]; j++) {
            float tmp_y = 0;
            for (int common=0; common<kernel_shape[0]; common++) {
                tmp_y += _multiply(x(i, common), kernel(common, j));
            }
            //output.m_data[(i*out_n_cols)+j] = tmp_y;
            //tmp_y[j] = tmp_ele;
            output.set(tmp_y, i, j);
        }
        //dy.m_data[i].swap(tmp_y);
    }
    return;
};


template <typename IN_T=float, typename KER_T=float, typename OUT_T=float>
void dot(Matrix<IN_T> &x, Matrix<KER_T> &kernel, Matrix<bool> &output){
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> kernel_shape = kernel.shape();
    for (int i=0; i<x_shape[0]; i++) {
        //std::vector<float> tmp_y(kernel_shape[1], 0);
        for (int j=0; j<kernel_shape[1]; j++) {
            float tmp_y = 0;
            for (int common=0; common<kernel_shape[0]; common++) {
                tmp_y += _multiply(x(i, common), kernel(common, j));
            }
            //output.m_data[(i*out_n_cols)+j] = tmp_y;
            //tmp_y[j] = tmp_ele;
            if (tmp_y >= 0) {
                output.set(1, i, j);
            } else {
                output.set(0, i, j);
            }
        }
        //dy.m_data[i].swap(tmp_y);
    }
    return;
};
/*
template <template<typename> class MAT_CONTAINER = Matrix>
void _copy_row_to_vec(MAT_CONTAINER<float> &x, size_t row_idx, std::vector<float> &row_vec) {
    for (size_t i=0; i<row_vec.size(); i++) {
        row_vec[i] = x(row_idx, i);
    }
}

template <template<typename> class MAT_CONTAINER = Matrix>
void _copy_vec_to_row(MAT_CONTAINER<float> &x, size_t row_idx, std::vector<float> &row_vec) {
    size_t vec_size = row_vec.size();
    for (size_t i=0; i<row_vec.size(); i++) {
        x[row_idx*vec_size+i] = row_vec[i];
    }
}

template <template<typename> class MAT_CONTAINER = Matrix>
void _overflow_handling(MAT_CONTAINER<float> &x, size_t row_idx, std::vector<size_t> &offsets, std::vector<float> &row_x, std::vector<float> &y) {
    // Get input shape
    size_t x_col = x.shape()[1];
    
    // copy to matrix
    if (offsets[1] == 0) {
        // last index == 0, no output is written to matrix OR
        // length of output < column of matrix
        for (size_t i=0; i<x_col; i++) {
            x.set(y[i], row_idx, i);
        }
        
        // copy to overflow
        for (size_t i=x_col; i<y.size(); i++) {
            x[offsets[0]+offsets[1]] = y[i];
            offsets[1]++;
        }
    } else if (offsets[1] < x_col) {
        // copy overflow back to matrix
        for (size_t i=0; i<offsets[1]; i++) {
            x.set(x[offsets[0]+i], row_idx, i);
        }

        // move y to remaining col
        size_t offset_reduction_count=offsets[1];
        for (; offset_reduction_count<(x_col-offsets[1]); offset_reduction_count++) {
            x.set(y[offset_reduction_count], row_idx, offset_reduction_count);
        }
        offsets[1] = 0; // update last indexverflow

        // move remaining y to overflow
        for (size_t i=offset_reduction_count; i<y.size(); i++) {
            x[offsets[0]+offsets[1]] = y[i];
            offsets[1] ++; // update last index
        }
    } else {
        // copy overflow back to matrix
        for (size_t i=0; i<x_col; i++) {
            x.set(x[offsets[0]+i], row_idx, i);
        }

        // move overflow forward
        size_t offset_reduction_count;
        for (size_t i=x_col; i<offsets[1]; i++) {
            x[offsets[0]+i-x_col] = x[offsets[0]+i];
        }
        offsets[1] -= x_col; // update last index

        //copy y to overflow
        for (size_t i=x_col; i<y.size(); i++) {
            x[offsets[0]+offsets[1]] = y[i];
            offsets[1] ++; // update last index
        }
    }
}

template <typename T=bool>
void float_dot_bit(Matrix<float> &x, Matrix<T> &kernel){
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> kernel_shape = kernel.shape();
    if (x_shape[1] >= kernel_shape[1]) {
        for (int i=0; i<x_shape[0]; i++) {
            // copy row of the matrix to vector
            std::vector<float> x_tmp(kernel_shape[1], 0);
            _copy_row_to_vec(x, i, x_tmp);
            // initialize vector for result storage
            std::vector<float> tmp_y_vec(kernel_shape[1], 0);
            for (int j=0; j<kernel_shape[1]; j++) {
                float tmp_y = 0;
                for (int common=0; common<kernel_shape[0]; common++) {
                    if (kernel(common, j)) {
                        tmp_y += x_tmp[common];
                    } else {
                        tmp_y -= x_tmp[common];
                    }
                }
                tmp_y_vec[j] = tmp_y;
            }
            _copy_vec_to_row(x, i, tmp_y_vec);//output.m_data[(i*out_n_cols)+j] = tmp_y;
        }
        x.resize({x_shape[0], kernel_shape[1]});
    } else {
        std::vector<size_t> offsets(2, 0);
        offsets[0] = x.size(); // original size
        offsets[1] = 0; // last 
        x.m_data.resize(x_shape[0]*kernel_shape[1], 0);
        for (int i=0; i<x_shape[0]; i++) {
            // copy row of the matrix to vector
            // initialize vector for result storage
            // copy to vec
            std::vector<float> x_tmp(kernel_shape[1], 0);
            _copy_row_to_vec(x, i, x_tmp);
            std::vector<float> tmp_y_vec(kernel_shape[1], 0);
            for (int j=0; j<kernel_shape[1]; j++) {
                float tmp_y = 0;
                for (int common=0; common<kernel_shape[0]; common++) {
                    if (kernel(common, j)) {
                        tmp_y += x_tmp[common];
                    } else {
                        tmp_y -= x_tmp[common];
                    }
                }
                tmp_y_vec[j] = tmp_y;
            }
            _overflow_handling(x, i, offsets, x_tmp, tmp_y_vec);//output.m_data[(i*out_n_cols)+j] = tmp_y;);
        }
        x.resize({x_shape[0], kernel_shape[1]});
    }
    return;
};
//*/
///*
template <typename T=float16_t, template<typename> class MAT_CONTAINER = Matrix>
void _copy_row_to_vec(MAT_CONTAINER<T> &x, size_t row_idx, std::vector<T> &row_vec) {
    for (size_t i=0; i<row_vec.size(); i++) {
        row_vec[i] = x(row_idx, i);
    }
}

template <typename T=float16_t>
void _align_memory_before_dot(Matrix<T> &x, std::vector<size_t> &kernel_shape) {
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> x_strides = x.strides();
    
    x.resize({x_shape[0], kernel_shape[1]});
    
    for (int i=x_shape[0]-1; i>-1; i--) {
        for (int j=kernel_shape[1]-1; j>-1; j--) {
            if (j>=x_shape[1]) {
                x.set(0,i, j);
            } else {
                x.set(x.m_data[i*x_strides[0]+j],i, j);
            }
        }
    }
}

template <typename T=float16_t>
void _align_memory_after_dot(Matrix<T> &x, std::vector<size_t> output_shape) {
    std::vector<size_t> x_shape = x.shape();
    
    size_t stride_n =  output_shape[1];
    
    for (int i=0; i<output_shape[0]; i++) {
        for (int j=0; j<output_shape[1]; j++) {
            x.m_data[i*stride_n+j] = x(i, j);
        }
    }
    x.resize(output_shape);
}



template <typename IN_T=float16_t, typename KER_T=float16_t>
void float_dot_bit(Matrix<IN_T> &x, Matrix<KER_T> &kernel){
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> kernel_shape = kernel.shape();
    if (x_shape[1] < kernel_shape[1]) {
        //printf("align memory all");
        _align_memory_before_dot<IN_T>(x, kernel_shape);
    }
    // printf("align memory x: "); print_mat(x);
    
    for (int i=0; i<x_shape[0]; i++) {
        // copy row of the matrix to vector
        std::vector<IN_T> x_tmp(x_shape[1], 0);
        _copy_row_to_vec(x, i, x_tmp);
        //print_vec(x_tmp);
        for (int j=0; j<kernel_shape[1]; j++) {
            float tmp_y = 0;
            for (int common=0; common<kernel_shape[0]; common++) {
                if (kernel(common, j)) {
                    tmp_y += x_tmp[common];
                } else {
                    tmp_y -= x_tmp[common];
                }
            }
            x.set(tmp_y, i, j);
        }//output.m_data[(i*out_n_cols)+j] = tmp_y;
    }
    if (x_shape[1] > kernel_shape[1]) {
        //printf("re align memory all");
        _align_memory_after_dot<IN_T>(x, {x_shape[0], kernel_shape[1]});
    }
    
    return;
};
//*/
template <typename T=bool>
void float_dot_bit_ovw(Matrix2D<float> &dy, Matrix2D<T> &kernel){
    // Args:
    //     x (m, n)
    //     y (c, bitset(n)): original shape(n, c) bit packed to (c, bitset(n))
    // 
    // Return:
    //     out (m, c)
    std::vector<size_t> dy_shape = dy.shape();
    std::vector<size_t> kernel_shape = kernel.shape();
    for (int i=0; i<dy_shape[0]; i++) {
        std::vector<float> tmp_y(kernel_shape[1], 0);
        for (int j=0; j<kernel_shape[1]; j++) {
            float tmp_ele = 0;
            for (int common=0; common<kernel_shape[0]; common++) {
                if (kernel(common, j)) {
                    tmp_ele += dy(i, common);
                } else {
                    tmp_ele -= dy(i, common);
                }
            }
            //output.m_data[(i*out_n_cols)+j] = tmp_y;
            tmp_y[j] = tmp_ele;
        }
        //dy.m_data[i].resize(kernel_shape[1]);
        dy.m_data[i].swap(tmp_y);
        //dy.m_data[i] = tmp_y;
    }
    dy.resize({dy_shape[0], kernel_shape[1]});
    return;
};


#endif