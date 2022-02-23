#ifndef conv_utils_ovw_h
#define conv_utils_ovw_h

#include "data_type.h"
#include "conv_utils.h"
#include "arithmetic_ops.h"


template <typename IN_T=float>
void extract_row_from_matrix_4d(Matrix<IN_T> &x, std::vector<IN_T> &row, std::vector<size_t> kernel_shape, std::vector<size_t> x_shape_original, std::vector<size_t> offsets, std::vector<size_t> pad_width={0,0}, float pad_value = 0) {
    // Handle same padding
    // Parameters:
    //   x: input with storage format nhwc
    //   row: variable to 
    //   kernel_shape
    //   offsets: offset of width and height
    //   pad_width_h: pad width of w and h
    
    //std::vector<size_t> x_shape_original = x.shape();
    
    size_t strides_l = 1; // channel
    size_t strides_k = kernel_shape[2]; // width
    size_t strides_j = strides_k * x_shape_original[2]; // height
    
    if (offsets[1] == 0) {
        for (size_t j=0; j<kernel_shape[0]; j++) { // height of kernel
            // If pad in height dimension, set the col to 0
            // But here, it is assumed that when offsets[1] == 0, which
            // is the first row of the nth data, the array is already initialized to 0.
            if (j<pad_width[0]) {
                continue;
            }
            for (size_t l=0; l<kernel_shape[2]; l++) { // output channels of matrix
                for (size_t k=0; k<x_shape_original[2]; k++) {  // width of matrix
                    // Should deal with the left right pad values, but as above, it is 
                    // assumed to be 0.
                    //row[k*strides_k+(i+pad_width[0])*strides_i+j+pad_width[1]] = x(nth_sample, k, nth_row+i, j);
                    // not dealing with padding in left right position
                    row[j*strides_j+k*strides_k+l] = x(offsets[0], j-pad_width[0], k, l);
                }
            }
        }
    } else {
        // shift a row up
        // copy new element to last row
        for (size_t j=0; j<kernel_shape[0]; j++) {
            if (j != (kernel_shape[0]-1)) {
                for (size_t k=0; k<x_shape_original[2]; k++) {  // width of matrix
                    for (size_t l=0; l<kernel_shape[2]; l++) { // output channels of matrix
                        // set element of next row to current row
                        row[j*strides_j+k*strides_k+l] = row[(j+1)*strides_j+k*strides_k+l];
                    }
                }
            } else {
                for (size_t l=0; l<kernel_shape[2]; l++) { // output channels of matrix
                    // set element of next row to current row
                    // set element of matrix to last row
                    for (size_t k=0; k<x_shape_original[2]; k++) {  // width of matrix
                        // Should deal with the left right pad values, but as above, it is 
                        // assumed to be 0.
                        // If n_th row + i > x_shape (in the case of padding, set it to 0.
                        if ((offsets[1]+j-pad_width[0]) >= x_shape_original[2]) {
                            row[j*strides_j+k*strides_k+l] = pad_value;
                        } else {
                            row[j*strides_j+k*strides_k+l] = x(offsets[0], offsets[1]-pad_width[0]+j, k, l);
                        }
                    }
                }
            }
        }
    }
    //printf("x_shape_original: "); print_vec(x_shape_original);
    //printf("kernel_shape: "); print_vec(kernel_shape);
    //printf("offsets: [%d, %d] \n", offsets[0], offsets[1]);
    //printf("pad_width: [%d, %d] \n", pad_width[0], pad_width[1]);
    //printf("pad_value: [%f] \n", pad_value);
    //print_vec(row);
}

template <typename IN_T=float, typename T=size_t>
void _align_memory_before_conv(Matrix<IN_T> &x, std::vector<T> kernel_shape, std::vector<T> output_shape) {
    std::vector<T> x_shape = x.shape();
    std::vector<T> x_strides = x.strides();
    x.resize(output_shape);
    for (int i=(output_shape[0]-1); i>-1; i--) {
        for (int j=(output_shape[1]-1); j>-1; j--) {
            for (int k=(output_shape[2]-1); k>-1; k--) {
                for (int l=(output_shape[3]-1); l>-1; l--) {
                    if (j>=x_shape[1]) {
                        x.set(0, i, j, k, l);
                    } else if (k>=x_shape[2]) {
                        x.set(0, i, j, k, l);
                    } else if (l>=x_shape[3]) {
                        x.set(0, i, j, k, l);
                    } else {
                        x.set(x.m_data[i*x_strides[0]+j*x_strides[1]+k*x_strides[2]+l], i, j, k, l);
                    }
                    //printf("index: (%d, %d, %d, %d)\n", i, j, k, l);
                }
            }
        }
    }
}

template <typename IN_T=float, typename T=size_t>
void _align_memory_after_conv(Matrix<IN_T> &x, std::vector<T> output_shape) {
    std::vector<T> x_shape = x.shape();
    size_t stride_w =  output_shape[3];
    size_t stride_h =  output_shape[2]*stride_w;
    size_t stride_n =  output_shape[1]*stride_h;
    
    for (int i=0; i<output_shape[0]; i++) {
        for (int j=0; j<output_shape[1]; j++) {
            for (int k=0; k<output_shape[2]; k++) {
                for (int l=0; l<output_shape[3]; l++) {
                    x.m_data[i*stride_n+j*stride_h+k*stride_w+l] = x(i, j, k, l);
                    //printf("index: (%d, %d, %d, %d)\n", i, j, k, l);
                }
            }
        }
    }
    x.resize(output_shape);
}


template <typename IN_T=float, typename KER_T=float>
void _convolution_ovw(Matrix<IN_T> &x, Matrix<KER_T> &kernel, std::vector<size_t> output_shape, std::vector<size_t> stride={1,1}, std::vector<size_t> pad_width={0,0}, float pad_value = 0) {
    //printf("-------------------------------------------------------");
    //printf("Input shape: "); print_shape(x);
    //printf("Output shape: "); print_vec(output_shape);
    //printf("kernel shape: "); print_shape(kernel);
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    
    size_t i, j, k, l, i_ker, j_ker, in_ker, out_ker;
    
    size_t strides_l = 1; // channel
    size_t strides_k = kernel_shape[2]; // width
    size_t strides_j = strides_k * x_shape[2]; // height
    
    // resize shape of x if kernel output channel > x output channel
    std::vector<size_t> align_shape = x_shape;
    if (x_shape[3] < kernel_shape[3]) {
        align_shape[3] = kernel_shape[3];
    }
    if (x_shape[1] < output_shape[1]) {
        align_shape[1] = output_shape[1];
    }
    if (x_shape[2] < output_shape[2]) {
        align_shape[2] = output_shape[2];
    }
    if (x_shape != align_shape) {
        _align_memory_before_conv<IN_T>(x, kernel_shape, align_shape);
    }
    
    //printf("aligend_x shape: "); print_shape(x);
    
    // loop of n samples
    for (i=0; i<output_shape[0]; i++) {
        // initialize vector for storing convolution rows.
        std::vector<IN_T> row_copy(x_shape[2]*kernel_shape[0]*kernel_shape[2], pad_value);
        // loop over height of output (row)
        for (j=0; j<output_shape[1]; j++) {
            // copy to row
            extract_row_from_matrix_4d<IN_T>(x, row_copy, kernel_shape, x_shape, {i, j}, pad_width, pad_value);
            // loop over width of output
            for (k=0; k<output_shape[2]; k++) {
                // loop over output channels 
                for (out_ker=0; out_ker<kernel_shape[3]; out_ker++) {
                    float tmp_y = 0;
                    // loop over kernel height
                    for (i_ker=0; i_ker<kernel_shape[0]; i_ker++) {
                        // loop over kernel width
                        for (j_ker=0; j_ker<kernel_shape[1]; j_ker++) {
                            int k_offset = k+j_ker-pad_width[1];
                            bool k_cond = (k_offset >= x_shape[2]) || (k_offset<0);
                            // loop over input channel
                            for (l=0; l<kernel_shape[2]; l++) {
                                if (k_cond) {
                                    tmp_y += _multiply(
                                        pad_value, 
                                        kernel(i_ker, j_ker, l, out_ker)
                                    );
                                } else {
                                    tmp_y += _multiply(
                                        row_copy[i_ker*strides_j+k_offset*strides_k+l], 
                                        kernel(i_ker, j_ker, l, out_ker)
                                    );
                                }
                            } // end loop over input channel
                        } // end loop over kernel width
                    } // end loop over kernel height
                    x.set(tmp_y, i, j, k, out_ker);
                } // end loop over output channels 
            } // end loop over width of output
        } // loop over height of output
    } // end loop of n samples
        
    if (x_shape!=output_shape) {
        _align_memory_after_conv<IN_T>(x, output_shape);
    }
    return;
}

#endif