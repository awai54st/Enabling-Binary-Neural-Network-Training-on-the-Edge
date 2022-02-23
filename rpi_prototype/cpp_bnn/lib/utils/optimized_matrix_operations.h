#ifndef optimized_matrix_operations_h
#define optimized_matrix_operations_h

#include "data_type.h"
#include "conv_utils_ovw.h"
#include "dot_utils.h"
#include <cmath>
#include <algorithm>
extern "C"
{
    #include <cblas.h>
}

// https://www.netlib.org/blas/cblas.h
// https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm/
// https://www.ibm.com/docs/en/zos/2.3.0?topic=uatlasal-examples-compiling-linking-running-simple-matrix-multiplication-atlas-program

template <typename T=float>
void matmul(Matrix<float> & A, Matrix<float> & B, Matrix<float> & C, float scale=1.0) {
    // std::vector<int8_t> A(rowsA * common, 1); 
    // std::vector<float> B(common * colsB, 1);
    // std::vector<float> C(rowsA * colsB);
    int rowsA = A.shape()[0];
    int common = A.shape()[1];
    int colsB = B.shape()[1];
    
    //gemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); // col-major
    //gemm('N', 'N', n, m, k, alpha, b, ldb, a, lda, beta, c, ldc); // row-major
    // sgemm c = alpha*AB + beta*C
    // sgemm c = alpha*AB^T + beta*C
    // sgemm c = alpha*A^TB + beta*C
    // sgemm c = alpha*A^TB^T + beta*C
    // sgemm(
    //    data layout, transa, transb, 
    //    C rows, C cols, common column, 
    //    alpha, matrix A, rows of a (lda), 
    //    matrix B, rows of b (ldb),
    //    beta, matrix C, rows of matrix c)
    // c/c++ a[something][LDA]
    // fotran a[LDA][something]
    
    cblas_sgemm(
        CblasRowMajor,CblasNoTrans,CblasNoTrans, 
        rowsA, colsB, common, 
        scale, A.data(), common,
        B.data(), colsB,
        0.0, C.data(), colsB);
};

template <typename T=float>
void matmul_transa(Matrix<float> & A, Matrix<float> & B, Matrix<float> & C, float scale=1.0) {
    int rowsA = A.shape()[1];
    int common = A.shape()[0];
    int colsB = B.shape()[1];
    
    cblas_sgemm(
        CblasRowMajor,CblasTrans,CblasNoTrans, 
        rowsA, colsB, common,
        scale, A.data(), rowsA,
        B.data(), colsB,
        0.0, C.data(), colsB);
};

template <typename T=float>
void matmul_transb(Matrix<float> & A, Matrix<float> & B, Matrix<float> & C, float scale=1.0) {
    int rowsA = A.shape()[0];
    int common = A.shape()[1];
    int colsB = B.shape()[0];
    
    cblas_sgemm(
        CblasRowMajor,CblasNoTrans,CblasTrans, 
        rowsA, colsB, common, 
        scale, A.data(), common,
        B.data(), common,
        0.0, C.data(), colsB);
};


template <typename T=size_t>
T get_number_of_pad_opt(T& input_shape, T& output_shape, T& kernel_shape, T& stride) {
    return T(1.0/2.0*((output_shape-1)*stride+kernel_shape-input_shape));
};



template <typename IN_T=float>
float get_val(IN_T a) {
    return a;
}

template <typename IN_T=bool>
float get_val(bool a) {
    if (a) {
        return 1.0;
    } else {
        return -1.0;
    }
}

template <typename T=float>
void im2col(Matrix<T> &im, size_t index, Matrix<float> &col, std::vector<size_t> im_shape, std::vector<size_t> kernel_shape, std::vector<size_t> output_shape, float pad_val = 0.0, size_t stride = 1) {
    //std::vector<size_t> im_shape = im.shape();
    std::vector<size_t> col_shape = col.shape();
    
    size_t stride_l = kernel_shape[0]*kernel_shape[1];
    size_t stride_i = kernel_shape[1];
    
    // check if there is padding or not
    size_t i_ker_offset = get_number_of_pad_opt<size_t>(im_shape[1], output_shape[1], kernel_shape[0], stride);
    size_t j_ker_offset = get_number_of_pad_opt<size_t>(im_shape[2], output_shape[2], kernel_shape[1], stride);
    size_t j, k, i_col, l, i_ker, j_ker;
    
    // loop over height of im
    // TODO: add strides
    for (j=0; j<output_shape[1]; j++) {
        // loop over width of im
        // TODO: add strides
        for (k=0; k<output_shape[2]; k++) {
                float tmp_y = 0;
                // loop over input channel
            for (l=0; l<kernel_shape[2]; l++) {
                // loop over kernel width
                for (j_ker=0; j_ker<kernel_shape[1]; j_ker++) {
                    int k_offset = k+j_ker-j_ker_offset;
                    bool k_cond = (k_offset >= im_shape[2]) || (k_offset<0);
                    // loop over kernel height
                    for (i_ker=0; i_ker<kernel_shape[0]; i_ker++) {
                        int j_offset = j+i_ker-i_ker_offset;
                        bool j_cond = (j_offset >= im_shape[1]) || (j_offset<0);
                        if (j_cond || k_cond) {
                            col.set(pad_val, j*output_shape[2]+k, j_ker+i_ker*stride_i+l*stride_l, 0, 0);
                        } else {
                            col.set(get_val(im(index, j_offset, k_offset, l)), j*output_shape[2]+k, j_ker+i_ker*stride_i+l*stride_l, 0, 0);
                        }
                        //printf("offsets: %d, %d\n", i_col, i_ker+j_ker*stride_j+l*stride_l);
                    } // end loop over kernel width
                } // end loop over kernel height
            } // end loop over input channel
        } // end loop over width of im
    } // loop over height of im
}


template <typename T=float>
void col2im(Matrix<float> &col, Matrix<T> &im, std::vector<size_t> im_shape, size_t index) {
    std::vector<size_t> col_shape = col.shape();
    //std::vector<size_t> im_shape = im.shape();
    
    size_t stride_k = im_shape[2];
    
    size_t out_ker, j, k, l;
    
    // loop over height of im
    for (j=0; j<im_shape[1]; j++) {
        // loop over width of im
        // TODO: add strides
        for (k=0; k<im_shape[2]; k++) {
            // loop over output channel of im
            for (l=0; l<im_shape[3]; l++) {
                im.set(col(j*stride_k+k, l), index, j, k, l);
            } // loop over output channel of im
        } // end loop over width of im
    } // loop over height of im
}

template <typename T=bool>
void col2im(Matrix<float> &col, Matrix<bool> &im, std::vector<size_t> im_shape, size_t index) {
    std::vector<size_t> col_shape = col.shape();
    //std::vector<size_t> im_shape = im.shape();
    
    size_t stride_k = im_shape[2];
    
    size_t out_ker, j, k, l;
    
    // loop over height of im
    for (j=0; j<im_shape[1]; j++) {
        // loop over width of im
        // TODO: add strides
        for (k=0; k<im_shape[2]; k++) {
            // loop over output channel of im
            for (l=0; l<im_shape[3]; l++) {
                if (col(j*stride_k+k, l) >= 0) {
                    im.set(1, index, j, k, l);
                } else {
                    im.set(0, index, j, k, l);
                }
            } // loop over output channel of im
        } // end loop over width of im
    } // loop over height of im
}

template <typename T=float>
void kernel2col(Matrix<T> &kernel, Matrix<float> &col) {
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> col_shape = col.shape();
    
    size_t stride_j = kernel_shape[0];
    size_t stride_k = kernel_shape[1]*stride_j;
        
    size_t i, j, k, l;
    
    // loop over height of kernel
    for (i=0; i<kernel_shape[0]; i++) {
        // loop over width of kernel
        for (j=0; j<kernel_shape[1]; j++) {
            // loop over input channel
            for (k=0; k<kernel_shape[2]; k++) {
                // loop over output channel
                for (l=0; l<kernel_shape[3]; l++) {
                    col.set(get_val(kernel(i, j, k, l)), i+j*stride_j+k*stride_k, l, 0, 0);
                }
            }
        }
    }
}


// https://www.notion.so/Optimizing-CNN-implementation-through-im2col-1c63e789783245d9a734771f00f3f8ba
template <typename IN_T=float, typename KER_T=bool, typename OUT_T=float>
void convolution(Matrix<IN_T> &x, Matrix<KER_T> &kernel, Matrix<OUT_T> &output, size_t stride=1, float pad_val=0) {
    // Get kernel shape, input shape, output shape
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> output_shape = output.shape();
    
    std::vector<size_t> image_col_shape = {output_shape[1]*output_shape[2], kernel_shape[0]*kernel_shape[1]*kernel_shape[2]};
    
    // kernel to col (only need to copy once)
    Matrix<float> kernel_col({image_col_shape[1], kernel_shape[3]});
    kernel2col<KER_T>(kernel, kernel_col);
        
    size_t i;
    
    // loop of n samples
    for (i=0; i<output_shape[0]; i++) {
        // initialize temporary matrix
        Matrix<float> im_col(image_col_shape);
        Matrix<float> tmp_output({image_col_shape[0], kernel_shape[3]}, 0);
        // copy matrix (im2col), which is done for every n samples in the dataset
        im2col<IN_T>(x, i, im_col, x_shape, kernel_shape, output_shape, pad_val, stride);
        
        // matrix multiplication
        matmul(im_col, kernel_col, tmp_output);
        col2im<OUT_T>(tmp_output, output, output_shape, i);
    } // end loop of n samples
}


template <typename IN_T=float, typename KER_T=bool>
void convolution(Matrix<IN_T> &x, Matrix<KER_T> &kernel, std::vector<size_t> output_shape, size_t stride=1, float pad_val=0) {
    // Get kernel shape, input shape, output shape
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    
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
        _align_memory_before_conv(x, kernel_shape, align_shape);
    }
    
    std::vector<size_t> image_col_shape = {output_shape[1]*output_shape[2], kernel_shape[0]*kernel_shape[1]*kernel_shape[2]};
    
    // kernel to col (only need to copy once)
    Matrix<float> kernel_col({image_col_shape[1], kernel_shape[3]});
    kernel2col<KER_T>(kernel, kernel_col);
        
    size_t i;
    
    // loop of n samples
    for (i=0; i<output_shape[0]; i++) {
        // initialize temporary matrix
        Matrix<float> im_col(image_col_shape);
        Matrix<float> tmp_output({image_col_shape[0], kernel_shape[3]}, 0);
        // copy matrix (im2col), which is done for every n samples in the dataset
        im2col<IN_T>(x, i, im_col, x_shape, kernel_shape, output_shape, pad_val, stride);
        
        // matrix multiplication
        matmul(im_col, kernel_col, tmp_output);
        col2im<IN_T>(tmp_output, x, output_shape, i);
    } // end loop of n samples
        
    if (x_shape!=output_shape) {
        _align_memory_after_conv(x, output_shape);
    }
}


// Dot product --------------------------------------------------------------------------
template <typename IN_T=float, typename OUT_T=float>
void kernel_formatting(Matrix<IN_T> &x, Matrix<OUT_T> &x_copy) {
    std::vector<size_t> x_shape = x.shape();
    for (size_t i=0; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            if (x(i,j)) {
                x_copy.set(1.0, i, j);
            } else {
                x_copy.set(-1.0, i, j);
            }
        }
    }
}


template <typename IN_T=float, typename OUT_T=float>
void copy_mat_to_vec_dot(Matrix<IN_T> &x, Matrix<OUT_T> &x_copy, std::vector<size_t> x_shape, size_t row_offset = 0) {
    //std::vector<size_t> x_shape = x.shape();
    for (size_t i=row_offset; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            x_copy.set(x(i,j), i, j);
        }
    }
}

template <typename IN_T=float>
void copy_vec_to_mat_dot(Matrix<IN_T> &x, Matrix<float> &x_copy, std::vector<size_t> x_shape, size_t row_offset = 0) {
    //std::vector<size_t> x_shape = x.shape();
    for (size_t i=0; i<x_shape[0]; i++) {
        for (size_t j=0; j<x_shape[1]; j++) {
            x.set(x_copy(i,j), i+row_offset, j);
        }
    }
}

template <typename IN_T=float, typename KER_T=bool>
void dot_opt(Matrix<IN_T> &x, Matrix<KER_T> &kernel) {
    // Get kernel shape, input shape, output shape
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    size_t copy_count = x_shape[0];
    if (x_shape[0]>4) {
        size_t copy_count = round(float(x_shape[0])/4.0);
    }
    
    Matrix<float> kernel_float(kernel_shape);
    kernel_formatting(kernel, kernel_float);
    
    if (x_shape[1] < kernel_shape[1]) {
        //printf("align memory all");
        _align_memory_before_dot<IN_T>(x, kernel_shape);
    }
    
    for (size_t i = 0; i<x_shape[0]; i+=copy_count ) {
        size_t n_rows = std::min((i+1)*copy_count, x_shape[0]);
        
        Matrix<float> x_float({n_rows, x_shape[1]});
        Matrix<float> out_float({n_rows, kernel_shape[1]}, 0);
        copy_mat_to_vec_dot(x, x_float, x_shape, i);
        
        
        matmul(x_float, kernel_float, out_float, 1.0);
        copy_vec_to_mat_dot(x, out_float, {n_rows, kernel_shape[1]}, i);
    }
    
    if (x_shape[1] > kernel_shape[1]) {
        //printf("re align memory all");
        _align_memory_after_dot<IN_T>(x, {x_shape[0], kernel_shape[1]});
    }
}

template <typename IN_T=float, typename KER_T=bool, typename OUT_T=float>
void dot_opt(Matrix<IN_T> &x, Matrix<KER_T> &kernel, Matrix<OUT_T> &output) {
    // Get kernel shape, input shape, output shape
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> out_shape = output.shape();
    size_t copy_count = x_shape[0];
    if (x_shape[0]>4) {
        size_t copy_count = round(float(x_shape[0])/4.0);
    }
    
    Matrix<float> kernel_float(kernel_shape);
    kernel_formatting(kernel, kernel_float);
    
    for (size_t i = 0; i<x_shape[0]; i+=copy_count ) {
        size_t n_rows = std::min((i+1)*copy_count, x_shape[0]);
        
        Matrix<float> x_float({n_rows, x_shape[1]});
        Matrix<float> out_float({n_rows, out_shape[1]}, 0);
        copy_mat_to_vec_dot(x, x_float, x_shape, i);
        //copy_mat_to_vec_dot(output, out_float, i);
        
        matmul(x_float, kernel_float, out_float, 1.0);
        copy_vec_to_mat_dot(output, out_float, out_shape, i);
    }
}


#endif