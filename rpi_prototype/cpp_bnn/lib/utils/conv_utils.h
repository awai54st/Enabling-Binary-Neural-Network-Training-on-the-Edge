#ifndef conv_utils_h
#define conv_utils_h

#include "data_type.h"
#include "arithmetic_ops.h"


template <typename T=size_t>
T get_number_of_pad(T& input_shape, T& output_shape, T& kernel_shape, T stride) {
    return 1.0/2*((output_shape-1)*stride+kernel_shape-input_shape);
};

template <typename T=size_t>
std::vector<T> get_output_shape(std::vector<T>& input_shape, std::vector<T>& kernel_shape, std::string& padding="valid", T stride=1) {
    if (padding == "valid") {
        size_t out_h = (input_shape[1]-kernel_shape[0])/stride +1;
        size_t out_w = (input_shape[2]-kernel_shape[1])/stride +1;
        return {input_shape[0], out_h, out_w, kernel_shape[3]};
    } else {
        return {input_shape[0], input_shape[1], input_shape[2], kernel_shape[3]};
    }
};

/*
template <typename T>
std::vector<T> get_output_shape(std::vector<T>& input_shape, std::vector<T>& kernel_shape, T& pad, T& stride=1) {
    size_t out_h = (input_shape[1]+2*pad-(kernel_shape[0]-1)-1)/stride +1;
    size_t out_w = (input_shape[2]+2*pad-(kernel_shape[1]-1)-1)/stride +1;
    return {input_shape[0], out_h, out_w, kernel_shape[3]};
};
*/

template <typename T=size_t>
std::vector<T> calc_output_shape(std::vector<size_t>& input_shape, std::vector<size_t>& kernel_shape, std::vector<size_t> &stride={1,1}, std::vector<size_t> &pad_width={0,0}) {
    size_t out_h = (input_shape[1]-kernel_shape[0]+pad_width[0]+stride[0])/stride[0];
    size_t out_w = (input_shape[2]-kernel_shape[1]+pad_width[1]+stride[1])/stride[1];
    return {input_shape[0], out_h, out_w, kernel_shape[3]};
};


template <typename IN_T=float, typename KER_T=float, typename OUT_T=float>
void _convolution(Matrix<IN_T> &x, Matrix<KER_T> &kernel, Matrix<OUT_T> &output, std::vector<size_t> stride={1,1}, std::vector<size_t> pad_width={0,0}, float pad_value = 0) {
    // shape declaration
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> output_shape = output.shape();
    /*
    std::vector<size_t> output_shape;
    
    if (output.data()==NULL) {
        // Calculate the output shape using input shape, kernel shape, stride and pad
        output_shape = calc_output_shape<size_t>(x_shape, kernel_shape, stride, pad_width);
        output.resize(output_shape);
    } else {
        output_shape = output.shape();
    }
    */
        
    size_t i, j, k, l, i_ker, j_ker, in_ker, out_ker;
    // loop of n samples
    for (i=0; i<output_shape[0]; i++) {
        // loop over height of output (row)
        for (j=0; j<output_shape[1]; j++) {
            // loop over width of output
            for (k=0; k<output_shape[2]; k++) {
                // loop over output channels 
                for (out_ker=0; out_ker<kernel_shape[3]; out_ker++) {
                    float tmp_y = 0;
                    // loop over kernel height
                    for (i_ker=0; i_ker<kernel_shape[0]; i_ker++) {
                        size_t j_offset = j+i_ker-pad_width[0];
                        bool j_cond = (j_offset >= x_shape[1]) || (j_offset<0);
                        // loop over kernel width
                        for (j_ker=0; j_ker<kernel_shape[1]; j_ker++) {
                            size_t k_offset = k+j_ker-pad_width[1];
                            bool k_cond = (k_offset >= x_shape[2]) || (k_offset<0);
                            // loop over input channel
                            for (l=0; l<kernel_shape[2]; l++) {
                                if (k_cond || j_cond) {
                                    tmp_y += _multiply(
                                        pad_value, 
                                        kernel(i_ker, j_ker, l, out_ker)
                                    );
                                } else {
                                    tmp_y += _multiply(
                                        x(i, j_offset, k_offset, l), 
                                        kernel(i_ker, j_ker, l, out_ker)
                                    );
                                }
                            } // end loop over input channel
                        } // end loop over kernel width
                    } // end loop over kernel height
                    output.set(tmp_y, i, j, k, out_ker);
                } // end loop over output channels 
            } // end loop over width of output
        } // loop over height of output
    } // end loop of n samples
    return;
}

template <typename IN_T=float, typename KER_T=float>
void _convolution(Matrix<IN_T> &x, Matrix<KER_T> &kernel, Matrix<bool> &output, std::vector<size_t> stride={1,1}, std::vector<size_t> pad_width={0,0}, float pad_value = 0) {
    // shape declaration
    std::vector<size_t> kernel_shape = kernel.shape();
    std::vector<size_t> x_shape = x.shape();
    std::vector<size_t> output_shape = output.shape();
    /*
    std::vector<size_t> output_shape;
    
    if (output.data()==NULL) {
        // Calculate the output shape using input shape, kernel shape, stride and pad
        output_shape = calc_output_shape<size_t>(x_shape, kernel_shape, stride, pad_width);
        output.resize(output_shape);
    } else {
        output_shape = output.shape();
    }
    */
        
    size_t i, j, k, l, i_ker, j_ker, in_ker, out_ker;
    // loop of n samples
    for (i=0; i<output_shape[0]; i++) {
        // loop over height of output (row)
        for (j=0; j<output_shape[1]; j++) {
            // loop over width of output
            for (k=0; k<output_shape[2]; k++) {
                // loop over output channels 
                for (out_ker=0; out_ker<kernel_shape[3]; out_ker++) {
                    float tmp_y = 0;
                    // loop over kernel height
                    for (i_ker=0; i_ker<kernel_shape[0]; i_ker++) {
                        size_t j_offset = j+i_ker-pad_width[0];
                        bool j_cond = (j_offset >= x_shape[1]) || (j_offset<0);
                        // loop over kernel width
                        for (j_ker=0; j_ker<kernel_shape[1]; j_ker++) {
                            size_t k_offset = k+j_ker-pad_width[1];
                            bool k_cond = (k_offset >= x_shape[2]) || (k_offset<0);
                            // loop over input channel
                            for (l=0; l<kernel_shape[2]; l++) {
                                if (k_cond || j_cond) {
                                    tmp_y += _multiply(
                                        pad_value, 
                                        kernel(i_ker, j_ker, l, out_ker)
                                    );
                                } else {
                                    tmp_y += _multiply(
                                        x(i, j_offset, k_offset, l), 
                                        kernel(i_ker, j_ker, l, out_ker)
                                    );
                                }
                            } // end loop over input channel
                        } // end loop over kernel width
                    } // end loop over kernel height
                    if (tmp_y >= 0) {
                        output.set(1, i, j, k, out_ker);
                    } else {
                        output.set(0, i, j, k, out_ker);
                    }
                } // end loop over output channels 
            } // end loop over width of output
        } // loop over height of output
    } // end loop of n samples
    return;
}
#endif