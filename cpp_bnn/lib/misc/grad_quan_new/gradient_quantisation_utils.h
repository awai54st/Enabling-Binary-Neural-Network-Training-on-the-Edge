#ifndef gradient_quantisation_utils_h
#define gradient_quantisation_utils_h

#include "../../../utils/data_type.h"

int8_t get_po2_bias(const Matrix2D<float> & x);
void scale_with_po2_bias(Matrix2D<float> & x, const int8_t bias);
void log_quantize(Matrix2D<float> & x, const int8_t bias, const int8_t width=4);

#endif