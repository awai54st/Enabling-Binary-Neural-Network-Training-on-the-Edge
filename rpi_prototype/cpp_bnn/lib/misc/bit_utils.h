#ifndef BIT_UTILS_H
#define BIT_UTILS_H

#include "data_types.h"


VECTOR_2D_T<bool> pack_5_2d(const VECTOR_2D_T<float>& x);
VECTOR_2D_T<float> unpack_5_2d(const VECTOR_2D_T<bool>& x);

VECTOR_2D_T<bool> transpose(const VECTOR_2D_T<bool>& x);
VECTOR_2D_T<bool> pack_bits(const VECTOR_2D_T<float> x, bool transpose=false);

#endif