#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include "data_types.h"

extern VECTOR_2D_T<float> glorot_normal_initializer_2d(int c_in, int c_out, int seed=0);
extern VECTOR_4d glorot_normal_initializer_4d(int w, int h, int c_in, int c_out, int seed=0);

#endif