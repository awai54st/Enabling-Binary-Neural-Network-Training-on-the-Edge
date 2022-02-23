#ifndef GRADIENT_QUANTISATION_H
#define GRADIENT_QUANTISATION_H

#include "data_types.h"

template <class T> T shift_multiplier(T f, int shift_amnt);
template <class T> T shift_divider(T f, int shift_amnt);
template <class T> T neg_shift_divider(T f, int shift_amnt);
template <class T> constexpr const T& clip(const T &value, const T &min_val, const T &max_val);
template <class T> T sign(T value);

float abs_max_2d(const VECTOR_2D_T<float> x);
VECTOR_2D_T<float> pre_gradient_quatisation(VECTOR_2D_T<float> x, const int bias);
VECTOR_2D_T<float> gradient_quantisation(VECTOR_2D_T<float> x, const int width, int bias);

#endif