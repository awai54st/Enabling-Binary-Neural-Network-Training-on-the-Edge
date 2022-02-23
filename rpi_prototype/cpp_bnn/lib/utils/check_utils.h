#ifndef check_utils_h
#define check_utils_h

#include "data_type.h"
#include <limits>
#include <stdexcept> // std::runtime_error

template <typename T1>
void has_inf(Matrix<T1> & x) {
    const int x_size = x.size();
    printf("Start checking for infinity ..................................\n");
    for (int i=0; i<x_size; i++) {
        if (std::isinf(x[i])) {
             throw std::runtime_error("infinity");
        }
    }
    printf("Finish checking, no infinity ..................................\n");
}

template <typename T1>
void has_nan(Matrix<T1> & x) {
    const int x_size = x.size();
    printf("Start checking for nan ..................................\n");
    for (int i=0; i<x_size; i++) {
        if (std::isnan(x[i])) {
             throw std::runtime_error("nan");
        }
    }
    printf("Finish checking, no nan ..................................\n");
}



#endif