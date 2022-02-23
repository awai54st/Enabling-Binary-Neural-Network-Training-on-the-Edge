#ifndef bit_packing_h
#define bit_packing_h

#include "data_type.h"


template <typename T1, template<typename> class MAT_CONTAINER = Matrix>
void pack_bits(const MAT_CONTAINER<T1> & x, MAT_CONTAINER<bool> & packed_to, bool transpose=false) {
    int x_size = x.size();
    for (int i=0; i<x_size; i++) {
        packed_to[i] = (x[i] >= 0);
    }
};

template <typename T1, typename T2=float, template<typename> class MAT_CONTAINER = Matrix>
void pack_bits(const MAT_CONTAINER<T1> & x, MAT_CONTAINER<T2> & packed_to, bool transpose=false) {
    int x_size = x.size();
    for (int i=0; i<x_size; i++) {
        packed_to[i] = (x[i] >= 0) ? 1:-1;
    }
};

#endif