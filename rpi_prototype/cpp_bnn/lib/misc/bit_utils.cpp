#include "bit_utils.h"
#include <arm_neon.h>
#include <boost/dynamic_bitset.hpp>
#include <iostream>

//std::tr2::dynamice_bitset
/*
void dec_to_bit5(int dec, VECTOR_1D_T<bool>::iterator &vec_bit5_begin) {
    VECTOR_1D_T<bool>::iterator vec_bit5_end = vec_bit5_begin+5;
    while (vec_bit5_begin != vec_bit5_end) {
        *vec_bit5_begin = dec%2;
        dec = dec >>1;
        vec_bit5_begin++;
    }
}
*/

VECTOR_2D_T<bool> pack_5_2d(const VECTOR_2D_T<float>& x) {
    std::size_t n = x.size(), m = x[0].size();
    std::size_t i, i_bit;
    VECTOR_2D_T<bool> x_T(n, VECTOR_1D_T<bool>(m*5) );
    
    for (i = 0; i< n; i++) {
        VECTOR_1D_T<bool>::iterator it = x_T[i].begin();
        for (i_bit = 0; i_bit < m; i_bit++) {
            /*
            dec_to_bit5((int)x[i][i_bit], it);
            it += i_bit*5;
            */
            for (int bit_width = 0; bit_width<5; bit_width++) {
                x_T[i][(i_bit*5)+bit_width] = (int)x[i][i_bit]>>bit_width%2;
            }
            
        }
    }
    return x_T;
};
VECTOR_2D_T<float> unpack_5_2d(const VECTOR_2D_T<bool>& x) {
    std::size_t n = x.size(), m = x[0].size()/5;
    std::size_t i, i_bit;
    VECTOR_2D_T<float> x_T(n, VECTOR_1D_T<float>(m) );
    
    for (i = 0; i< n; i++) {
        for (i_bit = 0; i_bit < m; i_bit++) {
            x_T[i][i_bit] = x[i][i_bit*5]+x[i][(i_bit*5)+1]<<1+x[i][(i_bit*5)+2]<<2+x[i][(i_bit*5)+3]<<3+x[i][(i_bit*5)+4]<<4;
            
            /*
            for (int bit_width = 0; bit_width<5; bit_width++) {
                x_T[i][i_bit] += x[i][(i_bit*5)+bit_width]>>bit_width;
            }
            */
        }
    }
    return x_T;
};


VECTOR_2D_T<bool> transpose(const VECTOR_2D_T<bool>& x) {
    std::size_t n = x.size(), m = x[0].size();
    std::size_t i, i_bit;
    VECTOR_2D_T<bool> x_T(m, VECTOR_1D_T<bool>(n) );
    
    for (i = 0; i< n; i++) {
        for (i_bit = 0; i_bit < m; i_bit++) {
            x_T[i_bit][i] = x[i][i_bit];
        }
    }
    return x_T;
};


VECTOR_2D_T<bool> pack_bits(const VECTOR_2D_T<float> x, bool transpose) {
    std::size_t i, j;
    
    if (transpose) {
        std::size_t n = x.size(), m = x[0].size();
        VECTOR_2D_T<bool> packed_data(m, VECTOR_1D_T<bool>(n) );
        
        
        for (j = 0; j< m; j++) {
            for (i = 0; i< n; i++) {
                packed_data[i][j] = (x[j][i] >= 0)? 1:0 ;
            }
        }
        return packed_data;
    } else {
        std::size_t n = x.size(), m = x[0].size();
        VECTOR_2D_T<bool> packed_data(n, VECTOR_1D_T<bool>(m) );
        
        for (int i = 0; i< n; i++) {
            for (int j = 0; j< m; j++) {
                packed_data[i][j] = (x[i][j] >= 0)? 1:0; 
            }
        }
        return packed_data;
    }
}