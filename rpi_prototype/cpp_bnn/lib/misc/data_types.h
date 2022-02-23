#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <bitset>
#include <vector>
#include <cstdint>

// alias-declaration
// ref: https://www.nextptr.com/tutorial/ta1193988140/how-cplusplus-using-or-aliasdeclaration-is-better-than-typedef
using BIT_64 = std::bitset<64>;
using BIT_63 = std::bitset<64>;

using VECTOR_1D_FLOAT = std::vector<float>;
using VECTOR_2D_FLOAT = std::vector<VECTOR_1D_FLOAT>;

using VECTOR_4d = std::vector< std::vector< std::vector< std::vector<float> > > >;
using VECTOR_3d = std::vector< std::vector< std::vector<float> > >; 
using VECTOR_2d = std::vector< std::vector<float> >;
using VECTOR_1d = std::vector<float>;

// Alias Templates
template<class T> using VECTOR_1D_T = std::vector<T>;
template<class T> using VECTOR_2D_T = std::vector<VECTOR_1D_T<T> >;
template<class T> using VECTOR_3D_T = std::vector<VECTOR_2D_T<T> >; 
template<class T> using VECTOR_4D_T = std::vector<VECTOR_3D_T<T> >;

using VECTOR_1D_BIT_64 = std::vector<BIT_64>;
using VECTOR_2D_BIT_64 = std::vector<VECTOR_1D_BIT_64>;
#endif