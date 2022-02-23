#ifndef arithmetic_ops_h
#define arithmetic_ops_h


#include "data_type.h"


template <typename T=float>
float _multiply(float16_t input_1, PO2_5bits_t input_2) {
    //float input_PO2 = PO2_calculation(input_2);
    return input_2*input_1;
};

template <typename T=float>
float _multiply(float input_1, PO2_5bits_t input_2) {
    //float input_PO2 = PO2_calculation(input_2);
    return input_2*input_1;
};

template <typename T=float>
float _multiply(PO2_5bits_t input_1, float16_t input_2) {
    //float input_PO2 = PO2_calculation(input_2);
    return input_1*input_2;
};

template <typename T=float>
float _multiply(bool input_1, PO2_5bits_t input_2) {
    //float input_PO2 = PO2_calculation(input_2);
    //printf("multiply PO2: %f \n", input_PO2);
    return input_2*input_1;
};

template <typename T=float>
float _multiply(PO2_5bits_t input_1, bool input_2) {
    //float input_PO2 = PO2_calculation(input_2);
    //printf("multiply PO2: %f \n", input_PO2);
    return input_1*input_2;
};


template <typename T=float>
float _multiply(bool input_1, T input_2) {
    if (input_1) {
        return input_2;
    } else {
        return -input_2;
    }
};

template <typename T=float>
float _multiply(T input_1, bool input_2) {
    if (input_2) {
        return input_1;
    } else {
        return -input_1;
    }
};

template <typename T1=float, typename T2=float>
float _multiply(T1 input_1, T2 input_2) {
    return float(input_1) * float(input_2);
};

template <typename T=float>
float PO2_calculation(PO2_5bits_t x) {
    int8_t po2_bias = x.value;
    long long unsigned int a=1;
    //printf("value PO2: %d \n", po2_bias);
    //printf("sign PO2: %d \n", int(x.sign));
    
    if (po2_bias < 0) {
        return x.sign / float(a << -po2_bias );
    } else {
        return x.sign * float(a << po2_bias );
    }
}

#endif