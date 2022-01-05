#include "gradient_quantisation.h"
#include <algorithm>
#include <math.h>

template <class T>
T shift_multiplier(T f, int shift_amnt) {
    return f * (T)(1<<shift_amnt);
};

template <class T>
T shift_divider(T f, int shift_amnt) {
    return f / (T)(1<<shift_amnt);
};

template <class T>
T neg_shift_divider(T f, int shift_amnt) {
    return shift_divider(f, -shift_amnt);
};

template <class T>
constexpr const T& clip(const T &value, const T &min_val, const T &max_val) {
    return std::min(
        max_val, std::max(min_val, value) );
};

template <class T>
T sign(T value) {
    return (value>=0)-(value<0);
};

float abs_max_2d(const VECTOR_2D_T<float> x) {
    // get VECTOR_2D_T<float>
    // using T = typename std::decay<decltype(*x[0].begin())>::type;
    float max_val = x[0][0];
    
    // https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-reduction.html
    float tmp_max = 0;
    for(const VECTOR_1D_T<float>& i : x) {
        for (const float& j:i) {
            tmp_max = std::max(max_val, abs(j) );
        }
        max_val = std::max(max_val, tmp_max);
    }
    
    return max_val;
};

VECTOR_2D_T<float> pre_gradient_quatisation(VECTOR_2D_T<float> x, const int bias) {
    const auto bias_shift = [bias](auto x) {
        if (bias>=0) {
            return shift_multiplier<float>(x, bias);
        } else {
            return shift_divider<float>(x, -bias);
        }
    };
    
    for(VECTOR_1D_T<float>& i : x) {
        for(float& j : i) {
            j = bias_shift(j);
        }
    }
    
    
    
    /*
    for(VECTOR_1D_T<float>& i : x) {
        std::transform(i.begin(), i.end(), i.begin(), bias_shift);
    }
    
    // https://en.cppreference.com/w/cpp/algorithm/transform
    for(VECTOR_1D_T<float>& i : x) {
        std::transform(std::execution::par, i.begin(), i.end(), i.begin(), 
                       [bias](float tmp_x) -> float {return run(tmp_x, bias);});
    }
    
    const auto run = (bias>=0) ? shift_multiplier<float> : neg_shift_divider<float> ;
    int i, j;
    #pragma omp parallel for private(i, j) num_threads(4)
    for(i=0; i<x.size(); i++) {
        for (j=0; j<x[0].size(); j++) {
            x[i][j] = run(x[i][j], bias);
        }
    }
    */
    return x;
};

VECTOR_2D_T<float> gradient_quantisation(VECTOR_2D_T<float> x, const int width, int bias) {
    const float max_value = (1<<(width-1));
    
    // currying
    auto _clip_partial = [max_value](float f) {
        return clip<float>(
            round( log2(abs(f)+1e-45f ) ), 
            -max_value, (max_value-1));
    };
    
    
    int i, j;
    int tmp_po2;
    for(i=0; i<x.size(); i++) {
        for (j=0; j<x[0].size(); j++) {
            tmp_po2 = _clip_partial(x[i][j]) - bias;
            if (tmp_po2 < 0) {
                x[i][j] = sign<float>(x[i][j]) / (1 << -tmp_po2 );
            } else {
                x[i][j] = sign<float>(x[i][j]) * (1 << tmp_po2 );
            }
        }
    }
    
    return x;
};