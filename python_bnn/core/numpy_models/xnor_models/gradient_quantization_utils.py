import numpy as np


def FXP_quantize(value, point, width, dtype=np.float32):
    value = dtype(value)
    point = dtype(point)
    width = dtype(width)
    maximum_value = np.array(2**(width-1), dtype=dtype)

    # x << (width - point)
    shift = dtype(2 ** (np.round(width) - np.round(point)))
    value_shifted = dtype(value * shift)
    
    # Quantize
    value_shifted = np.round(value_shifted)
    value_shifted = np.clip(value_shifted, -maximum_value, maximum_value - 1)
    
    # Revert bit-shift earlier
    return dtype(value_shifted/shift), point, width

def log2(x, eps=1e-37, dtype=np.float32):
    #x = dtype(x)
    #eps = dtype(eps)
    
    #numerator = np.log(x+eps, dtype=dtype)
    #denominator = np.log(2, dtype=dtype)
    #return numerator / denominator
    return dtype(np.log(x+eps)/np.log(2))

def LOG_quantize(value, width,  eps=1e-37, dtype=np.float32):
    #value = np.array(value, dtype=dtype)
    #width = np.array(width, dtype=dtype)
    #eps = np.array(eps, dtype=dtype)
    
    sign = dtype(np.sign(value + eps)) # Adding a small bias to remove zeros
    value = log2(np.abs(value), dtype=dtype)
    # Quantize
    value, __, __ = FXP_quantize(value, width, width, dtype=dtype)
    
    return dtype(sign * (2 ** value))