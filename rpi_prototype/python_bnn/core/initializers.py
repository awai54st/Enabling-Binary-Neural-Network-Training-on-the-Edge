# https://visualstudiomagazine.com/articles/2019/09/05/neural-network-glorot.aspx
import numpy as np


def uniform_initializer(size, lo=-0.01, hi=0.01, dtype=np.float16):
    """
    Args:
        size: size of array
        lo: lower limit of uniform initialization
        hi: higher limit of uniform initialization
        dtype: data type of the array
    """
    return np.random.uniform(lo, high, size).astype(dtype)

def glorot_normal_mod_initializer(size, dtype=np.float16):
    """
    Args:
        size: kernel_shape
        
    """
    stdv = 1/np.sqrt(np.prod(np.array(size[:len(size)-1])))
    return np.random.normal(
        loc=0.0, scale=stdv,
        size = size
    ).astype(dtype)

def glorot_uniform_initializer(size, n_input, n_hidden_node, dtype=np.float16):
    """
    Args:
        size: size of array
        n_input: lower limit of uniform initialization
        n_hidden_node: higher limit of uniform initialization
        dtype: data type of the array
    """
    sd = np.sqrt(6.0/(n_input + n_hidden_node), dtype=dtype)
    return np.random.uniform(-sd, sd, size).astype(dtype)

def glorot_normal_initializer(size, lo=-0.01, hi=0.01, dtype=np.float16):
    """
    Args:
        size: size of array
        n_input: lower limit of uniform initialization
        n_hidden_node: higher limit of uniform initialization
        dtype: data type of the array
    """
    sd = np.sqrt(2.0/(n_input + n_hidden_node), dtype=dtype)
    return np.random.uniform(-sd, sd, size).astype(dtype)