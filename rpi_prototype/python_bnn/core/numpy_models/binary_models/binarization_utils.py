import numpy as np
#from numba import jit


def binarize_weight(x, dtype=np.float32):
    """clipped = np.clip(x, -1, 1, dtype=dtype)
    rounded = np.sign(x, dtype=dtype)
    return clipped+(rounded-clipped)"""
    return np.sign(x, dtype=dtype)

'''
@jit(nopython=True, parallel=True)
def binarize_weight(x):
    """
    Get the sign of x
    
    - To use numba acceleration, ensure that type of x
    is casted before calling the function.
    
    Args:
        x (np.array): array which sign of its elements are taken
    """
    #return np.sign(x, dtype=dtype)
    return np.sign(x)
'''