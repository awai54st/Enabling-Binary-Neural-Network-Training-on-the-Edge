import numpy as np
from core.initializers import glorot_normal_mod_initializer

def test_dense_w_init():
    dtype = np.float32
    
    res_w = glorot_normal_mod_initializer(
        size=(10,20),
        dtype=dtype
    )
    assert res_w.dtype == dtype
    
def test_conv_w_init():
    dtype = np.float32
    
    res_w = glorot_normal_mod_initializer(
        size=(3, 3, 3, 5),
        dtype=dtype
    )
    assert res_w.dtype == dtype
    