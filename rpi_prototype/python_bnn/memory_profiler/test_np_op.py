import numpy as np

def float_32_mul_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float32)
    y = x*x
    
def float_16_mul_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    y = x*x

    
def float_16_32_16_mul_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    x = x.astype(np.float32)
    y = x*x
    y = y.astype(np.float16)

    
def float_16_32_16_mul_op_alt():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    x = x.astype(np.float32)
    y = np.float16(x*x)
    
def float_32_add_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float32)
    y = x+x

    
def float_16_add_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    y = x+x

    
def float_32_div_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float32)
    y = x/x

    
def float_16_div_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    y = x/x

    
def float_32_log_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float32)
    y = np.log(x)
    y = y.astype(np.float32)

    
def float_16_log_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    y = np.log(x)
    y = y.astype(np.float16)

    
def float_32_sign_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float32)
    y = np.sign(x)

    
def float_16_sign_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    y = np.sign(x)

    
def float_16_store_op():
    x = np.ones((100, 10, 10, 512), dtype=np.float16)
    y = x.tostring()
