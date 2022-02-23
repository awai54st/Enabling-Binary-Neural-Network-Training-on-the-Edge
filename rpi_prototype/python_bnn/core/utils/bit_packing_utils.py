import numpy as np


class PackBits:
    def __init__(self, dtype=np.float32, min_val=-1):
        self.dtype = dtype
        self.min_val = min_val
        
    """def gradient_quantization(self, dout):
        dout_max = np.max(np.abs(dout, dtype=self.dtype))
        dout_bias = -np.round(log2(dout_max, dtype=self.dtype))+8
        
        dout *= (2**dout_bias)
        dout = LOG_quantize(dout, 4.0, dtype=self.dtype)
        dout *= 2 ** (-dout_bias)
        
        return dout"""
        
    def pack_bits(self, arr, axis=0):
        flattened_arr = arr.reshape(-1)
        self.arr_shape = arr.shape
        self.packed_count = flattened_arr.shape[0]
        self.packed_arr_dtype = arr.dtype
        
        bool_flattened_arr = np.where(flattened_arr>0, 1, 0).astype(np.bool)
        
        packed_flattened_arr = np.packbits(bool_flattened_arr, axis=0)
        
        
        return packed_flattened_arr
        
    def unpack_bits(self, packed_flattened_arr):
        bool_flattened_arr = np.unpackbits(
            packed_flattened_arr, axis=0, count=self.packed_count
        ).astype(self.packed_arr_dtype)
        
        arr = np.where(bool_flattened_arr==0, self.min_val, 1).reshape(self.arr_shape)
        
        return arr.astype(self.packed_arr_dtype)
    
    


def test_pack_bits():
    dtype = np.float32
    np.random.seed(0)
    test_input = np.random.uniform(-1, 1, (100, 32, 32, 128)).astype(dtype)
        
    bool_flattened_arr = np.where(test_input>0, 1, 0).astype(np.bool)
    packed_flattened_arr = np.packbits(bool_flattened_arr, axis=0)