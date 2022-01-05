import numpy as np

from core.base_layers import Layer
from core.numpy_models.binary_models.binarization_utils import binarize_weight
from core.utils.bit_packing_utils import PackBits

class BinaryActivation(Layer):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.pack = PackBits(dtype=dtype, min_val=0)
    
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    def forward(self, prev_input, training=True):
        prev_input = prev_input.astype(np.float32)
        # assert prev_input.dtype == self.dtype
        
        prev_input_mask = np.abs(prev_input) <= 1
        prev_input_masked = np.ones_like(prev_input, dtype=np.float32)*prev_input_mask
        
        #self.prev_input = prev_input_masked
        self.prev_input = self.pack.pack_bits(prev_input_masked)
        
        #return (binarize_weight(prev_input, dtype=self.dtype)).astype(self.dtype)
        return binarize_weight(prev_input)
    
    def backprop(self, output_grad):
        output_grad = output_grad.astype(np.float32)
        
        prev_input = self.pack.unpack_bits(self.prev_input)
        #prev_input = self.prev_input
        
        return prev_input*output_grad
    
    def set_weights(self):
        pass
    