import os
if os.environ.get("USE_CUPY"):
    import cupy as np
else:
    import numpy as np
    
from core.base_layers import Layer



class StraightThroughActivation(Layer):
    def __init__(self, dtype=np.float16):
        self.dtype = dtype
    
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    def forward(self, prev_input, training=True):
        self.prev_input = prev_input.astype(self.dtype)
        return prev_input
    
    def backprop(self, output_grad):
        return (np.ones_like(self.prev_input, dtype=self.dtype)*output_grad).astype(self.dtype)
    
    def set_weights(self):
        pass
    