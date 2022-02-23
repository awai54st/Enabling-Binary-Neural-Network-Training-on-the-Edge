from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, dtype=None, random_seed=None, trainable=True, use_bias=False):
        self.trainable = trainable
        self.dtype = dtype
        self.random_seed = random_seed
        self.use_bias = use_bias
        
        self.w = None
        self.is_built = False
        
        self._dw = None
        self.b = None
        self._db = None
        
    @property
    def weights(self):
        if (self.w is None) and (self.b is None):
            return None
        elif self.use_bias:
            return [self.w, self.b]
        else:
            return [self.w]
    
    @property
    def gradients(self):
        if (self._dw is None) and (self._db is None):
            return None
        elif self.use_bias:
            return [self._dw, self._db]
        else:
            return [self._dw]
    
    @abstractmethod
    def forward(self, prev_input, training):
        pass
    
    @abstractmethod
    def backprop(self, dout):
        pass
    
    def set_weights(self, w):
        if self.trainable:
            self.w = w[0].astype(self.dtype)
            if self.use_bias:
                self.b = w[1].astype(self.dtype)