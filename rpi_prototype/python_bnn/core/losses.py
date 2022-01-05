import numpy as np

from core.base_layers import Layer
from core.numpy_models.full_precision_models.layers import Softmax

"""
# original
class CrossEntropy(Layer):
    # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def __init__(self, dtype=np.float32, eps=1e-37):
        self.dtype = dtype
        self.eps = np.array(eps, dtype)
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    def forward(self,logits, y):
        self.logits = logits.astype(self.dtype)+self.eps
        #self.logits = logits - np.max(logits, axis=-1, keepdims=True)
        self.y = y.astype(self.dtype)
        #return np.abs(np.sum(self.y*self.logits, axis=-1)/self.y.sum(axis=-1))
        return np.sum(self.y*-np.log(self.logits), axis=-1)/self.y.sum(axis=-1)
        #return (np.where(self.y==1,-np.log(self.x), 0)).sum(axis=1, dtype=self.dtype)

    def backprop(self):
        # assert np.isinf(np.abs(-1/self.x)).all() == False
        return np.where(self.y==1,-1/self.logits, 0)
        #print((np.where(self.logits==0, 1, 0) - self.y).astype(self.dtype))
        #return (np.where(self.logits==0, 1, 0) - self.y).astype(self.dtype)
        #return np.where(self.y==1,-1/self.logits, 0)
    
    def set_weights(self):
        pass
"""

# tensorflow black box
class CrossEntropy(Layer):
    """
    Internal softmax and cross entropy
    """
    # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def __init__(self, dtype=np.float32, eps=1e-38):
        self.dtype = dtype
        if dtype == np.float16:
            self.eps = dtype(1e-7)
        else:
            self.eps = dtype(1e-45)
        self.softmax = Softmax(dtype=np.float32)
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    def forward(self, logits, label):
        softmax_output = self.softmax.forward(logits)
        label = label.astype(np.float32)
        
        self.softmax_output = softmax_output.astype(self.dtype)
        self.label = label
        
        # normal CE output
        normal_CE_output = np.sum(label*-np.log(softmax_output+self.eps), axis=-1)/label.sum(axis=-1)
        
        return normal_CE_output

    def backprop(self):
        label = self.label.astype(np.float32)
        softmax_output = self.softmax_output.astype(np.float32)
        
        ONE = np.float32(1)
        CE_grad = np.where(label==1,-ONE/(softmax_output+1e-45), 0)
        softmax_grad = softmax_output * (CE_grad- np.sum(CE_grad*softmax_output, axis=1, keepdims=True, dtype=np.float32))
        
        return softmax_grad
    
    def set_weights(self):
        pass
    
'''# tensorflow black box
class CrossEntropy(Layer):
    """
    Internal softmax and cross entropy
    """
    # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def __init__(self, dtype=np.float32, eps=1e-38):
        self.dtype = dtype
        self.eps = np.array(eps, dtype)
        self.softmax = Softmax(dtype=dtype)
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    
    def get_softmax_gradient(self):
        CE_grad = np.where(self.label==1,-1/(self.softmax_output+self.eps), 0)
        self.softmax_grad = self.softmax_output * (CE_grad- np.sum(CE_grad*self.softmax_output, axis=1, keepdims=True, dtype=self.dtype))
        #self.mask = np.expand_dims((self.softmax_grad==0).all(axis=-1), axis=-1)
    
    def forward(self, logits, label):
        softmax_output = self.softmax.forward(logits)
        self.label = label.astype(self.dtype)
        
        self.softmax_output = softmax_output.astype(self.dtype)
        self.mask_generation = softmax_output[np.where(self.label)]
        mask = softmax_output[np.where(self.label)] < 1e-25
        #mask = self.mask[:, 0]
        self.mask = np.expand_dims(mask, -1)
        #print(np.where(mask))
        
        # normal CE output
        normal_CE_output = np.sum(self.label*-np.log(self.softmax_output+self.eps), axis=-1)/self.label.sum(axis=-1)
        odd_CE_output = np.sum(self.label*-self.softmax.x_shift, axis=-1)/self.label.sum(axis=-1)
        return (normal_CE_output*(1-mask) + odd_CE_output*mask).astype(self.dtype)
        #return (np.where(self.y==1,-np.log(self.x), 0)).sum(axis=1, dtype=self.dtype)

    def backprop(self):
        # assert np.isinf(np.abs(-1/self.x)).all() == False
        #print(self.mask.sum())
        self.get_softmax_gradient()
        #CE_grad = np.where(self.label==1,-1/(self.softmax_output+self.eps), 0)
        #softmax_grad = self.softmax_output * (CE_grad- np.sum(CE_grad*self.softmax_output, axis=1, keepdims=True, dtype=self.dtype))
        #self.softmax_grad = softmax_grad
        softmax_grad = self.softmax_grad.astype(np.float32)
        if self.mask.sum() == 0:
            #return softmax_grad.astype(self.dtype)
            return softmax_grad
        self.chkp1 = np.where(self.softmax.x_shift==0, 1, 0)
        chkp2 = self.softmax_output*(1-self.softmax_output)
        self.chkp2 = np.where(chkp2> 2*self.eps, chkp2, 0)
        #self.chkp3 = softmax_grad
        #self.mask_chkp2 = self.chkp2 > 2*self.eps
        # correct idx_2
        return (1-self.mask)*np.where(np.abs(softmax_grad)>self.eps/10,softmax_grad,0) + self.mask*(self.chkp1-self.label+(1-self.chkp1)*self.chkp2-self.chkp1*self.chkp2)
        #print((np.where(self.logits==0, 1, 0) - self.y).astype(self.dtype))
        #return (np.where(self.logits==0, 1, 0) - self.y).astype(self.dtype)
        #return np.where(self.y==1,-1/self.logits, 0)
    
    def set_weights(self):
        pass
'''