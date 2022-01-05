import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
# initialization technique: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

def binarize_weight(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = tf.clip_by_value(x, -1, 1)
    rounded = tf.sign(clipped)
    return clipped + tf.stop_gradient(rounded - clipped)


class BinaryActivationTf(Layer):
    def __init__(self, name=None):
        super(BinaryActivationTf, self).__init__(name=name)
    def call(self, x, mask=None):
        out_bin = binarize_weight(x)
        return out_bin
    
    
class Conv2DTf(Layer):
    def __init__(self, w, b, strides=[1,1,1,1], padding="VALID", name=None):
        super(Conv2DTf, self).__init__(name=name)
        self.w = w
        self.b = b
        self.strides = strides
        self.padding = padding
        self.dilation_rate = 1
    
    def build(self, input_shape):
        pass
            
    def call(self, inputs, training=None):
        conv = tf.nn.conv2d(
            input=inputs, 
            filters=binarize_weight(self.w),
            strides=self.strides, 
            dilations=self.dilation_rate,
            padding=self.padding.upper(), 
            name=self.name
        ) # Vanilla BNN            
        return conv+self.b
        
class DenseTf(Layer):
    def __init__(self, w, b, name=None):
        super(DenseTf, self).__init__(name=name)
        self.w = w
        self.b = b
        
    def build(self, input_shape):
        pass
        
    def call(self, inputs):
        out = tf.matmul(inputs, binarize_weight(self.w)) + self.b # Vanilla BNN
        return out