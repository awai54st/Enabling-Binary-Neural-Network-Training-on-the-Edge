import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from core.keras_models.keras_xnor_models.binarization_utils import tf_custom_gradient_method
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
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=None, padding='valid', activation=None, use_batch_norm=True, use_bias=True, name=None):
        super(Conv2DTf, self).__init__(name=name)
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.strides=strides
        #self.first_layer = first_layer
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        #self.is_first_layer = is_first_layer
    
    def build(self, input_shape):
        stdv=1/np.sqrt(self.kernel_size[0]*self.kernel_size[1]*input_shape[-1])
        w = np.random.normal(
            loc=0.0, 
            scale=stdv,
            size=[self.kernel_size[0],self.kernel_size[1],input_shape[-1],self.filters]).astype(np.float32)
        self.w = tf.Variable(w, name=f"{self.name}_w")
        self.b = tf.Variable(tf.zeros(self.filters), name=f"{self.name}_b")
            
    def call(self, inputs, training=None):
        conv = tf.nn.conv2d(
            input=inputs, 
            filters=self.w,
            strides=self.strides, 
            dilations=self.dilation_rate,
            padding=self.padding.upper(), 
            name=self.name
        ) # Vanilla BNN   
        if self.use_bias:
            return tf.nn.bias_add(conv, self.b)
        return conv
        
class DenseTf(Layer):
    def __init__(self, filters, activation=None, use_bias=True, name=None):
        super(DenseTf, self).__init__(name=name)
        self.filters=filters
        self.use_bias = use_bias
        
    def build(self, input_shape):
        stdv=1/np.sqrt(input_shape[-1])
        w = np.random.normal(
            loc=0.0, 
            scale=stdv,
            size=[input_shape[-1], self.filters]).astype(np.float32)
        self.w = tf.Variable(w, name=f"{self.name}_w")
        self.b = tf.Variable(tf.zeros(self.filters), name=f"{self.name}_b")
        
    def call(self, inputs):
        out = tf.matmul(inputs, binarize_weight(self.w)) # Vanilla BNN            
        if self.use_bias:
            return tf.nn.bias_add(out, self.b)
        return out
    
    
    
class l1_batch_norm_mod_conv(Layer):
    def __init__(self,batch_size,width_in,ch_in,momentum, **kwargs):
        super(l1_batch_norm_mod_conv, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.width_in = width_in
        self.ch_in = ch_in
        self.momentum = momentum

    def build(self, input_shape):
        super(l1_batch_norm_mod_conv, self).build(input_shape)
        beta = np.zeros([self.ch_in])*1.0
        self.beta=K.variable(beta)
        #self.trainable_weights=[self.beta]
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=[1,1,1,self.ch_in],
            initializer=tf.zeros_initializer(),
            trainable=False)
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=[1,1,1,self.ch_in],
            initializer=tf.initializers.ones(),
            trainable=False)

    def call(self, x):
        # Check if training or inference
        training = K.learning_phase()

        # Calculate mean and l1 variance
        N = self.batch_size*self.width_in*self.width_in
        self.mu = 1./N * K.sum(x, axis = [0,1,2])
        self.mu = K.reshape(self.mu,[1,1,1,-1])
        xmu = x - self.mu
        self.var = 1./N * K.sum(K.abs(xmu), axis = [0,1,2])
        self.var = K.reshape(self.var,[1,1,1,-1])

        # Update moving stats at training mode only
        mean_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_mean,
            self.mu, self.momentum),
            lambda:self.moving_mean)
        var_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_var,
            self.var, self.momentum),
            lambda:self.moving_var)
        self.add_update([mean_update, var_update])

        return self.quantise_gradient_op(x) + K.reshape(self.beta, [1,1,1,-1])


    @tf_custom_gradient_method
    def quantise_gradient_op(self, x):

        # Forward prop

        # Check if training or inference
        training = K.learning_phase()

        # Inference mode: apply moving stats
        if training in {0, False}:
            self.mu = self.moving_mean
            self.var = self.moving_var

        xmu = x - self.mu
        ivar = 1./self.var
        result = xmu * ivar


        def custom_grad(dy):
            # Backprop
            N = self.batch_size*self.width_in*self.width_in

            # BN backprop
            dy_norm_x = dy * ivar
            term_1 = dy_norm_x - K.reshape(1.0/N * K.sum(dy_norm_x, axis=[0,1,2]), [1,1,1,-1])
            term_2 = result # Vanilla BN
            term_3 = 1.0/N * K.sum(dy_norm_x * result, axis=[0,1,2]) # Vanilla BN
            term_3 = K.reshape(term_3, [1,1,1,-1])
            dx = term_1 - term_2 * term_3

            return dx

        return result, custom_grad

    def get_output_shape_for(self,input_shape):
        return input_shape
    def compute_output_shape(self,input_shape):
        return input_shape
    
    
class l1_batch_norm_mod_dense(Layer):
    def __init__(self,batch_size,ch_in,momentum, **kwargs):
        super(l1_batch_norm_mod_dense, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.ch_in = ch_in
        self.momentum = momentum

    def build(self, input_shape):
        super(l1_batch_norm_mod_dense, self).build(input_shape) 
        beta = np.zeros([self.ch_in])*1.0
        self.beta=K.variable(beta)
        #self.trainable_weights=[self.beta]
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=[1,self.ch_in],
            initializer=tf.zeros_initializer(),
            trainable=False)
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=[1,self.ch_in],
            initializer=tf.initializers.ones(),
            trainable=False)

    def call(self, x):
        # Check if training or inference
        training = K.learning_phase()

        # Calculate mean and l1 variance
        N = self.batch_size
        self.mu = 1./N * K.sum(x, axis = 0)
        self.mu = K.reshape(self.mu,[1,-1])
        
        xmu = x - self.mu
        self.var = 1./N * K.sum(K.abs(xmu), axis = 0)
        self.var = K.reshape(self.var,[1,-1])
        # Update moving stats at training mode only
        mean_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_mean,
            self.mu, self.momentum),
            lambda:self.moving_mean)
        var_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_var,
            self.var, self.momentum),
            lambda:self.moving_var)
        self.add_update([mean_update, var_update])

        return self.quantise_gradient_op(x) + K.reshape(self.beta, [1,-1])

    @tf_custom_gradient_method
    def quantise_gradient_op(self, x):

        # Forward prop

        # Check if training or inference
        training = K.learning_phase()

        # Inference mode: apply moving stats
        if training in {0, False}:
            self.mu = self.moving_mean
            self.var = self.moving_var
            
        xmu = x - self.mu
        ivar = 1./self.var
        
        result = xmu * ivar
        def custom_grad(dy):
            # Backprop
            N = self.batch_size

            # BN backprop
            dy_norm_x = dy * ivar
            
            term_1 = dy_norm_x - K.reshape(1.0/N * K.sum(dy_norm_x, axis=0), [1,-1])
            term_2 = result # Vanilla BN
            term_3 = 1.0/N * K.sum(dy_norm_x * result, axis=0) # Vanilla BN
            term_3 = K.reshape(term_3, [1,-1])
            dx = term_1 - term_2 * term_3

            return dx

        return result, custom_grad

    def get_output_shape_for(self,input_shape):
        return input_shape
    def compute_output_shape(self,input_shape):
        return input_shape
    