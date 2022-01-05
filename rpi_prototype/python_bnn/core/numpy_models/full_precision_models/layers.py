import os
"""
if os.environ.get("USE_CUPY"):
    import cupy as np
else:
    import numpy as np
"""
import numpy as np
from core.initializers import glorot_normal_mod_initializer
from core.base_layers import Layer
from core.utils.conv_utils import faster_convolution, faster_full_convolution, faster_backprop_dw


class Dense(Layer):
    def __init__(self, output_units, dtype=np.float32, trainable=True, random_seed=0, use_bias=False):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        super().__init__(dtype=dtype, random_seed=random_seed, trainable=trainable, use_bias=use_bias)
        self.output_units = output_units
        
        self.w = None
        self.b = None
        self.is_built = False
        
        self._dw = None
        self._db = None
    
    def init_weights(self, input_units):
        np.random.seed(self.random_seed)
        
        """self.w = glorot_uniform_initializer(
            size=(input_units,self.output_units),
            n_input=input_units, 
            n_hidden_node=self.output_units, 
            dtype=self.dtype
        )"""
        self.w = glorot_normal_mod_initializer(
            size=(input_units,self.output_units),
            dtype=self.dtype
        )
        if self.use_bias:
            self.b = np.zeros(self.output_units).astype(self.dtype)
        self.is_built = True
        
    def forward(self, prev_input, training=True):
        # assert prev_input.dtype == self.dtype
        
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        self.prev_input = prev_input.astype(self.dtype)
        if not self.is_built:
            self.init_weights(prev_input.shape[-1])
        
        return (np.dot(prev_input,self.w) + self.b).astype(self.dtype)
    
    def backprop(self, grad_output):
        # assert grad_output.dtype == self.dtype
        
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.w.T)
        
        # compute gradient w.r.t. weights and biases
        dw = np.dot(self.prev_input.T, grad_output)
        db = (grad_output.mean(axis=0)*self.prev_input.shape[0]).astype(self.dtype)
        
        self._dw = dw
        self._db = db
            
        return grad_input.astype(self.dtype)
    

class Flatten(Layer):
    # https://towardsdatascience.com/lets-code-convolutional-neural-network-in-plain-numpy-ce48e732f5d5
    def __init__(self, dtype=np.float32):
        super().__init__(dtype=dtype)
        
    def forward(self, x, training=True):
        self.shape = x.shape
        # return np.ravel(x).reshape(self.shape[0], -1).astype(self.dtype)
        return x.reshape(self.shape[0], -1).astype(self.dtype)
    
    def backprop(self, dout):
        return dout.reshape(self.shape).astype(self.dtype)
    
    def set_weights(self):
        pass
    
    
class Padding:
    def __init__(self, input_shape, kernel_shape, stride, padding="same"):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        
    @property
    def output_shape(self):
        n, h, w, c = self.input_shape
        h_ker, w_ker, _, _ = self.kernel_shape
        h_stride, w_stride = self.stride
        
        if self.padding.upper() == "SAME":
            return n, h, w, c
        
        elif self.padding.upper() == "VALID":
            # out = (w-f+2p)/s+1
            # valid means pad = 0.
            h_new = (h-h_ker)//h_stride+1
            w_new = (w-w_ker)//w_stride+1
            return n, h_new, w_new, c
        
    @property
    def pad(self):
        # p = 1/2((out-1)*s-w+f)
        _, output_h, output_w, _ = self.output_shape
        n, h, w, c = self.input_shape
        h_ker, w_ker, _, _ = self.kernel_shape
        h_stride, w_stride = self.stride

        pad_h = int(1/2*( (output_h-1)*h_stride+h_ker-h ))
        pad_w = int(1/2*( (output_w-1)*w_stride+w_ker-w ))
        
        if (pad_h%1 != 0) or (pad_w%1 != 0):
            #pad_h = np.floor(pad_h)
            #pad_w = np.floor(pad_w)
            raise Exception("Output shape error")
            
        return int(pad_h), int(pad_w)
    
    def pad_input(self, x, mode="reflect"):
        if self.pad == (0, 0):
            return x
        pad = self.pad
        return np.pad(
            array=x,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode=mode)
    
    
class Conv2D(Layer):
    def __init__(self, filters, kernel_size, stride=(1, 1), padding="same", dtype=np.float32, trainable=True, random_seed=0, use_bias=False):
        """
        Args:
            filters (int): output_c
            kernel_size (int, int): kernel_h, kernel_w
            [NA] w (np.array): array with shape (kernel_h, kernel_w, input_c, output_c)
            [NA] b (np.array): array with shape (n, output_c)
            stride (int, int): stride alond (width, height)
            padding (str): "same" or "valid"
        """
        super().__init__(dtype=dtype, random_seed=random_seed, trainable=trainable, use_bias=use_bias)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def init_weights(self, prev_input_shape):
        np.random.seed(self.random_seed)
        
        """self.w = glorot_uniform_initializer(
            size=(self.kernel_size[0], self.kernel_size[1], prev_input_shape[-1], self.filters),
            n_input=np.prod(prev_input_shape[1:]),
            n_hidden_node=np.prod(self.kernel_size)*prev_input_shape[-1]*self.filters, 
            dtype=self.dtype
        )"""
        self.w = glorot_normal_mod_initializer(
            size=(self.kernel_size[0], self.kernel_size[1], prev_input_shape[-1], self.filters),
            dtype=self.dtype
        )
        if self.use_bias:
            self.b = np.zeros(self.filters).astype(self.dtype)
        
        self.is_built = True
    
    def forward(self, prev_input, training=True):
        prev_input = prev_input.astype(self.dtype)
        if not self.is_built:
            self.init_weights(prev_input.shape)
        w, b = self.weights
        
        input_padder = Padding(
            input_shape=prev_input.shape, 
            kernel_shape=w.shape, 
            stride=self.stride, 
            padding=self.padding)
        self.pad = input_padder.pad
        
        n, h_out, w_out, _ = input_padder.output_shape
        self.prev_input = input_padder.pad_input(prev_input)
        
        output = faster_convolution(self.prev_input, w, dtype=self.dtype)
        
        return (output+b).astype(self.dtype)
    
    
    def backprop(self, output_gradient):
        """
        https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
        Args:
            dZ (np.array): The gradient of the cost with respect to the output of the conv layer Z)
                            shape -> (n, h, w, c)
        Returns:
            dprev_input: gradient of cost with respect to the input of the conv layer (prev_input)
                     shape -> (n, h_prev, w_prev, c_prev)
        """
        w, b = self.weights
        
        dprev_input = faster_full_convolution(dout=output_gradient, w=w, stride=self.stride, dtype=self.dtype)
        dw = faster_backprop_dw(grad_output=output_gradient, prev_input=self.prev_input, kernel_shape=w.shape, stride=self.stride, dtype=self.dtype)
        db = np.sum(output_gradient, axis=(0, 1, 2))
        self._dw = dw
        self._db = db
        if self.pad == (0, 0):
                return dprev_input
        return (dprev_input[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1], :]).astype(self.dtype)
        
        

        
class Softmax(Layer):
    # https://deepnotes.io/softmax-crossentropy
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def __init__(self, dtype=np.float32):
        #self.dtype = dtype
        pass
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    def forward(self, x, training=True):
        '''
        Performs softmax layer using the given x.
        Returns a 2d numpy array containing the respective probability values.
        - x can be any array with any dimensions.
        '''
        #assert x.dtype == self.dtype
        x = x.astype(np.float32)
        #self.x = x.astype(self.dtype)
        
        # Stable Softmax
        x_shift = (x-np.max(x, axis=-1, keepdims=True))
        exp_x = np.exp(x_shift, dtype=np.float32)
        output = (exp_x/np.sum(exp_x, axis=-1, keepdims=True, dtype=np.float32))
        
        return output
        
        
    def backprop(self, grad_output):
        return grad_output
    
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        # [:, None] None for new axis in the stated location
        # assert grad_output.dtype == self.dtype
        """
        grad_output = grad_output.astype(self.dtype)
        gradient = self.output * (grad_output - np.sum(grad_output*self.output, axis=1, keepdims=True, dtype=self.dtype))
        
        return gradient.astype(self.dtype)"""
    
    def set_weights(self):
        pass
    
class BatchNorm(Layer):
    """
    batach_norm backprop: https://zaffnet.github.io/batch-normalization
    https://www.programmersought.com/article/27904944221/#12_BN_90
    
    test_l1_batch_norm_mod_conv
    """
    def __init__(self, momentum, trainable=True, eps=1e-3, dtype=np.float32):
        self.momentum = momentum
        self.trainable = trainable
        self.is_built = False
        self.eps = eps
        self.dtype = dtype
        
        
        self.beta = None
        self.gamma = None
        self.dbeta = None
        self.dgamma = None
    
    
    @property
    def weights(self):
        if (self.beta is None) and (self.gamma is None):
            return None
        return self.beta, self.gamma
    
    @property
    def gradients(self):
        if (self.dbeta is None) and (self.dgamma is None):
            return None
        return self.dbeta, self.dgamma
        
    def build(self, input_shape):
        if not self.is_built:
            if len(input_shape) == 4:
                self.axis = (0, 1, 2)
                self.reshape_shape = (1, 1, 1, -1)
                self.moving_mean = np.zeros((1, 1, 1, input_shape[-1]), dtype=np.float32)
                self.moving_var = np.ones((1, 1, 1, input_shape[-1]), dtype=np.float32)
            elif len(input_shape) == 2:
                self.axis = 0
                self.reshape_shape = (1, -1)
                self.moving_mean = np.zeros((1, input_shape[-1]), dtype=np.float32)
                self.moving_var = np.ones((1, input_shape[-1]), dtype=np.float32)
            else:
                raise NotImplementedError
            self.beta = np.zeros(input_shape[-1], dtype=np.float32)
            self.gamma = np.ones(input_shape[-1], dtype=np.float32)
            self.is_built = True
    
    def moving_average_update(self, mu, var):
        # https://stats.stackexchange.com/questions/219808/how-and-why-does-batch-normalization-use-moving-averages-to-track-the-accuracy-o
        momentum = self.momentum
        
        self.moving_mean = momentum*self.moving_mean + (1-momentum)*mu
        self.moving_var = momentum*self.moving_var + (1-momentum)*var
    
    def get_output(self, prev_input, mu, var):
        if not self.trainable:
            self.mu = self.moving_mean
            self.var = self.moving_var
        else:
            self.mu = mu
            self.var = var
            
        return (prev_input-self.mu)/np.sqrt(self.var+self.eps)
        
    
    def forward(self, prev_input, training=True):
        if training:
            self.trainable = True
        else:
            self.trainable = False
            
        self.prev_input = prev_input
        self.prev_input_shape = prev_input.shape
        
        if not self.is_built:
            self.build(prev_input.shape)
            
        # Calculate mean and l1 variance
        mu = np.mean(prev_input, axis=self.axis)
        mu = mu.reshape(self.reshape_shape)
        
        prev_input_centered = prev_input - mu
        var = np.var(prev_input_centered, axis=self.axis)
        var = var.reshape(self.reshape_shape)
        
        # update moving stats at training mode only
        if self.trainable:
            self.moving_average_update(mu, var)
        self.out_std = self.get_output(prev_input, mu, var)
        
        return (self.gamma.reshape(self.reshape_shape)*self.out_std + self.beta.reshape(self.reshape_shape)).astype(self.dtype)
    
    def backprop(self, dout):
        # https://arxiv.org/pdf/1502.03167.pdf
        # weights update
        dgamma = np.sum(dout*self.out_std, axis=self.axis)
        dbeta = np.sum(dout, axis=self.axis)
        self.dbeta, self.dgamma = dbeta, dgamma
        
        # full precision backprob
        n = np.prod(self.prev_input_shape[:len(self.prev_input_shape)-1])
        dout_std = dout*self.gamma
        centered_prev_input = self.prev_input-self.mu
        ivar_eps = 1/np.sqrt(self.var+self.eps)
        
        dvar = -0.5*(ivar_eps**3)*np.sum(dout_std*centered_prev_input, axis=self.axis, keepdims=True)
        #print(dvar)
        dmu = np.sum(-dout_std*ivar_eps, axis=self.axis, keepdims=True) +dvar*np.mean(-2*centered_prev_input, axis=self.axis, keepdims=True)
        #print("dmu", dmu)
        dx = dout_std*ivar_eps + dvar*2*centered_prev_input/n + dmu/n
        #dx = -dout*dmu/n
        
        return dx
    
    def set_moving_mean(self, mean):
        self.moving_mean = mean
    
    def set_moving_var(self, var):
        self.moving_var = var
    
    def set_weights(self, w, b):
        self.beta = w
        self.gamma = b
        
        
        
class SlowConv2D(Layer):
    def __init__(self, filters, kernel_size, stride=(1, 1), padding="same", dtype=np.float32, trainable=True):
        """
        Args:
            filters (int): output_c
            kernel_size (int, int): kernel_h, kernel_w
            [NA] w (np.array): array with shape (kernel_h, kernel_w, input_c, output_c)
            [NA] b (np.array): array with shape (n, output_c)
            stride (int, int): stride alond (width, height)
            padding (str): "same" or "valid"
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype
        self.trainable = trainable
        
        self.w = None
        self.b = None
        self.is_built = False
        
        self._dw = None
        self._db = None

    def init_weights(self, prev_input_shape):
        np.random.seed(0)
        
        self.w = glorot_uniform_initializer(
            size=(self.kernel_size[0], self.kernel_size[1], prev_input_shape[-1], self.filters),
            n_input=np.prod(prev_input_shape[1:]),
            n_hidden_node=np.prod(self.kernel_size)*prev_input_shape[-1]*self.filters, 
            dtype=self.dtype
        )
        self.b = np.zeros(self.filters).astype(self.dtype)
        
        #assert self.w.dtype == self.dtype
        #assert self.b.dtype == self.dtype
        
        self.is_built = True
        
    
    @property
    def weights(self):
        if (self.w is None) or (self.b is None):
            return None
        return self.w, self.b
    
    @property
    def gradients(self):
        if (self._dw is None) or (self._db is None):
            return None
        return self._dw, self._db
    
    def forward(self, prev_input):
        if not self.is_built:
            self.init_weights(prev_input.shape)
        w, b = self.weights
        
        input_padder = Padding(
            input_shape=prev_input.shape, 
            kernel_shape=w.shape, 
            stride=self.stride, 
            padding=self.padding)
        self.pad = input_padder.pad
        
        n, h_out, w_out, _ = input_padder.output_shape
        self.prev_input = input_padder.pad_input(prev_input)
        output = np.zeros((n, h_out, w_out, self.filters))
        
        # package with c++ implementation -> python
        for i in range(h_out):
            for j in range(w_out):
                h_start = i*self.stride[0]
                h_end = h_start+self.kernel_size[0]
                
                w_start = j*self.stride[0]
                w_end = w_start+self.kernel_size[0]
                output[:, i, j, :] = np.tensordot(self.prev_input[:, h_start:h_end, w_start:w_end, :], w, axes=[(1, 2, 3), (0, 1, 2)])
        return output+b
                
    def backprop(self, output_gradient):
        """
        https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
        Args:
            dZ (np.array): The gradient of the cost with respect to the output of the conv layer Z)
                            shape -> (n, h, w, c)
        Returns:
            dprev_input: gradient of cost with respect to the input of the conv layer (prev_input)
                     shape -> (n, h_prev, w_prev, c_prev)
        """
        prev_input_shape = self.prev_input.shape
        _, h_grad, w_grad, _ = output_gradient.shape
        
        w, b = self.weights
        
        # Initialize gradient with the correct shapes
        dprev_input = np.zeros(self.prev_input.shape)
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)
        
        for i in range(h_grad):
            for j in range(w_grad):
                h_start = i
                h_end = h_start + self.kernel_size[0]
                
                w_start = j
                w_end = w_start + self.kernel_size[1]
                
                # apply gradient
                '''
                if input_shape (4, 5, 5, 2) 
                   w shape (3, 3, 2, 6)
                   output_gradient shape (4, 5, 5, 6)
                   dprev_input (shape (4, 3, 3, 2)) = tensordot(output_gradient (shape (4, 6)), w (shape (3, 3, 2, 6)))
                   dw (shape (3, 3, 2, 6)) = sum(output_gradient (shape (4, 1, 1, 1, 6)), prev_input (shape (4, 3, 3, 2, 1))))
                '''
                dprev_input[:, h_start:h_end, w_start:w_end, :] += np.tensordot(output_gradient[:, i, j, :], w, axes=[-1, -1])
                #print(np.tensordot(output_gradient[:, i, j, :], self.prev_input[:, h_start:h_end, w_start:w_end, :], axes=[0, 0]).shape)
                dw += np.sum(output_gradient[:, i, j, :][:, None, None, None, :]*self.prev_input[:, h_start:h_end, w_start:w_end, :][:, :, :, :, None], axis=0)
        db = np.sum(output_gradient, axis=(0, 1, 2))
        self._dw = dw
        self._db = db
        if self.pad == (0, 0):
                return dprev_input
        return dprev_input[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1], :]
    
    def set_weights(self, w, b):
        if self.trainable:
            self.w = w.astype(self.dtype)
            self.b = b.astype(self.dtype)
        
        #assert self.w.dtype == self.dtype
        #assert self.b.dtype == self.dtype
        