import numpy as np

from core.base_layers import Layer
from core.numpy_models.full_precision_models.layers import Conv2D, Dense, Padding
from core.numpy_models.binary_models.binarization_utils import binarize_weight
from core.utils.conv_utils import faster_convolution


class BinaryConv2D(Conv2D):
    def __init__(self, filters, kernel_size, stride=(1, 1), padding="same", thr=1, dtype=np.float32, trainable=True, use_bias=False, random_seed=0):
        """
        Args:
            filters (int): output_c
            kernel_size (int, int): kernel_h, kernel_w
            [NA] w (np.array): array with shape (kernel_h, kernel_w, input_c, output_c)
            [NA] b (np.array): array with shape (n, output_c)
            stride (int, int): stride alond (width, height)
            padding (str): "same" or "valid"
            thr (float): weight backprop threshold
        """
        super().__init__(
            filters=filters, kernel_size=kernel_size, stride=stride, 
            padding=padding, dtype=dtype, trainable=trainable, random_seed=random_seed,
            use_bias=use_bias)
        self.thr = thr
    
    def forward(self, prev_input, training=True):
        prev_input = prev_input.astype(np.float32)
        
        if not self.is_built:
            self.init_weights(prev_input.shape)
        w = self.w.astype(self.dtype)
        
        input_padder = Padding(
            input_shape=prev_input.shape, 
            kernel_shape=w.shape, 
            stride=self.stride, 
            padding=self.padding)
        self.pad = input_padder.pad
        
        n, h_out, w_out, _ = input_padder.output_shape
        prev_input = input_padder.pad_input(prev_input, mode="constant")
        binarized_w = binarize_weight(w, dtype=self.dtype)
        
        output = faster_convolution(prev_input, binarized_w, (1,1), dtype=np.float32)
        
        self.prev_input = prev_input.astype(self.dtype)
        """
        if self.use_bias:
            return output+self.b.astype(self.dtype)
        else:
            return output.astype(self.dtype)"""
        return output
        
                
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
        output_gradient = np.float32(output_gradient)
        prev_input_shape = self.prev_input.shape
        w_mask = self.dtype(np.abs(self.w)<self.thr)
        
        w = self.w.astype(self.dtype)
        rounded_w = binarize_weight(w, dtype=self.dtype)
        
        
        # Padding
        w_shape = rounded_w.shape
        w_rotated = rounded_w[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
        
        dout_padded = np.pad(
            array=output_gradient, 
            pad_width=((0, 0), (w_shape[0]-1, w_shape[0]-1), (w_shape[1]-1, w_shape[1]-1), (0, 0)),
            mode = "constant"
        )
        dprev_input = faster_convolution(
            x=dout_padded, w=w_rotated, stride=(1,1), pad=(0,0), dtype=np.float32)
        dout_padded = None
        #dprev_input = faster_full_convolution(dout=output_gradient, w=rounded_w, stride=self.stride)
        
        #dw = faster_backprop_dw(grad_output=output_gradient, prev_input=self.prev_input, kernel_shape=w.shape, stride=self.stride)
        x_trans = self.prev_input.transpose(3, 1, 2, 0)
        dout_trans = output_gradient.transpose(1, 2, 0, 3)
        dw = faster_convolution(x_trans, dout_trans, stride=(1,1), dtype=np.float32)
        dw = dw.transpose(1, 2, 0, 3)
        
        self._dw = (dw*w_mask).astype(self.dtype)
        if self.use_bias:
            b_mask = self.dtype(np.abs(self.b)<self.thr)
            self._db = b_mask*np.sum(output_gradient, axis=(0, 1, 2)).astype(self.dtype)
        else:
            self._db = self.dtype(0)
                
        if self.pad == (0, 0):
            return dprev_input
            
        return dprev_input[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1], :]

        
        
class BinaryDense(Dense):
    def __init__(self, output_units, thr=1, dtype=np.float32, trainable=True, use_bias=False, random_seed=0):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        super().__init__(output_units=output_units, dtype=dtype, trainable=trainable, 
                         random_seed=random_seed, use_bias=use_bias)
        self.thr = thr
        
    def forward(self, prev_input, training=True):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        self.prev_input = prev_input.astype(np.float32)
        if not self.is_built:
            self.init_weights(prev_input.shape[-1])
            
        output = np.dot(prev_input,binarize_weight(self.w, dtype=np.float32))
        prev_input = None
        if self.use_bias:
            return output+self.b
        else:
            return output
    
    def backprop(self,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        w_mask = np.abs(self.w)<self.thr
        rounded_w = binarize_weight(self.w, dtype=self.dtype)
        #rounded_w = np.clip(rounded_w, -1, 1)
        
        grad_input = np.dot(grad_output, rounded_w.T)
        
        # compute gradient w.r.t. weights and biases
        self._dw = (np.dot(self.prev_input.T, grad_output)*w_mask).astype(self.dtype)
        
        if self.use_bias:
            b_mask = np.abs(self.b)<self.thr
            self._db = (grad_output.mean(axis=0)*self.prev_input.shape[0]*b_mask).astype(self.dtype)
        else:
            self._db = self.dtype(0)
        
        return grad_input

    
    
class BatchNorm(Layer):
    """
    test_l1_batch_norm_mod_conv
    """
    def __init__(self, momentum, trainable=True, eps=1e-5, dtype=np.float32):
        self.momentum = dtype(momentum)
        self.trainable = trainable
        self.is_built = False
        self.eps = dtype(eps)
        self.dtype = dtype
        
        
        self.beta = None
        self.dbeta = None
    
    
    @property
    def weights(self):
        if self.beta is None:
            return None
        return [self.beta]
    
    @property
    def gradients(self):
        if self.dbeta is None:
            return None
        return [self.dbeta]
        
    def build(self, input_shape):
        if not self.is_built:
            if len(input_shape) == 4:
                self.axis = (0, 1, 2)
                self.reshape_shape = (1, 1, 1, -1)
                self.moving_mean = np.zeros((1, 1, 1, input_shape[-1]), dtype=self.dtype)
                self.moving_var = np.ones((1, 1, 1, input_shape[-1]), dtype=self.dtype)
            elif len(input_shape) == 2:
                self.axis = 0
                self.reshape_shape = (1, -1)
                self.moving_mean = np.zeros((1, input_shape[-1]), dtype=self.dtype)
                self.moving_var = np.ones((1, input_shape[-1]), dtype=self.dtype)
            else:
                raise NotImplementedError
            self.beta = np.zeros(input_shape[-1], dtype=self.dtype)
            self.is_built = True
    
    def moving_average_update(self, mu, var):
        # https://stats.stackexchange.com/questions/219808/how-and-why-does-batch-normalization-use-moving-averages-to-track-the-accuracy-o
        momentum = self.momentum
        
        self.moving_mean = self.dtype(momentum*self.moving_mean + (1-momentum)*mu)
        self.moving_var = self.dtype(momentum*self.moving_var + (1-momentum)*var)
    
    def get_output(self, prev_input, mu, var):
        if not self.trainable:
            self.mu = self.moving_mean
            self.var = self.moving_var
        else:
            self.mu = mu
            self.var = var
        
        return (prev_input-self.mu)/self.var
        
    
    def forward(self, prev_input, training=True):
        self.trainable = training
        
        self.prev_input_shape = prev_input.shape
        
        if not self.is_built:
            self.build(prev_input.shape)
            
        # Calculate mean and l1 variance
        mu = np.mean(prev_input, axis=self.axis)
        mu = mu.reshape(self.reshape_shape)
        
        prev_input_centered = prev_input - mu
        var = np.mean(np.abs(prev_input_centered), axis=self.axis)#+self.eps
        var = var.reshape(self.reshape_shape)
        # update moving stats at training mode only
        if self.trainable:
            self.moving_average_update(mu, var)
        self.out = self.get_output(prev_input, mu, var).astype(self.dtype)
        
        return self.out + self.beta.reshape(self.reshape_shape)
    
    def backprop(self, grad_output):
        grad_output = grad_output.astype(np.float32)
        grad_output = grad_output.astype(self.dtype)
        
        # dbeta
        self.dbeta = np.sum(grad_output, axis=self.axis, dtype=self.dtype)
        
        # BN backprob
        dy_norm_x = grad_output/self.var
        
        grad_output_shape = grad_output.shape
        N = self.dtype(np.prod(grad_output_shape[:len(grad_output_shape)-1]))
        term_1 = -1/N*np.sum(dy_norm_x, axis=self.axis).reshape(self.reshape_shape)+dy_norm_x
        term_2 = self.out # vanilla BN
        term_3 = np.sum(dy_norm_x*self.out, axis=self.axis)/N # Vanilla BN
        term_3 = term_3.reshape(self.reshape_shape)
        dprev_input = term_1-term_2*term_3
        
        
        return dprev_input
    
    def set_moving_mean(self, mean):
        self.moving_mean = mean.astype(self.dtype)
    
    def set_moving_var(self, var):
        self.moving_var = var.astype(self.dtype)
    
    def set_weights(self, w):
        self.beta = w[0].astype(self.dtype)
        