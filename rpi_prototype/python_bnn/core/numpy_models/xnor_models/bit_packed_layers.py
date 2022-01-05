import numpy as np

from core.base_layers import Layer
from core.numpy_models.full_precision_models.layers import Conv2D, Dense, Padding

from core.numpy_models.binary_models.binarization_utils import binarize_weight

#from core.cython.convolution import convolution_mem_view as faster_convolution

from core.utils.conv_utils import faster_convolution#, faster_full_convolution, faster_backprop_dw
#from core.test.utils.test_conv_utils import slow_convolution as faster_convolution
from core.numpy_models.xnor_models.gradient_quantization_utils import log2, LOG_quantize
from core.utils.bit_packing_utils import PackBits

    
class XNorDense(Dense):
    def __init__(self, output_units, first_layer=False, thr=1, trainable=True, dtype=np.float32, use_bias=False, random_seed=0):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        super().__init__(output_units=output_units, dtype=dtype, trainable=trainable, random_seed=random_seed, use_bias=use_bias)
        self.thr = dtype(thr)
        self.first_layer = first_layer
        if dtype == np.float16:
            self.eps = dtype(1e-7)
        else:
            self.eps = dtype(1e-45)
        
        self.pack = PackBits()
        
    def forward(self, prev_input, training=True):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        prev_input = prev_input.astype(np.float32)
        
        if not self.is_built:
            self.init_weights(prev_input.shape[-1])
            
        output = np.dot(
            prev_input,
            binarize_weight(self.w.astype(np.float16), dtype=np.float32)
        )
        
        if not self.first_layer:
            self.prev_input = self.pack.pack_bits(prev_input+self.eps)
            #self.prev_input = np.sign(prev_input+self.eps)
        else:
            # first layer
            self.prev_input = prev_input.astype(np.float32)
        
        if self.use_bias:
            return output+self.b
        else:
            return output
        
    def backprop(self,dout):
        dout = dout.astype(np.float32)
        w = self.w.astype(np.float32)
        w_mask = (np.abs(w) <= self.thr)
        rounded_w = binarize_weight(w.astype(np.float16), dtype=np.float32)
        
        # constant with type casting
        EIGHT = np.float32(8)
        
        dout_max = np.max(np.abs(dout))
        dout_bias = -np.round(log2(dout_max, dtype=np.float32))+EIGHT
        
        dout *= (2**dout_bias)
        dout = LOG_quantize(dout, 4.0, dtype=np.float32)
        dout *= 2 ** (-dout_bias)
        
        grad_input = np.dot(dout, rounded_w.T)
        # compute gradient w.r.t. weights and biases
        
        if not self.first_layer:
            prev_input = self.pack.unpack_bits(self.prev_input)
            #prev_input = self.prev_input
        else:
            prev_input = self.prev_input
            
        dw = np.dot(prev_input.T, dout)
        N = self.w.size
        dw = 1./np.sqrt(N) * np.sign(dw + self.eps)*w_mask
        self._dw = dw.astype(np.float16).astype(self.dtype)
        
        
        if self.use_bias:
            b_mask = (np.abs(self.b)<self.thr)
            #db = dout.mean(axis=0)*self.prev_input.shape[0]*b_mask
            db = self.b*b_mask
            self._db = db.astype(self.dtype)
        
        return grad_input
    

class XNorConv2D(Conv2D):
    def __init__(self, filters, kernel_size, stride=[1, 1], padding="same", thr=1, first_layer=False, dtype=np.float32, trainable=True, use_bias=False, pad_mode="constant", random_seed=0):
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
            filters=filters, kernel_size=kernel_size, stride=stride, use_bias=use_bias,
            padding=padding, dtype=dtype, trainable=trainable, random_seed=random_seed)
        self.thr = thr
        self.pad_mode = pad_mode
        
        self.first_layer = first_layer
        if dtype == np.float16:
            self.eps = dtype(1e-7)
        else:
            self.eps = dtype(1e-45)
        self.pack = PackBits()
    
    def forward(self, prev_input, training=True):
        prev_input = prev_input.astype(np.float32)
        self.prev_input_shape = prev_input.shape
        
        if not self.is_built:
            self.init_weights(self.prev_input_shape)
        
        input_padder = Padding(
            input_shape=prev_input.shape, 
            kernel_shape=self.w.shape, 
            stride=self.stride, 
            padding=self.padding)
        self.pad = input_padder.pad
        
        n, h_out, w_out, _ = input_padder.output_shape
        
        prev_input_padded = input_padder.pad_input(prev_input, mode=self.pad_mode)
        binarized_w = binarize_weight(self.w.astype(np.float16), dtype=np.float32)
        
        #output = faster_convolution(prev_input_padded, binarized_w, (1,1), dtype=self.dtype)
        output = faster_convolution(prev_input_padded, binarized_w, (1,1), dtype=np.float32)
        #self.x = prev_input_padded
        if not self.first_layer:
            self.prev_input = self.pack.pack_bits(prev_input_padded+self.eps)
            #self.prev_input = np.sign(prev_input_padded+self.eps)
        else:
            self.prev_input = prev_input_padded.astype(self.dtype)
            
        if self.use_bias:
            return output+self.b
        else:
            return output
        
    def backprop(self, dout):
        """
        https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
        Args:
            dZ (np.array): The gradient of the cost with respect to the output of the conv layer Z)
                            shape -> (n, h, w, c)
        Returns:
            dprev_input: gradient of cost with respect to the input of the conv layer (prev_input)
                     shape -> (n, h_prev, w_prev, c_prev)
        """
        dout = dout.astype(np.float32)
        w_mask = (np.abs(self.w) <= self.thr).astype(np.float32)
        rounded_w = binarize_weight(self.w.astype(np.float16), dtype=np.float32)
        
        w_rotated = rounded_w[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
        w_shape = rounded_w.shape
        
        # Padding
        dout_padded = np.pad(
            array=dout, 
            pad_width=((0, 0), (w_shape[0]-1, w_shape[0]-1), (w_shape[1]-1, w_shape[1]-1), (0, 0)),
            mode = "constant"
        )
        
        # constant with type casting
        EIGHT = np.float32(8)
        
        # PO2 dy
        dout_max = np.max(np.abs(dout_padded))
        dout_bias = -np.round(log2(dout_max, dtype=np.float32)) + EIGHT
        
        dout_padded *= 2 ** dout_bias
        dout_padded = LOG_quantize(dout_padded, 4., dtype=np.float32)
        dout_padded *= 2 ** -dout_bias
        
        #dprev_input = faster_convolution(
        #    x=dout_padded, w=w_rotated, stride=(1,1), pad=(0,0), dtype=self.dtype)
        dprev_input = faster_convolution(
            x=dout_padded, w=w_rotated, stride=(1,1), pad=(0,0), dtype=np.float32)
        
        # unpack
        if not self.first_layer:
            prev_input = self.pack.unpack_bits(self.prev_input)
            #prev_input = self.prev_input
        else:
            prev_input = self.prev_input.astype(np.float32)
            
        """if self.padding == "same":
            # Padding
            x_pad_number = [[0, 0,], [1, 1,],[1, 1,], [0, 0]]
            x_trans = np.pad(prev_input, x_pad_number).transpose(3, 1, 2, 0)
        # For "VALID" padding mode
        elif self.padding == "valid":
            x_trans = prev_input.transpose(3, 1, 2, 0)"""
        x_trans = prev_input.transpose(3, 1, 2, 0)
        dout_trans = dout.transpose(1, 2, 0, 3)
        # PO2 dy
        dout_trans_max = np.max(np.abs(dout_trans))
        dout_trans_bias = -np.round(log2(dout_trans_max, dtype=np.float32)) + EIGHT
        
        dout_trans *= (2 ** dout_trans_bias)
        dout_trans = LOG_quantize(dout_trans, 4., dtype=np.float32)
        dout_trans *= (2 ** -dout_trans_bias)
        
        #dw = faster_convolution(x_trans, dout_trans, stride=(1,1), dtype=self.dtype)
        dw = faster_convolution(x_trans, dout_trans, stride=(1,1), dtype=np.float32)
        dw = dw.transpose(1, 2, 0, 3)
                
        N = rounded_w.size
        dw = 1/np.sqrt(N) * np.sign(dw+self.eps)
        dw = dw.astype(np.float16)
        
        self._dw = self.dtype(dw*w_mask)
        if self.use_bias:
            b_mask = (np.abs(self.b) <= self.thr)
            db = np.array(0.)*b_mask # np.sum(output_gradient, axis=(0, 1, 2))
            self._db = self.dtype(db*b_mask)
        
        # undo padding
        if self.pad == (0, 0):
            return dprev_input
        return dprev_input[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1], :]
    
    
class BatchNorm(Layer):
    """
    https://arxiv.org/pdf/1502.03167.pdf
    
    test_l1_batch_norm_mod_conv
    """
    def __init__(self, momentum, trainable=True, dtype=np.float32):
        self.momentum = dtype(momentum)
        self.trainable = trainable
        self.is_built = False
        self.dtype = dtype
        if dtype == np.float16:
            self.eps = dtype(1e-7)
        else:
            self.eps = dtype(1e-45)
        
        self.beta = None
        self.dbeta = None
        self.pack = PackBits()
    
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
        
        self.moving_mean = self.dtype(momentum*self.moving_mean + (1.0-momentum)*mu)
        self.moving_var = self.dtype(momentum*self.moving_var + (1.0-momentum)*var)
    
    def get_output(self, prev_input, mu, var):
        if not self.trainable:
            self.mu = self.moving_mean.astype(self.dtype)
            self.var = self.moving_var.astype(self.dtype)
        else:
            self.mu = mu.astype(self.dtype)
            self.var = var.astype(self.dtype)
        return np.float32(1)/self.var*(prev_input-self.mu)
        
    
    def forward(self, prev_input, training=True):
        prev_input = prev_input.astype(np.float32)
        
        self.trainable = training
            
        self.prev_input_shape = prev_input.shape
        prev_input_shape_len = len(self.prev_input_shape)
        
        if not self.is_built:
            self.build(prev_input.shape)
            
        N = np.float32(np.prod(self.prev_input_shape[:prev_input_shape_len-1]))
        ONE = np.float32(1)
        
        # Calculate mean and l1 variance
        mu = ONE/N*np.sum(prev_input, axis=self.axis)
        mu = mu.reshape(self.reshape_shape)
        
        prev_input_centered = prev_input - mu
        var = ONE/N*np.sum(np.abs(prev_input_centered), axis=self.axis, dtype=np.float32)
        
        var = var.reshape(self.reshape_shape)
        assert not np.isnan(1/var).any()
        assert not np.isinf(1/var).any()
        
        # TODO: Cast moving stats to FP16 to simulate fp16 storage
        mu = mu.astype(np.float16)
        var = var.astype(np.float16)
        mu = mu.astype(np.float32)
        var = var.astype(np.float32)
        
        beta = self.beta.astype(np.float16)
        beta = beta.astype(np.float32)
        
        # update moving stats at training mode only
        if self.trainable:
            self.moving_average_update(mu, var)
        out = self.get_output(prev_input, mu, var)
        
        if training == True:
            self.out = self.pack.pack_bits(out+1e-37)
            self.out_mean = self.dtype(1/N*np.sum(np.abs(out, dtype=np.float32), axis=self.axis).reshape(self.reshape_shape))
            
        return out + beta.reshape(self.reshape_shape)
    
    def backprop(self, dout):
        # TODO: Cast dX to FP16 to simulate FP16 storage
        dout = dout.astype(np.float16).astype(np.float32)
        dout_shape = dout.shape
        dout_shape_len = len(dout_shape)
        
        N = np.prod(dout_shape[:dout_shape_len-1]).astype(np.float32)
        ONE = np.float32(1)
        
        # dbeta
        self.dbeta = np.sum(dout, axis=self.axis).astype(self.dtype)
        #self.dbeta = 0
        
        # BN backprob
        out = self.pack.unpack_bits(self.out)
        
        dy_norm_x = dout/self.var
        
        #term_1 = dy_norm_x - np.mean(dy_norm_x, axis=self.axis).reshape(self.reshape_shape)
        term_1 = dy_norm_x - ONE/N * np.sum(dy_norm_x, axis=self.axis).reshape(self.reshape_shape)
        
        # term 2 = self.out # vanilla BN
        term_2 = out
        
        # term 3 = 1/N * np.sum(dy_norm_x*self.out, axis=[0, 1, 2]) # Vanilla BN
        """
        term_3 = ONE/N*np.sum(
            np.sign(self.out)*ONE/N*np.sum(np.abs(self.out, dtype=self.dtype), axis=self.axis).reshape(self.reshape_shape)*dy_norm_x,
            axis = self.axis
        )
        """
        term_3 = ONE/N*np.sum(
            out*dy_norm_x*self.out_mean,
            axis = self.axis
        )
        
        term_3 = term_3.reshape(self.reshape_shape)
        
        dprev_input = term_1-term_2*term_3
        
        return dprev_input
    
    def set_weights(self, w):
        self.beta = w[0].astype(self.dtype)
        