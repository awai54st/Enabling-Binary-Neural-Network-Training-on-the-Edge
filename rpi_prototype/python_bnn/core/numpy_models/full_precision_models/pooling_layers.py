import numpy as np

from core.base_layers import Layer
from core.utils.bit_packing_utils import PackBits



'''
from core.cython.layer_utils_new import max_pool
class MaxPooling(Layer):
    def __init__(self, kernel_size, stride, pad=(0, 0), dtype=np.float32):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
        
    def forward(self, prev_input, pad=(0,0), training=True):
        """
        Args:
            prev_input: n, h, w, c
        """
        prev_input = prev_input.astype(self.dtype)
        
        if training:
            out, self.mask = max_pool(prev_input, list(self.kernel_size), list(self.stride), training=True)
        else:
            out = max_pool(prev_input, list(self.kernel_size), list(self.stride), training=False)
            
        return out

    def backprop(self, dout):
        """
        Args:
            dout: n, h, w, c
        """
        dout = dout.astype(self.dtype)
        return np.repeat(dout, self.stride[0], axis=1).repeat(self.stride[1], axis=2)*self.mask

    
    def set_weights(self):
        pass
    
'''
    
    
    
    
class Pooling(Layer):
    def __init__(self, kernel_size, stride, mode="MAX", dtype=np.float32):
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.dtype = dtype
        self.prev_input_col = np.array(0, dtype)
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    def forward(self, prev_input, training=True):
        prev_input = prev_input.astype(self.dtype)
        output_shape = self.output_shape(prev_input.shape)
        
        output = np.zeros(output_shape)
        output_mask = np.zeros(prev_input.shape)
        
        for i in range(output_shape[1]):
            for j in range(output_shape[2]):
                # Find the corners of the current "slice"
                h_start = i*self.stride[0]
                h_end = h_start +self.kernel_size[0]
                w_start = j*self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                if self.mode == "MAX":
                    max_val = np.max(prev_input[:, h_start:h_end, w_start:w_end, :], axis=(1,2), keepdims=True)
                    output[:, i, j, :] = np.squeeze(max_val)
                    output_mask[:, h_start:h_end, w_start:w_end, :] = prev_input[:, h_start:h_end, w_start:w_end, :] == max_val
                
                elif self.mode == "AVERAGE":
                    output[:, i, j, :] = np.mean(prev_input[:, h_start:h_end, w_start:w_end, :], axis=(1,2))
        self.output_mask = output_mask
        
        return output.astype(self.dtype)
    
    def backprop(self, dout):
        dout = dout.astype(self.dtype)
        input_grad = np.zeros(self.output_mask.shape)
        n, h_out_grad, w_out_grad, _ = dout.shape
        
        # for average pooling
        if self.mode == "AVERAGE":
            total_kernel_elements = np.prod(self.kernel_size)
            avg_dout = dout/total_kernel_elements
        
        for i in range(h_out_grad):
            for j in range(w_out_grad):
                # Find the corners of the current "slice"
                h_start = i*self.stride[0]
                h_end = h_start +self.kernel_size[0]
                w_start = j*self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                if self.mode == "MAX":
                    input_grad[:, h_start:h_end, w_start:w_end, :] += dout[:, i, j, :][:, None, None, :]*self.output_mask[:, h_start:h_end, w_start:w_end, :]
                elif self.mode == "AVERAGE":
                    input_grad[:, h_start:h_end, w_start:w_end, :] += avg_dout[:, i, j, :][:, None, None, :]/np.ones_like(self.output_mask[:, h_start:h_end, w_start:w_end, :])
        
        return input_grad.astype(self.dtype)
    
    def set_weights(self):
        pass
    
    def output_shape(self, input_shape):
        n, h, w, c = input_shape
        new_h = int(1+(h-self.kernel_size[0]) / self.stride[0])
        new_2 = int(1+(h-self.kernel_size[1]) / self.stride[1])
        return n, new_h, new_2, c
        
class MaxPoolingSlow(Pooling):
    def __init__(self, kernel_size, stride, dtype=np.float32):
        super().__init__(kernel_size, stride, mode="MAX", dtype=dtype)
        
class AveragePooling(Pooling):
    def __init__(self, kernel_size, stride):
        super().__init__(kernel_size, stride, mode="AVERAGE")
        

    
class MaxPooling(Layer):
    def __init__(self, kernel_size, stride, pad=(0, 0), dtype=np.float32):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        self.pack = PackBits(min_val=0)
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
        
    def forward(self, prev_input, pad=(0,0), training=True):
        """
        Args:
            prev_input: n, h, w, c
        """
        prev_input = prev_input.astype(np.float32)
        (n, h, w, c) = prev_input.shape
        prev_input_strides = prev_input.strides
        
        pad = self.pad
        kernel_size = self.kernel_size
        stride = self.stride
        
        out_h = int((h + 2 * pad[0] - kernel_size[0]) / stride[0]) + 1
        out_w = int((w + 2 * pad[1] - kernel_size[1]) / stride[1]) + 1
        
        windows = np.lib.stride_tricks.as_strided(
            prev_input,
            shape=(n, out_h, out_w, self.kernel_size[0], self.kernel_size[1], c),
            strides=(prev_input_strides[0], 
                     self.stride[0] * prev_input_strides[1],
                     self.stride[1] * prev_input_strides[2],
                     prev_input_strides[1], prev_input_strides[2], prev_input_strides[3])
        )
        out = np.max(windows, axis=(3, 4), keepdims=True)
        
        if training:
            kernel_size = self.kernel_size[0]*self.kernel_size[1]
            broadcast_shape = (n, out_h, out_w, kernel_size, c)
            out_max = windows.reshape(broadcast_shape)
            out_max_idx = np.argmax(out_max, axis=3)
            
            broadcasted = np.broadcast_to(np.arange(kernel_size).reshape(1, 1, 1, kernel_size, 1), broadcast_shape)
            mask = broadcasted==out_max_idx[:, :, :, None, :]
            
            #self.mask = np.where(test_MaxPooling.out == out, 1, 0).transpose(0, 1, 3, 2, 4, 5).reshape((n, h, w, c))
            mask = mask.reshape(n, out_h, out_w, self.kernel_size[0], self.kernel_size[1], c)
            self.mask = self.pack.pack_bits(mask)
        return out[:, :, :, 0, 0, :]

    def backprop(self, dout):
        """
        Args:
            dout: n, h, w, c
        """
        dout = dout.astype(np.float32)
        dout_strides = dout.strides
        (n, h, w, c) = dout.shape
        dout_patch = np.lib.stride_tricks.as_strided(
            dout,
            shape=(n, h, w, 1, 1, c),
            strides=(dout_strides[0], dout_strides[1], dout_strides[2],
                     dout_strides[1], dout_strides[2], dout_strides[3])
        )
        
        mask = self.pack.unpack_bits(self.mask)
        dout = dout_patch*mask
        #TODO: get prev shape, pad output for odd windows
        return dout.transpose(0, 1, 3, 2, 4, 5).reshape((n, h*self.stride[0], w*self.stride[0], c))

    
    def set_weights(self):
        pass
    
    
class MaxPoolingError(Layer):
    def __init__(self, kernel_size, stride, pad=(0, 0), dtype=np.float32):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
        
    def forward(self, prev_input, pad=(0,0), training=True):
        """
        Args:
            prev_input: n, h, w, c
        """
        prev_input = prev_input.astype(self.dtype)
        pad = self.pad
        kernel_size = self.kernel_size
        stride = self.stride
        
        prev_input_t = prev_input.transpose(0, 3, 1, 2)
        n, c, h, w = prev_input_t.shape
        self.prev_input_t_shape = (n, h, w, c)

        out_h = int((h + 2 * pad[0] - kernel_size[0]) / stride[0]) + 1
        out_w = int((w + 2 * pad[1] - kernel_size[1]) / stride[1]) + 1

        # reshape to convert fully to column
        prev_input_t_reshape = prev_input_t.reshape(n*c, 1, h, w)

        self.prev_input_col = im2col(
            prev_input_t_reshape, kernel_size=kernel_size, 
            stride=stride, pad=pad)
        #print(self.prev_input_col.shape)
        
        # get max index of each window
        self.max_idx = np.argmax(self.prev_input_col, axis=0)
        out = self.prev_input_col[self.max_idx, range(self.max_idx.size)]
        
        # reshape to h, w, n, c
        return out.reshape(n, c, out_h, out_w).transpose(0, 2, 3, 1).astype(self.dtype)

    def backprop(self, dout):
        """
        Args:
            dout: n, h, w, c
        """
        dout = dout.astype(np.float32)
        dprev_input_col = np.zeros_like(self.prev_input_col)
        
        # reshape to h, w, n, c then flatten
        dout_flat = dout.transpose(1, 2, 0, 3).ravel()
        
        # Fill maximum index of each column with gradient
        dprev_input_col[self.max_idx, range(self.max_idx.size)] = dout_flat
        
        # undo col2im operation
        # output nxc, 1, h, w
        n, h, w, c = self.prev_input_t_shape
        dprev_input = col2im(
            dX_col=dprev_input_col, X_shape=(n*c, 1, h, w), 
            kernel_size=self.kernel_size, stride=self.stride, pad=self.pad)
        return dprev_input.reshape((n, c, h, w)).transpose(0, 2, 3, 1).astype(self.dtype)
        
        #return dprev_input
        #dprev_input = dprev_input.reshape(n, -1)
        #dprev_input = np.array(np.hsplit(dprev_input, c))
        #return dprev_input.reshape((n, c, h, w)).transpose(0, 2, 3, 1)
    
    def set_weights(self):
        pass
    
    def output_shape(self, input_shape):
        n, h, w, c = input_shape
        new_h = int(1+(h-self.kernel_size[0]) / self.stride[0])
        new_2 = int(1+(h-self.kernel_size[1]) / self.stride[1])
        return n, new_h, new_2, c
        
    

# https://hackmd.io/@bouteille/B1Cmns09I
def get_indices(X_shape, kernel_size, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -kernel_size: filter (height, width)
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad[0] - kernel_size[0]) / stride[0]) + 1
    out_w = int((n_W + 2 * pad[1] - kernel_size[1]) / stride[1]) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(kernel_size[0]), kernel_size[1])
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride[0] * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(kernel_size[1]), kernel_size[0])
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride[1] * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), kernel_size[0] * kernel_size[1]).reshape(-1, 1)

    return i, j, d

def im2col(X, kernel_size, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - kernel_size: filter (height, width)
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad[0], pad[0]), (pad[1], pad[1])), mode='constant')
    i, j, d = get_indices(X.shape, kernel_size, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, kernel_size, stride, pad):
    """
        Transform our matrix back to the input image.

        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - kernel_size: filter (height, width)
        - stride: stride value.
        - pad: padding value.

        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded = H + 2 * pad[0]
    W_padded = W + 2 * pad[1]
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    # Index matrices, necessary to transform our input image into a matrix. 
    i, j, d = get_indices(X_shape, kernel_size, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad[0] == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]