import numpy as np
import time
from numpy.lib.stride_tricks import as_strided#, sliding_window_view

'''def faster_convolution(x, w, output_shape=None, stride=(1,1), pad=(0,0), dtype=np.float32):
    """
    Fast implementation of convolution.
    
    Args:
        x: numpy array with shape (n, h, w, c)
        w: weights
    """
    w = w.astype(dtype)
    x = x.astype(dtype)
    
    w_shape = w.shape
    n, h_prev, w_prev, c_prev = x.shape
    
    if output_shape is None:
        Hout = int((h_prev + 2 * pad[0] - w_shape[0])/ stride[0]) + 1
        Wout = int((w_prev + 2 * pad[1] - w_shape[1])/ stride[1]) + 1
    else:
        Hout = output_shape[1]
        Wout = output_shape[2]
        
    x_strides = x.strides
    
    x_strided = as_strided(
        x=x, 
        shape=(x.shape[0], Hout, Wout, w_shape[0], w_shape[1], w_shape[2]), 
        strides= [x_strides[0], x_strides[1]*stride[0], x_strides[2]*stride[1],
                  x_strides[1], x_strides[2], x_strides[3]
                 ]
    )
    
    return np.tensordot(x_strided, w, axes=3)'''

def faster_convolution(x, w, stride=(1,1), pad=(0,0), dtype=np.float32, output_shape=None):
    """
    Fast implementation of convolution.
    
    Args:
        x: numpy array with shape (n, h, w, c)
        w: weights
    """
    w = w.astype(dtype)
    x = x.astype(dtype)
    
    w_shape = w.shape
    n, h_prev, w_prev, c_prev = x.shape
    
    if output_shape is None:
        Hout = int((h_prev + 2 * pad[0] - w_shape[0])/ stride[0]) + 1
        Wout = int((w_prev + 2 * pad[1] - w_shape[1])/ stride[1]) + 1
    else:
        Hout = output_shape[1]
        Wout = output_shape[2]
        
    x_strides = x.strides
    
    x_strided = as_strided(
        x=x, 
        shape=(x.shape[0], Hout, Wout, w_shape[0], w_shape[1], w_shape[2]), 
        strides= [x_strides[0], x_strides[1]*stride[0], x_strides[2]*stride[1],
                  x_strides[1], x_strides[2], x_strides[3]
                 ]
    )
    
    return np.tensordot(x_strided, w, axes=3)
    #return np.einsum("nhwklc, klco -> nhwo", x_strided, w)



def dilate_array(arr, stride, dtype=np.float32):
    """
    Pad each element with 0s
    1, 1       1, 0, 1
    1, 1   ->  0, 0, 0,
               1, 0, 1
    """
    arr = arr.astype(dtype)
    
    if stride == (1, 1):
        return arr
    
    n, h, w, c = arr.shape
    arr_padded = np.zeros((n, h*stride[0], w*stride[1], c), dtype=arr.dtype)
    
    # generate index of arr in arr_padded
    idx_h = np.arange(arr.shape[1]*stride[0], step=stride[0])
    idx_w = np.arange(arr.shape[2]*stride[1], step=stride[1])
    idx = np.array(np.meshgrid(idx_h, idx_w)).T.reshape(-1, 2)
    
    # assign original arr value to arr_padded (zeros)
    arr_padded[:, idx[:, 0], idx[:, 1], :] = arr.reshape(n, -1, c)
    
    return arr_padded[:, :-(stride[0]-1), :-(stride[1]-1), :]


def faster_full_convolution(dout, w, prev_input_shape=None, stride=(1,1), pad=(0, 0), dtype=np.float32):
    """
    Bit packing method of full_convolution. Normally use to calculate dprev_input
    
    Args:
        w: kernel/filter with shape (kernel_h, kernel_w, c_in, c_out)
        dout: output gradient with shape (n, h, w, c)
    """
    dout = dout.astype(dtype)
    w = w.astype(dtype)
    
    n, h_out, w_out, c_out = dout.shape
    w_shape = w.shape
    if prev_input_shape is None:
        h_prev = int((h_out-1)*stride[0] - 2 * pad[0] + w_shape[0])
        w_prev = int((w_out-1)*stride[1] - 2 * pad[1] + w_shape[1])
        prev_input_shape = (n, h_prev, w_prev, w_shape[2])
        
    w_rotated = w.transpose(0, 1, 3, 2)[::-1, ::-1, :, :]
    pad_h = w_shape[0]-1
    pad_w = w_shape[1]-1
    
    #pad_h = (prev_input_shape[1]-1)*stride[0] - h_out + w.shape[0]
    #pad_w = (prev_input_shape[2]-1)*stride[1] - w_out + w.shape[1]
    
    dout_padded = dilate_array(arr=dout, stride=stride)
    
    dout_padded = np.pad(
        array = dout_padded,
        pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode = "constant"
    )
    return faster_convolution(
        x=dout_padded, w=w_rotated, output_shape=prev_input_shape, stride=(1,1), pad=(0,0), dtype=dtype)


def faster_backprop_dw(grad_output, prev_input, kernel_shape, stride=(1,1), pad=(0, 0), dtype=np.float32):
    """
    Fast implementation of convolution.
    convolution(input x, loss gradient)
    
    Args:
        x: numpy array with shape (n, h, w, c)
        w: weights
    """
    grad_output = grad_output.astype(dtype)
    prev_input = prev_input.astype(dtype)
    
    grad_output_shape = grad_output.shape
    prev_input_shape = prev_input.shape
    
    """if kernel_shape is None:
        kernel_h = int(prev_input_shape[1] + 2 * pad[0] - (grad_output_shape[1]-1)*stride[0])
        kernel_w = int(prev_input_shape[2] + 2 * pad[1] - (grad_output_shape[2]-1)*stride[1])
        kernel_shape = (kernel_h, kernel_w, prev_input_shape[-1], grad_output_shape[-1])"""
        
    
    dilated_dout = dilate_array(arr=grad_output, stride=stride)
    dilated_dout_shape = dilated_dout.shape
    
    prev_input_strides = prev_input.strides
    
    prev_input_strided = as_strided(
        x=prev_input, 
        shape=(kernel_shape[0], kernel_shape[1], kernel_shape[2], 
               dilated_dout_shape[0], dilated_dout_shape[1], dilated_dout_shape[2]), 
        strides=prev_input.strides[1:] + prev_input_strides[:3]
    )
    return np.tensordot(prev_input_strided, dilated_dout, axes=3)