import numpy as np
from core.utils.conv_utils import faster_convolution, faster_full_convolution, faster_backprop_dw


def slow_dw_backprop(dout, prev_input, kernel_size, stride=(1,1), dtype=np.float32):
    _, h_grad, w_grad, _ = dout.shape
    prev_input_shape = prev_input.shape

    # Initialize gradient with the correct shapes
    dw = np.zeros(kernel_size, dtype=dtype)

    for i in range(h_grad):
        for j in range(w_grad):
            h_start = i*stride[0]
            h_end = h_start + kernel_size[0]
            if h_end > prev_input_shape[1]:
                break

            w_start = j*stride[1]
            w_end = w_start + kernel_size[1]
            if w_end > prev_input_shape[2]:
                break

            # apply gradient
            dw += np.sum(
                dout[:, i, j, :][:, None, None, None, :]*prev_input[:, h_start:h_end, w_start:w_end, :][:, :, :, :, None], 
                axis=0)
    return dw

def slow_dprev_input_backprop(dout, w, prev_input_shape=None, stride=(1,1), pad=(0,0), dtype=np.float32):
    n, h_out, w_out, _ = dout.shape
    w_shape = w.shape
    if prev_input_shape is None:
        h_prev = int((h_out-1)*stride[0] - 2 * pad[0] + w_shape[0])
        w_prev = int((w_out-1)*stride[1] - 2 * pad[1] + w_shape[1])
        prev_input_shape = (n, h_prev, w_prev, w_shape[2])

    # Initialize gradient with the correct shapes
    dprev_input = np.zeros(prev_input_shape, dtype=dtype)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i*stride[0]
            h_end = h_start + w_shape[0]
            if h_end > prev_input_shape[1]:
                break

            w_start = j*stride[1]
            w_end = w_start + w_shape[1]
            if w_end > prev_input_shape[2]:
                break

            # apply gradient
            dprev_input[:, h_start:h_end, w_start:w_end, :] += np.tensordot(dout[:, i, j, :], w, axes=[-1, -1])
    return dprev_input

def slow_convolution(x, w, output_shape=None, pad=(0, 0), stride = (1, 1), dtype=np.float32):
    # package with c++ implementation -> python
    kernel_size = w.shape
    n, h_prev, w_prev, c_prev = x.shape
    
    w_shape = w.shape
    h_out = int((h_prev + 2 * pad[0] - w_shape[0])/ stride[0]) + 1
    w_out = int((w_prev + 2 * pad[1] - w_shape[1])/ stride[1]) + 1
        
    output = np.zeros((x.shape[0], h_out, w_out, kernel_size[-1]), dtype=dtype)
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i*stride[0]
            h_end = h_start+kernel_size[0]
            if h_end > h_prev:
                break

            w_start = j*stride[0]
            w_end = w_start+kernel_size[0]
            if w_end > w_prev:
                break
            #output[:, i, j, :] = np.tensordot(x[:, h_start:h_end, w_start:w_end, :], w, axes=[(1, 2, 3), (0, 1, 2)])
            #output[:, i, j, :] = np.einsum("nhwc, hwco -> no", x[:, h_start:h_end, w_start:w_end, :], w)
            for idx in range(x.shape[0]):
                for c_o in range(w.shape[3]):
                    output[idx, i, j, c_o] = np.sum(w[:, :, :, c_o]*x[idx, h_start:h_end, w_start:w_end, :], axis=(0,1,2))
    return output


def test_faster_convolution_w_4x4x5x6_s_1():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (4, 4, 5, 6))
    test_x = np.random.uniform(-1, 1, (3, 256, 256, w.shape[2]))

    out_slow = slow_convolution(test_x, w)

    out_bit_pack = faster_convolution(test_x, w, dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype
    
def test_faster_convolution_w_3x3x512x512_s_1():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (3, 3, 512, 512))
    test_x = np.random.uniform(-1, 1, (3, 32, 32, w.shape[2]))

    out_slow = slow_convolution(test_x, w, dtype=dtype)
    out_bit_pack = faster_convolution(test_x, w, dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype
    
    
def test_faster_convolution_w_4x4x5x6_s_2():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (4, 4, 5, 6))
    test_x = np.random.uniform(-1, 1, (3, 256, 256, w.shape[2]))

    out_slow = slow_convolution(test_x, w, stride=(2, 2), dtype=dtype)
    out_bit_pack = faster_convolution(test_x, w, stride=(2, 2), dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype

    
def test_faster_convolution_w_3x3x2x4_s_1():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (3, 3, 2, 4))
    test_x = np.random.uniform(-1, 1, (3, 256, 256, w.shape[2]))

    out_slow = slow_convolution(test_x, w, dtype=dtype)

    out_bit_pack = faster_convolution(test_x, w, dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype
    
    
def test_faster_convolution_w_3x3x2x4_s_2():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (3, 3, 2, 4))
    test_x = np.random.uniform(-1, 1, (3, 256, 256, w.shape[2]))

    out_slow = slow_convolution(test_x, w, stride=(2, 2), dtype=dtype)
    out_bit_pack = faster_convolution(test_x, w, stride=(2, 2), dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype
    
    
def test_faster_backprop_dw_w_3x3x2x4_s_1():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (3, 3, 2, 4))
    
    test_x = np.random.uniform(-1, 1, (2, 256, 256, w.shape[3]))
    test_x_prev_input = np.random.uniform(-1, 1, (2, 258, 258, w.shape[2]))

    out_slow = slow_dw_backprop(test_x, test_x_prev_input, w.shape, stride=(1, 1), dtype=dtype)
    out_bit_pack = faster_backprop_dw(test_x, test_x_prev_input, w.shape, stride=(1, 1), dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype
    
    
"""def test_faster_backprop_dw_w_3x3x2x4_s_1_mod():
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (3, 3, 2, 4))
    
    test_x = np.random.uniform(-1, 1, (2, 256, 256, w.shape[3]))
    test_x_prev_input = np.random.uniform(-1, 1, (2, 258, 258, w.shape[2]))

    out_slow = slow_dw_backprop(test_x, test_x_prev_input, w.shape, stride=(1, 1))
    out_bit_pack = faster_backprop_dw(test_x, test_x_prev_input, stride=(1, 1))
    
    assert np.allclose(out_slow, out_bit_pack)"""
    
    
def test_faster_backprop_dw_w_3x3x2x4_s_2():
    dtype = np.float64
    
    np.random.seed(0)
    w = np.random.uniform(-1, 1, (3, 3, 2, 4))
    
    test_x = np.random.uniform(-1, 1, (2, 128, 128, w.shape[3]))
    test_x_prev_input = np.random.uniform(-1, 1, (2, 258, 258, w.shape[2]))

    out_slow = slow_dw_backprop(test_x, test_x_prev_input, w.shape, stride=(2, 2), dtype=dtype)
    out_bit_pack = faster_backprop_dw(test_x, test_x_prev_input, w.shape, stride=(2, 2), dtype=dtype)
    
    assert np.allclose(out_slow, out_bit_pack)
    assert out_bit_pack.dtype == dtype