import numpy as np
import tensorflow as tf
from core.test.test_helper.binary_tf_model import BinaryActivationTf
from core.numpy_models.binary_models.activation_layers import BinaryActivation


def test_BinaryActivation_large_conv_int():
    dtype = np.float32

    test_input = (np.random.randint(-2000, 2000, (4, 128, 128, 512)))

    # numpy implementation
    test_BinaryActivation = BinaryActivation(dtype=dtype)
    res_forward_np = test_BinaryActivation.forward(test_input)
    res_grads_np = test_BinaryActivation.backprop(output_grad=np.ones_like(res_forward_np, dtype=dtype))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    activation_tf = BinaryActivationTf()
    with tf.GradientTape() as tape:
        y = activation_tf(z)
    res_grads_tf = tape.gradient(y, z)

    # (res_forward_np == 0).any() # there are 0s
    assert res_forward_np.dtype == dtype
    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()
    assert (res_grads_np>0).any()
    
    
def test_BinaryActivation_large_conv():
    dtype = np.float32
    
    test_input = (np.random.uniform(-10, 10, (4, 128, 128, 512)))
    
    # numpy implementation
    test_BinaryActivation = BinaryActivation(dtype=dtype)
    res_forward_np = test_BinaryActivation.forward(test_input)
    res_grads_np = test_BinaryActivation.backprop(output_grad=np.ones_like(res_forward_np, dtype=dtype))
    
    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    activation_tf = BinaryActivationTf()
    with tf.GradientTape() as tape:
        y = activation_tf(z)
    res_grads_tf = tape.gradient(y, z)
    
    assert res_forward_np.dtype == dtype
    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()
    

def test_BinaryActivation_small_conv():
    dtype = np.float32
    
    test_input = np.random.uniform(-5, 5, (4, 128, 128, 64))
    
    # numpy implementation
    test_BinaryActivation = BinaryActivation(dtype=dtype)
    res_forward_np = test_BinaryActivation.forward(test_input)
    res_grads_np = test_BinaryActivation.backprop(output_grad=np.ones_like(res_forward_np, dtype=dtype))
    
    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    activation_tf = BinaryActivationTf()
    with tf.GradientTape() as tape:
        y = activation_tf(z)
    res_grads_tf = tape.gradient(y, z)
    
    assert res_forward_np.dtype == dtype
    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()

def test_BinaryActivation_large_dense():
    dtype = np.float32
    
    test_input = (np.random.uniform(-10, 10, (100, 1024)))
    
    # numpy implementation
    test_BinaryActivation = BinaryActivation(dtype=dtype)
    res_forward_np = test_BinaryActivation.forward(test_input)
    res_grads_np = test_BinaryActivation.backprop(output_grad=np.ones_like(res_forward_np, dtype=dtype))
    
    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    activation_tf = BinaryActivationTf()
    with tf.GradientTape() as tape:
        y = activation_tf(z)
    res_grads_tf = tape.gradient(y, z)
    
    assert res_forward_np.dtype == dtype
    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()