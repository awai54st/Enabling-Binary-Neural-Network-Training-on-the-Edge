import numpy as np
import tensorflow as tf
from core.numpy_models.full_precision_models.pooling_layers import MaxPooling, AveragePooling 
#from core.numpy_models.full_precision_models.pooling_layers import MaxPoolingSlow as MaxPooling

def test_MaxPooling_2_2_large_int_input():
    np.random.seed(10)
    dtype = np.float32
    test_input = np.random.randint(-2000, 20000, (100, 32, 32, 512))

    # numpy implementation
    test_MaxPooling = MaxPooling((2, 2), (2,2), dtype=dtype)
    res_forward_np = test_MaxPooling.forward(test_input)

    res_grads_np = test_MaxPooling.backprop(dout=np.ones_like(res_forward_np))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    with tf.GradientTape() as tape:
        y = tf.nn.max_pool(z, 2, 2, "VALID")
    res_grads_tf = tape.gradient(y, z)

    assert res_grads_tf.dtype == dtype
    assert res_forward_np.dtype == dtype
    #assert test_MaxPooling.prev_input_col.dtype == dtype

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()


def test_MaxPooling_2_2_small_int_input():
    np.random.seed(10)
    dtype = np.float32
    test_input = np.random.randint(-300, 300, (100, 32, 32, 512))

    # numpy implementation
    test_MaxPooling = MaxPooling((2, 2), (2,2), dtype=dtype)
    res_forward_np = test_MaxPooling.forward(test_input)

    res_grads_np = test_MaxPooling.backprop(dout=np.ones_like(res_forward_np))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    with tf.GradientTape() as tape:
        y = tf.nn.max_pool(z, 2, 2, "VALID")
    res_grads_tf = tape.gradient(y, z)

    assert res_grads_tf.dtype == dtype
    assert res_forward_np.dtype == dtype
    #assert test_MaxPooling.prev_input_col.dtype == dtype

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()
    
    
def test_MaxPooling_2_2():
    np.random.seed(0)
    dtype = np.float32
    test_input = np.random.uniform(-1, 1, (5, 10, 10, 4))
    
    # numpy implementation
    test_MaxPooling = MaxPooling((2, 2), (2,2), dtype=dtype)
    res_forward_np = test_MaxPooling.forward(test_input)
    
    res_grads_np = test_MaxPooling.backprop(dout=np.ones_like(res_forward_np))
    
    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    with tf.GradientTape() as tape:
        y = tf.nn.max_pool(z, 2, 2, "VALID")
    res_grads_tf = tape.gradient(y, z)
    
    assert res_grads_tf.dtype == dtype
    assert res_forward_np.dtype == dtype
    #assert test_MaxPooling.prev_input_col.dtype == dtype
    
    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_grads_tf.numpy()).all()
    
    

def test_AveragePooling_2_2():
    np.random.seed(0)
    test_input = np.random.uniform(-1, 1, (5, 10, 10, 4))
    
    # numpy implementation
    test_AveragePooling = AveragePooling((2, 2), (2, 2))
    res_forward_np = test_AveragePooling.forward(test_input)
    res_grads_np = test_AveragePooling.backprop(output_grad=np.ones((5, 5, 5, 4)))
    
    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    with tf.GradientTape() as tape:
        y = tf.nn.avg_pool(z, 2, 2, "VALID")
    res_grads_tf = tape.gradient(y, z)
    
    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-4)
    assert np.allclose(res_grads_np, res_grads_tf.numpy())
    