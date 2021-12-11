import numpy as np
import tensorflow as tf
from core.numpy_models.full_precision_models.layers import Dense, Conv2D, Flatten
from core.test.test_helper.full_precision_tf_model import Linear
from core.test.test_helper.assert_helper import assert_by_percentage


def test_Flatten():
    np.random.seed(0)
    test_input = np.random.uniform(-10, 10, size=(20, 128, 128, 64))
    test_flatten = Flatten()
    res_forward = test_flatten.forward(test_input)
    res_backward = test_flatten.backprop(np.ones_like(res_forward))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    flatten_tf = tf.keras.layers.Flatten()
    with tf.GradientTape() as tape:
        y = flatten_tf(z)
    res_input_grads_tf = tape.gradient(y, z)

    assert res_forward.shape == (20, 128*128*64)
    assert res_backward.shape == test_input.shape
    assert (res_forward == test_input.reshape(20, -1).astype(np.float32)).all()

    assert (res_input_grads_tf.numpy() == res_backward).all()
    assert (y.numpy() == res_forward).all()
    

def test_Flatten_large_input_int():
    np.random.seed(0)
    test_input = np.random.randint(-2000, 2000, size=(100, 16, 16, 512))
    test_flatten = Flatten()
    res_forward = test_flatten.forward(test_input)
    res_backward = test_flatten.backprop(np.ones_like(res_forward))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    flatten_tf = tf.keras.layers.Flatten()
    with tf.GradientTape() as tape:
        y = flatten_tf(z)
    res_input_grads_tf = tape.gradient(y, z)

    assert res_forward.shape == (100, 16*16*512)
    assert res_backward.shape == test_input.shape
    assert (res_forward == test_input.reshape(100, -1).astype(np.float32)).all()

    assert (res_input_grads_tf.numpy() == res_backward).all()
    assert (y.numpy() == res_forward).all()
    
    
def test_Dense_4():
    np.random.seed(0)
    dtype=np.float32

    test_input = np.random.uniform(-1, 1, (5, 10)).astype(dtype)
    nodes = 4

    # numpy implementation
    test_Dense = Dense(nodes)
    res_forward_np = test_Dense.forward(test_input)
    res_grads_np = test_Dense.backprop(np.ones((5, 4), dtype=dtype))

    assert test_Dense.w.dtype == np.float32
    assert test_Dense.b.dtype == np.float32
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_Dense._dw.dtype == np.float32
    assert test_Dense._db.dtype == np.float32

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_Dense.w, dtype = tf.float32)
    b_tf = tf.Variable(test_Dense.b, dtype = tf.float32)
    linear_tf = Linear(w_tf, b_tf)

    with tf.GradientTape() as tape:
        y = linear_tf(z)
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_input_grads_tf.numpy()).all()
    assert (test_Dense.gradients[0] == res_w_grads_tf.numpy()).all()
    assert (test_Dense.gradients[1] == res_b_grads_tf.numpy()).all()
    
def test_Dense_16():
    np.random.seed(4)
    dtype=np.float32
    
    test_input = np.random.uniform(-3, 3, (5, 10)).astype(dtype)
    nodes = 16

    # numpy implementation
    test_Dense = Dense(nodes)
    res_forward_np = test_Dense.forward(test_input)
    res_grads_np = test_Dense.backprop(np.ones_like(res_forward_np))

    assert test_Dense.w.dtype == np.float32
    assert test_Dense.b.dtype == np.float32
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_Dense._dw.dtype == np.float32
    assert test_Dense._db.dtype == np.float32

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_Dense.w, dtype = tf.float32)
    b_tf = tf.Variable(test_Dense.b, dtype = tf.float32)
    linear_tf = Linear(w_tf, b_tf)

    with tf.GradientTape() as tape:
        y = linear_tf(z)
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_input_grads_tf.numpy()).all()
    assert (test_Dense.gradients[0] == res_w_grads_tf.numpy()).all()
    assert (test_Dense.gradients[1] == res_b_grads_tf.numpy()).all()
        
    
def test_Dense_64():
    np.random.seed(10)
    dtype=np.float32
    
    test_input = np.random.uniform(-3, 3, (5, 10)).astype(dtype)
    nodes = 64

    # numpy implementation
    test_Dense = Dense(nodes)
    res_forward_np = test_Dense.forward(test_input)
    res_grads_np = test_Dense.backprop(np.ones_like(res_forward_np))

    assert test_Dense.w.dtype == np.float32
    assert test_Dense.b.dtype == np.float32
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_Dense._dw.dtype == np.float32
    assert test_Dense._db.dtype == np.float32

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_Dense.w, dtype = tf.float32)
    b_tf = tf.Variable(test_Dense.b, dtype = tf.float32)
    linear_tf = Linear(w_tf, b_tf)

    with tf.GradientTape() as tape:
        y = linear_tf(z)
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np == res_input_grads_tf.numpy()).all()
    assert (test_Dense.gradients[0] == res_w_grads_tf.numpy()).all()
    assert (test_Dense.gradients[1] == res_b_grads_tf.numpy()).all()
    
def test_Conv2D_5_3x3():
    np.random.seed(0)
    dtype=np.float32
    
    test_input = np.random.uniform(-3, 3, (5, 10, 10, 2)).astype(dtype)

    # numpy implementation
    test_Conv2D = Conv2D(5, (3, 3), padding="valid", dtype=dtype)
    res_forward_np = test_Conv2D.forward(test_input)
    res_grads_np = test_Conv2D.backprop(np.ones_like(res_forward_np) )

    assert test_Conv2D.w.dtype == np.float32
    assert test_Conv2D.b.dtype == np.float32
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_Conv2D._dw.dtype == np.float32
    assert test_Conv2D._db.dtype == np.float32

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_Conv2D.w, dtype = tf.float32)
    b_tf = tf.Variable(test_Conv2D.b, dtype = tf.float32)

    with tf.GradientTape() as tape:
        y = tf.nn.conv2d(z, w_tf, [1, 1, 1, 1], "VALID")+b_tf
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])


    assert (res_forward_np == y.numpy()).all()
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy(), rtol=1e-5)
    assert (test_Conv2D.gradients[0] == res_w_grads_tf.numpy()).all()
    assert (test_Conv2D.gradients[1] == res_b_grads_tf.numpy()).all()
    
    
    
def test_Conv2D_8_4x4():
    np.random.seed(10)
    dtype=np.float32
    
    test_input = np.random.uniform(-3, 3, (5, 10, 10, 2)).astype(dtype)

    # numpy implementation
    test_Conv2D = Conv2D(5, (4, 4), padding="valid", dtype=dtype)
    res_forward_np = test_Conv2D.forward(test_input)
    res_grads_np = test_Conv2D.backprop(np.ones_like(res_forward_np) )

    assert test_Conv2D.w.dtype == np.float32
    assert test_Conv2D.b.dtype == np.float32
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_Conv2D._dw.dtype == np.float32
    assert test_Conv2D._db.dtype == np.float32

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_Conv2D.w, dtype = tf.float32)
    b_tf = tf.Variable(test_Conv2D.b, dtype = tf.float32)

    with tf.GradientTape() as tape:
        y = tf.nn.conv2d(z, w_tf, [1, 1, 1, 1], "VALID")+b_tf
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])


    assert (res_forward_np == y.numpy()).all()
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy(), rtol=1e-5)
    assert (test_Conv2D.gradients[0] == res_w_grads_tf.numpy()).all()
    assert (test_Conv2D.gradients[1] == res_b_grads_tf.numpy()).all()
        
def test_Conv2D_5_3x3_large():
    np.random.seed(10)
    dtype=np.float32
    
    test_input = np.random.uniform(-3, 3, (10, 32, 32, 128)).astype(dtype)

    # numpy implementation
    test_Conv2D = Conv2D(5, (3, 3), padding="valid", dtype=dtype)
    res_forward_np = test_Conv2D.forward(test_input)
    res_grads_np = test_Conv2D.backprop(np.ones_like(res_forward_np) )

    assert test_Conv2D.w.dtype == np.float32
    assert test_Conv2D.b.dtype == np.float32
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_Conv2D._dw.dtype == np.float32
    assert test_Conv2D._db.dtype == np.float32

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_Conv2D.w, dtype = tf.float32)
    b_tf = tf.Variable(test_Conv2D.b, dtype = tf.float32)

    with tf.GradientTape() as tape:
        y = tf.nn.conv2d(z, w_tf, [1, 1, 1, 1], "VALID")+b_tf
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-2)
    assert assert_by_percentage(res_forward_np, y.numpy())
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy(), rtol=1e-4)
    assert np.allclose(test_Conv2D.gradients[0], res_w_grads_tf.numpy(), rtol=1e-3)
    assert np.allclose(test_Conv2D.gradients[1], res_b_grads_tf.numpy(), rtol=1e-4)