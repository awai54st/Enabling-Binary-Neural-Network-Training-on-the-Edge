import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from core.numpy_models.binary_models.layers import BinaryConv2D, BinaryDense, BatchNorm
from core.test.test_helper.binary_tf_model import Conv2DTf, DenseTf
from core.keras_models.keras_binary_models.binary_tf_model import l1_batch_norm_mod_conv, l1_batch_norm_mod_dense


def test_BinaryConv2D_w_smaller_than_1():
    dtype = np.float32
    
    test_input = np.random.uniform(-1, 1, (5, 10, 10, 2)).astype(dtype)

    # numpy implementation
    test_BinaryConv2D = BinaryConv2D(8, (4, 4), padding="valid", dtype=dtype, use_bias=True)
    res_forward_np = test_BinaryConv2D.forward(test_input)
    res_grads_np = test_BinaryConv2D.backprop(np.ones_like(res_forward_np) )

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_BinaryConv2D.w, dtype = tf.float32)
    b_tf = tf.Variable(test_BinaryConv2D.b, dtype = tf.float32)
    tf_conv = Conv2DTf(w_tf, b_tf)
    
    with tf.GradientTape() as tape:
        y = tf_conv(z)
        
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-3)
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy())
    assert np.allclose(test_BinaryConv2D.gradients[0], res_w_grads_tf)
    assert np.allclose(test_BinaryConv2D.gradients[1], res_b_grads_tf)
    
    
def test_BinaryConv2D_w_larger_than_1():
    dtype = np.float32
    
    test_input = np.random.uniform(-1, 1, (5, 10, 10, 2)).astype(dtype)

    # numpy implementation
    test_BinaryConv2D = BinaryConv2D(8, (4, 4), padding="valid", dtype=dtype, use_bias=True)
    test_BinaryConv2D.forward(test_input)  # init weights
    test_BinaryConv2D.w = test_BinaryConv2D.w*2
    res_forward_np = test_BinaryConv2D.forward(test_input)
    res_grads_np = test_BinaryConv2D.backprop(np.ones_like(res_forward_np) )

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_BinaryConv2D.w, dtype = tf.float32)
    b_tf = tf.Variable(test_BinaryConv2D.b, dtype = tf.float32)
    tf_conv = Conv2DTf(w_tf, b_tf)
    
    with tf.GradientTape() as tape:
        y = tf_conv(z)
        
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-3)
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy())
    assert np.allclose(test_BinaryConv2D.gradients[0], res_w_grads_tf)
    assert np.allclose(test_BinaryConv2D.gradients[1], res_b_grads_tf)
    
    
def test_BinaryDense_w_smaller_than_1():
    dtype = np.float32
    
    test_input = np.random.uniform(-1, 1, (5, 10)).astype(dtype)
    nodes = 4
    
    # numpy implementation
    test_BinaryDense = BinaryDense(nodes, dtype=dtype, use_bias=True)
    res_forward_np = test_BinaryDense.forward(test_input)
    res_grads_np = test_BinaryDense.backprop(np.ones((5, 4)))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_BinaryDense.w, dtype = tf.float32)
    b_tf = tf.Variable(test_BinaryDense.b, dtype = tf.float32)
    tf_DenseTf = DenseTf(w_tf, b_tf)

    with tf.GradientTape() as tape:
        y = tf_DenseTf(z)
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert np.allclose(res_forward_np, y.numpy())
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy())
    assert np.allclose(test_BinaryDense.gradients[0], res_w_grads_tf)
    assert np.allclose(test_BinaryDense.gradients[1], res_b_grads_tf)
    
    
def test_BinaryDense_w_larger_than_1():
    dtype = np.float32
    
    test_input = np.random.uniform(-1, 1, (5, 10))
    nodes = 4
    
    # numpy implementation
    test_BinaryDense = BinaryDense(nodes, dtype=dtype, use_bias=True)
    test_BinaryDense.forward(test_input)  # init weights
    test_BinaryDense.w = test_BinaryDense.w*2
    res_forward_np = test_BinaryDense.forward(test_input)
    res_grads_np = test_BinaryDense.backprop(np.ones((5, 4)))

    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(test_BinaryDense.w, dtype = tf.float32)
    b_tf = tf.Variable(test_BinaryDense.b, dtype = tf.float32)
    tf_DenseTf = DenseTf(w_tf, b_tf)

    with tf.GradientTape() as tape:
        y = tf_DenseTf(z)
    res_input_grads_tf, res_w_grads_tf, res_b_grads_tf = tape.gradient(y, [z, w_tf, b_tf])

    assert np.allclose(res_forward_np, y.numpy())
    assert np.allclose(res_grads_np, res_input_grads_tf.numpy())
    assert np.allclose(test_BinaryDense.gradients[0], res_w_grads_tf)
    assert np.allclose(test_BinaryDense.gradients[1], res_b_grads_tf)
    


def test_BatchNorm_conv():
    np.random.seed(0)
    test_input = np.random.uniform(-100, 100, (2, 5, 5, 2))
    test_shape = test_input.shape

    test_BatchNorm = BatchNorm(0.9)
    res_forward_np = test_BatchNorm.forward(test_input)
    res_grads_np = test_BatchNorm.backprop(np.ones_like(test_input))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_batch = l1_batch_norm_mod_conv(test_shape[0], test_shape[1], test_shape[-1], 0.9)

    with tf.GradientTape() as tape:
        y = tf_batch(z)
    res_grads_tf, res_beta_grads_tf = tape.gradient(y, [z, tf_batch.beta])

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-5) # checked visually, equal when 2f significant figure
    assert np.allclose(res_grads_tf, res_grads_tf)
    assert np.allclose(test_BatchNorm.mu, tf_batch.mu.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.var, tf_batch.var.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.dbeta, res_beta_grads_tf.numpy())

    
def test_BatchNorm_dense():
    np.random.seed(0)
    test_input = np.random.uniform(-100, 100, (5, 9)).astype(np.float32)
    test_shape = test_input.shape

    test_BatchNorm = BatchNorm(0.9)
    res_forward_np = test_BatchNorm.forward(test_input)
    res_grads_np = test_BatchNorm.backprop(np.ones_like(test_input))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_batch = l1_batch_norm_mod_dense(test_shape[0], test_shape[1], 0.9)

    with tf.GradientTape() as tape:
        y = tf_batch(z)
    res_grads_tf, res_beta_grads_tf = tape.gradient(y, [z, tf_batch.beta])

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-5) # checked visually, equal when 2f significant figure
    assert np.allclose(res_grads_np, res_grads_tf.numpy(), rtol=1e-3) # checked visually, equal when 3f significant figure
    assert np.allclose(test_BatchNorm.mu, tf_batch.mu.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.var, tf_batch.var.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy(), rtol=1e-4)
    assert np.allclose(test_BatchNorm.dbeta, res_beta_grads_tf.numpy())