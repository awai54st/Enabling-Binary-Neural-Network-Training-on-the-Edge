import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from core.test.test_helper.assert_helper import assert_by_percentage

from core.numpy_models.xnor_models.bit_packed_layers import XNorConv2D, XNorDense, BatchNorm

from core.keras_models.keras_xnor_models.binarization_utils import (
    l1_batch_norm_mod_conv, l1_batch_norm_mod_dense, binary_conv, binary_dense)


def test_XNorConv2D():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 5, 5, 3))
    test_shape = test_input.shape

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "valid")

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="valid", dtype=dtype)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None


    assert (res_forward_np==y.numpy()).all()
    #assert (res_grads_np==res_grads_tf.numpy()).all()
    #assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()


def test_XNorConv2D_large_weight():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 5, 5, 3))
    test_shape = test_input.shape
    random_w = np.random.uniform(-5, 5, (3, 3, test_shape[-1], 4))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    w_tf = tf.Variable(random_w, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "valid")
    tf_binary_conv.w = w_tf
    tf_binary_conv.built = True

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="valid", dtype=dtype)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None


    assert (res_forward_np==y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()
    
    
def test_XNorConv2D_same():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 5, 5, 3))
    test_shape = test_input.shape

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "same")

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="same", dtype=dtype)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None


    assert (res_forward_np==y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()
    

def test_XNorConv2D_same_first_layer():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 5, 5, 3))
    test_shape = test_input.shape

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "same", first_layer=True)

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="same", dtype=dtype, first_layer=True)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None


    assert (res_forward_np==y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()
    
    
def test_XNorConv2D_large():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 256, 256, 16))
    test_shape = test_input.shape

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "same", first_layer=True)

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="same", dtype=dtype, first_layer=True)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None


    assert (res_forward_np==y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()
    
def test_XNorConv2D_256():
    # can use as_strided
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 32, 32, 256))
    test_shape = test_input.shape

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "same", first_layer=True)

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="same", dtype=dtype, first_layer=True)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None
    
    assert np.allclose(res_forward_np, y.numpy(), rtol=3e-3)
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()
    
    
def test_XNorConv2D_512():
    np.random.seed(20)
    dtype = np.float16

    test_input = np.random.randint(-2000, 2000, (100, 32, 32, 512))
    test_shape = test_input.shape

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_conv = binary_conv(4, test_shape[-1], 3, "valid")

    with tf.GradientTape() as tape:
        y = tf_binary_conv(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_conv.w])


    test_XNorConv2D = XNorConv2D(4, (3,3), padding="valid", dtype=dtype)
    test_XNorConv2D.w = tf_binary_conv.w.numpy().astype(dtype)
    test_XNorConv2D.b = np.zeros((4), dtype=dtype)
    test_XNorConv2D.is_built = True
    res_forward_np = test_XNorConv2D.forward(test_input)
    res_grads_np = test_XNorConv2D.backprop(np.ones_like(res_forward_np))

    assert test_XNorConv2D.w.dtype == dtype
    assert test_XNorConv2D.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    #assert test_XNorConv2D._dw.dtype == dtype
    assert test_XNorConv2D._db is None


    assert (res_forward_np==y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorConv2D._dw==res_w_grads_tf.numpy()).all()

    
def test_XNorDense():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-1, 1, (2, 5))
    test_shape = test_input.shape
    nodes = 3

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_dense = binary_dense(test_shape[-1], nodes)

    with tf.GradientTape() as tape:
        y = tf_binary_dense(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_dense.w])


    test_XNorDense = XNorDense(nodes, dtype=dtype)
    test_XNorDense.w = tf_binary_dense.w.numpy().astype(dtype)
    test_XNorDense.b = np.zeros((2,2), dtype=dtype)
    test_XNorDense.is_built = True
    res_forward_np = test_XNorDense.forward(test_input)
    res_grads_np = test_XNorDense.backprop(np.ones_like(res_forward_np))

    assert test_XNorDense.w.dtype == dtype
    assert test_XNorDense.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_XNorDense._dw.dtype == dtype
    assert test_XNorDense._db is None

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorDense._dw == res_w_grads_tf.numpy()).all()

    
def test_XNorDense_large():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-5, 5, (100, 256))
    test_shape = test_input.shape
    nodes = 3

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_dense = binary_dense(test_shape[-1], nodes)

    with tf.GradientTape() as tape:
        y = tf_binary_dense(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_dense.w])


    test_XNorDense = XNorDense(nodes, dtype=dtype)
    test_XNorDense.w = tf_binary_dense.w.numpy().astype(dtype)
    test_XNorDense.b = np.zeros((2,2), dtype=dtype)
    test_XNorDense.is_built = True
    res_forward_np = test_XNorDense.forward(test_input)
    res_grads_np = test_XNorDense.backprop(np.ones_like(res_forward_np))

    assert test_XNorDense.w.dtype == dtype
    assert test_XNorDense.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_XNorDense._dw.dtype == dtype
    assert test_XNorDense._db is None

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorDense._dw == res_w_grads_tf.numpy()).all()

    
def test_XNorDense_1024():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.uniform(-5, 5, (100, 1024))
    test_shape = test_input.shape
    nodes = 3

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_dense = binary_dense(test_shape[-1], nodes)

    with tf.GradientTape() as tape:
        y = tf_binary_dense(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_dense.w])


    test_XNorDense = XNorDense(nodes, dtype=dtype)
    test_XNorDense.w = tf_binary_dense.w.numpy().astype(dtype)
    test_XNorDense.b = np.zeros((2,2), dtype=dtype)
    test_XNorDense.is_built = True
    res_forward_np = test_XNorDense.forward(test_input)
    res_grads_np = test_XNorDense.backprop(np.ones_like(res_forward_np))

    assert test_XNorDense.w.dtype == dtype
    assert test_XNorDense.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_XNorDense._dw.dtype == dtype
    assert test_XNorDense._db is None

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-4)
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorDense._dw == res_w_grads_tf.numpy()).all()
    
def test_XNorDense_1024_large_int():
    np.random.seed(0)
    dtype = np.float16
    
    test_input = np.random.randint(-2000, 2000, (100, 1024))
    test_shape = test_input.shape
    nodes = 1024

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_binary_dense = binary_dense(test_shape[-1], nodes)

    with tf.GradientTape() as tape:
        y = tf_binary_dense(z)
    res_grads_tf, res_w_grads_tf = tape.gradient(y, [z, tf_binary_dense.w])
    
    test_XNorDense = XNorDense(nodes, dtype=dtype)
    test_XNorDense.w = tf_binary_dense.w.numpy().astype(dtype)
    test_XNorDense.b = np.zeros((2,2), dtype=dtype)
    test_XNorDense.is_built = True
    res_forward_np = test_XNorDense.forward(test_input)
    res_grads_np = test_XNorDense.backprop(np.ones_like(res_forward_np))

    assert test_XNorDense.w.dtype == dtype
    assert test_XNorDense.b.dtype == dtype
    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_XNorDense._dw.dtype == dtype
    assert test_XNorDense._db is None

    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-4)
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_XNorDense._dw == res_w_grads_tf.numpy()).all()
    

def test_BatchNorm_conv():
    np.random.seed(0)
    dtype = np.float32
    test_input = np.random.uniform(-1, 1, (2, 5, 5, 3))
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
    res_grads_tf, dbeta_tf = tape.gradient(y, [z, tf_batch.beta])

    assert res_forward_np.dtype == dtype
    assert res_grads_np.dtype == dtype
    assert test_BatchNorm.mu.dtype == dtype
    assert test_BatchNorm.var.dtype == dtype
    assert test_BatchNorm.moving_mean.dtype == dtype
    assert test_BatchNorm.moving_var.dtype == dtype
    assert test_BatchNorm.beta.dtype == dtype
    assert test_BatchNorm.dbeta.dtype == dtype

    assert np.allclose(res_forward_np, y.numpy())
    assert np.allclose(res_grads_np, res_grads_tf.numpy())
    assert (test_BatchNorm.mu == tf_batch.mu.numpy()).all()
    assert (test_BatchNorm.var == tf_batch.var.numpy()).all()
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy())
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy())
    assert (test_BatchNorm.dbeta == dbeta_tf.numpy()).all()
    
def test_BatchNorm_conv_large():
    np.random.seed(0)
    dtype = np.float32
    test_input = np.random.uniform(-1, 1, (1000, 8, 8, 512))
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
    res_grads_tf, dbeta_tf = tape.gradient(y, [z, tf_batch.beta])

    assert res_forward_np.dtype == dtype
    assert res_grads_np.dtype == dtype
    assert test_BatchNorm.mu.dtype == dtype
    assert test_BatchNorm.var.dtype == dtype
    assert test_BatchNorm.moving_mean.dtype == dtype
    assert test_BatchNorm.moving_var.dtype == dtype
    assert test_BatchNorm.beta.dtype == dtype
    assert test_BatchNorm.dbeta.dtype == dtype

    assert np.allclose(res_forward_np, y.numpy())
    assert np.allclose(res_grads_np, res_grads_tf.numpy())
    assert (test_BatchNorm.mu == tf_batch.mu.numpy()).all()
    assert (test_BatchNorm.var == tf_batch.var.numpy()).all()
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy())
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy())
    assert (test_BatchNorm.dbeta == dbeta_tf.numpy()).all()
    


    
def test_BatchNorm_dense():
    np.random.seed(0)
    dtype = np.float32
    test_input = np.random.uniform(-1, 1, (2, 8))
    test_shape = test_input.shape

    test_BatchNorm = BatchNorm(0.9)
    res_forward_np = test_BatchNorm.forward(test_input)
    res_grads_np = test_BatchNorm.backprop(np.ones_like(test_input))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_batch = l1_batch_norm_mod_dense(test_shape[0], test_shape[-1], 0.9)

    with tf.GradientTape() as tape:
        y = tf_batch(z)
    res_grads_tf, dbeta_tf = tape.gradient(y, [z, tf_batch.beta])

    assert res_forward_np.dtype == dtype
    assert res_grads_np.dtype == dtype
    assert test_BatchNorm.mu.dtype == dtype
    assert test_BatchNorm.var.dtype == dtype
    assert test_BatchNorm.moving_mean.dtype == dtype
    assert test_BatchNorm.moving_var.dtype == dtype
    assert test_BatchNorm.beta.dtype == dtype
    assert test_BatchNorm.dbeta.dtype == dtype

    assert np.allclose(res_forward_np, y.numpy())
    assert (res_grads_np == res_grads_tf.numpy()).all()
    assert (test_BatchNorm.mu == tf_batch.mu.numpy()).all()
    assert (test_BatchNorm.var == tf_batch.var.numpy()).all()
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy())
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy())
    assert (test_BatchNorm.dbeta == dbeta_tf.numpy()).all()
    
def test_BatchNorm_dense_large():
    # [solved] Error due to how numpy handles precision error. Rearrange the operation to solve it
    np.random.seed(0)
    dtype = np.float32
    test_input = np.random.uniform(-1, 1, (10, 1024))
    test_shape = test_input.shape

    test_BatchNorm = BatchNorm(0.9)
    res_forward_np = test_BatchNorm.forward(test_input)
    res_grads_np = test_BatchNorm.backprop(np.ones_like(test_input))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_batch = l1_batch_norm_mod_dense(test_shape[0], test_shape[-1], 0.9)

    with tf.GradientTape() as tape:
        y = tf_batch(z)
    res_grads_tf, dbeta_tf = tape.gradient(y, [z, tf_batch.beta])

    assert res_forward_np.dtype == dtype
    assert res_grads_np.dtype == dtype
    assert test_BatchNorm.mu.dtype == dtype
    assert test_BatchNorm.var.dtype == dtype
    assert test_BatchNorm.moving_mean.dtype == dtype
    assert test_BatchNorm.moving_var.dtype == dtype
    assert test_BatchNorm.beta.dtype == dtype
    assert test_BatchNorm.dbeta.dtype == dtype

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_BatchNorm.mu == tf_batch.mu.numpy()).all()
    assert (test_BatchNorm.var == tf_batch.var.numpy()).all()
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy())
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy())
    assert (test_BatchNorm.dbeta == dbeta_tf.numpy()).all()
    
    
def test_BatchNorm_dense_large_int():
    # [solved] Error due to how numpy handles precision error. Rearrange the operation to solve it
    np.random.seed(0)
    dtype = np.float32
    test_input = np.random.randint(-2000, 2000, (100, 1024))
    test_shape = test_input.shape

    test_BatchNorm = BatchNorm(0.9)
    res_forward_np = test_BatchNorm.forward(test_input)
    res_grads_np = test_BatchNorm.backprop(np.ones_like(test_input))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_batch = l1_batch_norm_mod_dense(test_shape[0], test_shape[-1], 0.9)

    with tf.GradientTape() as tape:
        y = tf_batch(z)
    res_grads_tf, dbeta_tf = tape.gradient(y, [z, tf_batch.beta])

    assert res_forward_np.dtype == dtype
    assert res_grads_np.dtype == dtype
    assert test_BatchNorm.mu.dtype == dtype
    assert test_BatchNorm.var.dtype == dtype
    assert test_BatchNorm.moving_mean.dtype == dtype
    assert test_BatchNorm.moving_var.dtype == dtype
    assert test_BatchNorm.beta.dtype == dtype
    assert test_BatchNorm.dbeta.dtype == dtype

    assert (res_forward_np == y.numpy()).all()
    assert (res_grads_np==res_grads_tf.numpy()).all()
    assert (test_BatchNorm.mu == tf_batch.mu.numpy()).all()
    assert (test_BatchNorm.var == tf_batch.var.numpy()).all()
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy())
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy())
    assert (test_BatchNorm.dbeta == dbeta_tf.numpy()).all()
    
    
def test_BatchNorm_dense_large_int_f16():
    # [solved] Error due to how numpy handles precision error. Rearrange the operation to solve it
    np.random.seed(0)
    dtype = np.float16
    test_input = np.random.randint(-2000, 2000, (100, 1024))
    test_shape = test_input.shape

    test_BatchNorm = BatchNorm(0.9, dtype=dtype)
    res_forward_np = test_BatchNorm.forward(test_input)
    res_grads_np = test_BatchNorm.backprop(np.ones_like(test_input))

    # tensorflow implementation
    K.set_learning_phase(1)
    z = tf.Variable(test_input, dtype = tf.float32)
    tf_batch = l1_batch_norm_mod_dense(test_shape[0], test_shape[-1], 0.9)

    with tf.GradientTape() as tape:
        y = tf_batch(z)
    res_grads_tf, dbeta_tf = tape.gradient(y, [z, tf_batch.beta])

    assert res_forward_np.dtype == np.float32
    assert res_grads_np.dtype == np.float32
    assert test_BatchNorm.mu.dtype == dtype
    assert test_BatchNorm.var.dtype == dtype
    assert test_BatchNorm.moving_mean.dtype == dtype
    assert test_BatchNorm.moving_var.dtype == dtype
    assert test_BatchNorm.beta.dtype == dtype
    assert test_BatchNorm.dbeta.dtype == dtype

    assert (test_BatchNorm.mu == tf_batch.mu.numpy()).all()
    assert (test_BatchNorm.var == tf_batch.var.numpy()).all()
    assert np.allclose(test_BatchNorm.moving_var, tf_batch.moving_var.numpy(), rtol=1e-2)
    assert np.allclose(test_BatchNorm.moving_mean, tf_batch.moving_mean.numpy(), rtol=1e-2)
    assert (test_BatchNorm.dbeta == dbeta_tf.numpy()).all()
    assert np.allclose(res_forward_np, y.numpy(), rtol=1e-3)
    assert np.allclose(res_grads_np, res_grads_tf.numpy(), rtol=1e-3)