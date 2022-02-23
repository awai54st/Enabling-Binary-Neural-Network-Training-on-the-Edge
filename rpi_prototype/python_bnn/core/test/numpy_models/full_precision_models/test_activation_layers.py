import numpy as np
import tensorflow as tf
from core.numpy_models.full_precision_models.activation_layers import StraightThroughActivation


def test_StraightThroughActivation():
    test_input = np.ones((5, 10, 10, 3))
    
    # numpy implementation
    test_StraightThroughActivation = StraightThroughActivation()
    res_forward_np = test_StraightThroughActivation.forward(test_input)
    res_grads_np = test_StraightThroughActivation.backprop(output_grad=1)
    
    # tensorflow implementation
    z = tf.Variable(test_input, dtype = tf.float32)
    with tf.GradientTape() as tape:
        y = tf.keras.activations.linear(z)
    res_grads_tf = tape.gradient(y, z)
    
    assert np.allclose(res_forward_np, y.numpy())
    assert np.allclose(res_grads_np, res_grads_tf.numpy())